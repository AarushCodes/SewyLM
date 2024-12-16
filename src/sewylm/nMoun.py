import os
import torch
import torch.distributed as dist
from torch.cuda.amp import custom_fwd, custom_bwd
import torch._dynamo as dynamo

# Custom autograd function for better memory efficiency
class NewtonSchulz(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, G, steps=10, eps=1e-7):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps)
        transpose = G.size(0) > G.size(1)
        if transpose:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = torch.addcmul(c * (A @ A), b, A)
            X = torch.addmm(alpha=a, mat1=B, mat2=X, input=X)
        if transpose:
            X = X.T
        return X.to(G.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output, None, None

@torch.compile(mode="reduce-overhead", fullgraph=True)
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    return NewtonSchulz.apply(G, steps, eps)

class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0, model=None):
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                       adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                       adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Pre-allocate states and buffers
        self.muon_params = []
        self.adamw_params = []
        if model is None:
            raise ValueError("Muon optimizer requires a model to be passed.")
        self.model = model        
        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = True
                self.muon_params.append(p)
                self.state[p]['momentum_buffer'] = torch.zeros_like(p)
            else:
                self.state[p]['use_muon'] = False
                self.adamw_params.append(p)
                self._init_adamw_state(p)

        for p in adamw_params:
            self.state[p]['use_muon'] = False
            self.adamw_params.append(p)
            self._init_adamw_state(p)

        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

        # Pre-allocate flat buffer for distributed updates
        self.total_muon_params = sum(p.numel() for p in self.muon_params)
        self.updates_flat = torch.zeros(self.total_muon_params, device='cuda', dtype=torch.bfloat16)

    def _init_adamw_state(self, p):
        state = self.state[p]
        state['step'] = 0
        state['moment1'] = torch.zeros_like(p)
        state['moment2'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon update
            if self.muon_params:
                self._muon_step(group)

            # AdamW update
            if self.adamw_params:
                self._adamw_step(group)
                
        self.normalize_model()

        return loss

    def _muon_step(self, group):
        lr = group['lr']
        momentum = group['momentum']
        self.updates_flat.zero_()
        
        curr_idx = 0
        for i, p in enumerate(self.muon_params):
            if i % self.world_size == self.rank:
                g = p.grad.view(p.size(0), -1) if p.grad.ndim > 2 else p.grad
                buf = self.state[p]['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                g.mul_(max(1, g.size(0)/g.size(1))**0.5)
                self.updates_flat[curr_idx:curr_idx+p.numel()].copy_(g.flatten())
            curr_idx += p.numel()

        if self.world_size > 1:
            dist.all_reduce(self.updates_flat, op=dist.ReduceOp.SUM)

        curr_idx = 0
        for p in self.muon_params:
            numel = p.numel()
            p.add_(self.updates_flat[curr_idx:curr_idx+numel].view_as(p), alpha=-lr)
            curr_idx += numel

    def _adamw_step(self, group):
        lr = group['adamw_lr_ratio'] * group['lr']
        beta1, beta2 = group['adamw_betas']
        eps = group['adamw_eps']
        weight_decay = group['adamw_wd']

        for p in self.adamw_params:
            if p.grad is None:
                continue
                
            state = self.state[p]
            state['step'] += 1
            
            # Update moments
            state['moment1'].lerp_(p.grad, 1-beta1)
            state['moment2'].lerp_(p.grad.square(), 1-beta2)

            # Compute bias correction
            bias_correction1 = 1 - beta1**state['step']
            bias_correction2 = 1 - beta2**state['step']
            step_size = lr * (bias_correction1 / bias_correction2**0.5)

            # Update parameters
            p.mul_(1 - lr * weight_decay).addcdiv_(
                state['moment1'],
                (state['moment2'].sqrt() + eps),
                value=-step_size
            )

    def normalize_model(self):
        with torch.no_grad():
            # Normalize input and output embeddings
            self.model.embed_tokens.weight.data = torch.nn.functional.normalize(
                self.model.embed_tokens.weight.data, dim=1
            )
            self.lm_head.weight.data = torch.nn.functional.normalize(
                self.lm_head.weight.data, dim=1
            )
            for layer in self.model.layers:
                # Normalize Q,K,V projection matrices
                layer.self_attn.q_proj.weight.data = torch.nn.functional.normalize(
                    layer.self_attn.q_proj.weight.data, dim=1
                )
                layer.self_attn.k_proj.weight.data = torch.nn.functional.normalize(
                    layer.self_attn.k_proj.weight.data, dim=1
                )
                layer.self_attn.v_proj.weight.data = torch.nn.functional.normalize(
                    layer.self_attn.v_proj.weight.data, dim=1
                )
                
                # Normalize output projection
                layer.self_attn.o_proj.weight.data = torch.nn.functional.normalize(
                    layer.self_attn.o_proj.weight.data, dim=1
                )

                # Normalize MLP matrices
                layer.mlp.gate_proj.weight.data = torch.nn.functional.normalize(
                    layer.mlp.gate_proj.weight.data, dim=1
                )
                layer.mlp.up_proj.weight.data = torch.nn.functional.normalize(
                    layer.mlp.up_proj.weight.data, dim=1
                )
                layer.mlp.down_proj.weight.data = torch.nn.functional.normalize(
                    layer.mlp.down_proj.weight.data, dim=1
                )