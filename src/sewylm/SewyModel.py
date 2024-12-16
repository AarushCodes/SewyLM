# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, HybridCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal,
    logging,
)
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaModel,
    GemmaPreTrainedModel,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
try:
    from torch.nn.attention.flex_attention import flex_attention
except:
    pass


logger = logging.get_logger(__name__)


class SEWYConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SEWYModel`]. It is used to instantiate an SEWY
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SEWY-7B.
    e.g. [google/SEWY-7b](https://huggingface.co/google/SEWY-7b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the SEWY model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SEWYModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256): scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in SEWY, every other layer uses sliding window attention. This is the
            size of the sliding window.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

        coconut_iter = the number of iterations for the Training Large Language Models to Reason in a
Continuous Latent Space
        n_coconut_layers = the number of last n layers to apply the coconut to. (default is 4)

    ```python
    >>> from transformers import SEWYModel, SEWYConfig
    >>> # Initializing a SEWY SEWY-7b style configuration
    >>> configuration = SEWYConfig()
    >>> # Initializing a model from the SEWY-7b style configuration
    >>> model = SEWYModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "SEWY"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="relu2",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        cache_implementation="hybrid",
        coconut_iter=4,
        init_neutreno_lambda=2.0,
        n_coconut_layers=4,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation
        self.coconut_iter = coconut_iter    
        self.init_neutreno_lambda = float(init_neutreno_lambda)
        self.n_coconut_layers = int(n_coconut_layers)
        


class SEWYRMSNorm(GemmaRMSNorm):
    pass


class SEWYMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        # Add learnable scaling factors
        self.s_u = nn.Parameter(torch.ones(self.intermediate_size))
        self.s_nu = nn.Parameter(torch.ones(self.intermediate_size))
        
        # Store init and scale values
        self.s_u_init = 1.0
        self.s_u_scale = 1.0  # As specified
        
        self.s_nu_init = 1.0
        self.s_nu_scale = 1.0  # As specified
    def normalize(self, x):
        """Layer normalization without affine transformation"""
        return x / torch.norm(x, dim=-1, keepdim=True)

    def forward(self, x):
        # Get intermediate states
        u = self.gate_proj(x)  # First projection 
        nu = self.up_proj(x)   # Second projection
        
        # Apply scaling factors
        s_u_actual = self.s_u * (self.s_u_init / self.s_u_scale) 
        s_nu_actual = self.s_nu * (self.s_nu_init / self.s_nu_scale)
        
        # Scale according to equations mlpscaling1 and mlpscaling2
        u = u * s_u_actual
        nu = nu * s_nu_actual * (self.config.hidden_size ** 0.5)  # Apply sqrt(d_model) scaling
        
        # Apply activation and combine
        activated = self.act_fn(nu)  # SiLU activation
        gated = u * activated  # Element-wise multiplication
        
        # Final projection
        return self.normalize(self.down_proj(gated))
        #return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SEWYRotaryEmbedding(GemmaRotaryEmbedding):
    pass


def eager_attention_forward(
    config: SEWYConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, config.num_key_value_groups)
    value_states = repeat_kv(value, config.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * config.scaling

    if config.attn_logit_softcapping is not None:
        attn_weights = attn_weights / config.attn_logit_softcapping
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * config.attn_logit_softcapping
    if mask is not None:  # no matter the length, we just slice it
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=config.attention_dropout, training=config.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def flash_attention_forward(
    config: SEWYConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> Tuple[torch.Tensor, None]:
    if mask is not None:
        seq_len = mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor rotary embedding
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    dropout_rate = config.attention_dropout if config.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        mask,
        seq_len,
        dropout=dropout_rate,
        softmax_scale=config.scaling,
        is_causal=config.is_causal,
        sliding_window=config.sliding_window,
        use_top_left_mask=config._flash_attn_uses_top_left_mask,
        softcap=config.attn_logit_softcapping if is_flash_attn_greater_or_equal("2.6.0") else None,
    )

    return attn_output, None


def flex_attention_forward(
    config: SEWYConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    output_attentions: bool = False,
    **_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    def tanh_softcap(score, b, h, q_idx, kv_idx):
        soft_cap = config.attn_logit_softcapping
        score = soft_cap * torch.tanh(score / soft_cap)
        if mask is not None:
            return score + mask[b][0][q_idx][kv_idx]
        return score

    attn_output = flex_attention(
        query,
        key,
        value,
        score_mod=tanh_softcap,
        enable_gqa=True,
        scale=config.scaling,
        return_lse=output_attentions,
    )
    if not output_attentions:
        attn_weights = None
    else:
        attn_output, attn_weights = attn_output

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    config: SEWYConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    **_kwargs,
) -> Tuple[torch.Tensor, None]:
    key = repeat_kv(key, config.num_key_value_groups)
    value = repeat_kv(value, config.num_key_value_groups)

    causal_mask = mask
    if mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query.device.type == "cuda" and causal_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and query.shape[1] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=config.attention_dropout if config.training else 0.0,
        is_causal=is_causal,
        scale=config.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


SEWY_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class SEWYAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)
    def __init__(self, config: SEWYConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None
        self.attn_logit_softcapping = config.attn_logit_softcapping
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = SEWYRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


        #### NGPT #####


          # Change scaling factor from 1/sqrt(d_k) to sqrt(d_k)
        self.scaling = self.head_dim ** 0.5  # sqrt(d_k) instead of 1/sqrt(d_k)
        
        # Add learnable scaling factors for query/key vectors
        self.s_qk = nn.Parameter(
            torch.ones(self.head_dim)  # Initialize to ones
        )
        
        # Store init and scale values
        self.s_qk_init = 1.0
        self.s_qk_scale = 1.0 / (config.hidden_size ** 0.5)



        ### DIFF TRANSFORMER ###
        self.n_rep = self.num_heads // self.num_key_value_heads
        # print(layer_idx)
        self.lambda_init = self.lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0,std=0.1))

        self.subln = torch.nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        
    def normalize(self, x):
        """Layer normalization without affine transformation"""
        return x / torch.norm(x, dim=-1, keepdim=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        formal_layer_values: Optional[list] = None,
        neutreno_lambda: Optional[nn.Parameter] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.layer_idx == 0:
            formal_layer_values.append(value_states)
        else:
            current_value = value_states
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        ### DIFF TRANSFORMERR ####
        query_states = query_states.reshape(bsz, q_len, self.num_heads, 2, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)



        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.normalize(query_states)
        key_states = self.normalize(key_states)
        s_qk_actual = self.s_qk * (self.s_qk_init / self.s_qk_scale)
        query_states = query_states * s_qk_actual.view(1, 1, 1, -1)
        key_states = key_states * s_qk_actual.view(1, 1, 1, -1)
        

        ### DIFF TRANSFORMERR ####
        # query_states = query_states.reshape(bsz, q_len, self.num_heads, 2, self.head_dim)
        # key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)
        q1, q2 = query_states[:, :, :, 0], query_states[:, :, :, 1]
        k1, k2 = key_states[:, :, :, 0], key_states[:, :, :, 1]
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if output_attentions and self.config._attn_implementation in ["sdpa", "flash_attention_2"]:
            logger.warning_once("Setting `attention_type` to `flex_attention` because `output_attentions=True`")
            attention_type = "flex_attention"
        else:
            attention_type = self.config._attn_implementation

        attn1 , attn_weights = SEWY_ATTENTION_FUNCTION[attention_type](
            self, q1, k1, value_states, attention_mask, output_attentions=output_attentions
        )

        attn2 , attn_weights = SEWY_ATTENTION_FUNCTION[attention_type](
            self, q2, k2, value_states, attention_mask, output_attentions=output_attentions
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, q_len, self.num_heads * 2 * self.head_dim)
        


        ### neutreno
        if self.layer_idx != 0:
            attn_output = attn_output + neutreno_lambda*(formal_layer_values[0]-current_value)
        attn = self.out_proj(attn)
        # attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        # attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn, attn_weights, past_key_value, formal_layer_values


class SEWYFlashAttention2(SEWYAttention):
    def __init__(self, config: SEWYConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config._attn_implementation = "flash_attention_2"
        logger.warning_once(
            "The `SEWYFlashAttention2` class is deprecated in favor of simply modifying the `config._attn_implementation`"
            "attribute of the `GemmaAttention` class! It will be removed in v4.48"
        )


class SEWYSdpaAttention(SEWYAttention):
    def __init__(self, config: SEWYConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config._attn_implementation = "sdpa"
        logger.warning_once(
            "The `SEWYFlashAttention2` class is deprecated in favor of simply modifying the `config._attn_implementation`"
            "attribute of the `GemmaAttention` class! It will be removed in v4.48"
        )

class SEWYDecoderLayer(nn.Module):
    def __init__(self, config: SEWYConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.is_sliding = not bool(layer_idx % 2)
        self.self_attn = SEWYAttention(config=config, layer_idx=layer_idx)
        self.mlp = SEWYMLP(config)
        # self.input_layernorm = SEWYRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = SEWYRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # self.pre_feedforward_layernorm = SEWYRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_feedforward_layernorm = SEWYRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

        ### NGPT ###
        self.alpha_A = nn.Parameter(
            torch.full((config.hidden_size,), 0.05/config.num_hidden_layers)
        )
        self.alpha_M = nn.Parameter(
            torch.full((config.hidden_size,), 0.05/config.num_hidden_layers) 
        )
        
        # Add alpha scaling factor
        self.alpha_scale = 1.0 / (config.hidden_size ** 0.5)
        # formal_layer_values: Optional[list] = None,
        self.neutreno_lambda = nn.Parameter(torch.tensor(self.config.init_neutreno_lambda))
    def normalize(self, x):
        """Layer normalization without affine transformation"""
        return x / torch.norm(x, dim=-1, keepdim=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, formal_layer_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            neutreno_lambda=self.neutreno_lambda,
            formal_layer_values=formal_layer_values,
        )
        attn_output = self.normalize(attn_output)

        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.normalize(mlp_output)
        # hidden_states = self.post_feedforward_layernorm(hidden_states)
        alpha_M = self.alpha_M * self.alpha_scale  
        delta_M = mlp_output - residual
        hidden_states = self.normalize(residual + alpha_M.unsqueeze(1) * delta_M)
        #hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, formal_layer_values


class SEWYPreTrainedModel(GemmaPreTrainedModel):
    _supports_quantized_cache = False

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False):
        """
        Overloads `PreTrainedModel._check_and_enable_sdpa` so as to DISABLE torch SDPA by default on SEWY models.
        SDPA reduces the model performance on SEWY because of the logits softcapping.
        """
        config = super()._check_and_enable_sdpa(config, hard_check_only=hard_check_only)

        # if using the default path -> swap sdpa by eager
        if not hard_check_only and config._attn_implementation == "sdpa":
            config._attn_implementation = "eager"

        return config


class SEWYModel(GemmaModel, SEWYPreTrainedModel):
    def __init__(self, config: SEWYConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [SEWYDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # SEWY downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        formal_layer_values = []
        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)

        #     if self.gradient_checkpointing and self.training:
        #         layer_outputs  = self._gradient_checkpointing_func(
        #             decoder_layer.__call__,
        #             hidden_states,
        #             causal_mask,
        #             position_ids,
        #             past_key_values,
        #             output_attentions,
        #             use_cache,
        #             cache_position,
        #         )
        #     else:
        #         layer_outputs , formal_layer_values = decoder_layer(
        #             hidden_states,
        #             attention_mask=causal_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_values,
        #             output_attentions=output_attentions,
        #             use_cache=use_cache,
        #             cache_position=cache_position,
        #             formal_layer_values=formal_layer_values,
        #         )

        #     hidden_states = layer_outputs[0]

        #     if output_attentions:
        #         all_self_attns += (layer_outputs[1],)

        for decoder_layer in self.layers[: self.config.num_hidden_layers-(self.config.n_coconut_layers)]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs  = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs , formal_layer_values = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    formal_layer_values=formal_layer_values,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        for _ in range(0, self.config.coconut_iter):
            for decoder_layer in self.layers[self.config.num_hidden_layers-self.config.n_coconut_layers: self.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs  = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs , formal_layer_values = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        formal_layer_values=formal_layer_values,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)




        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but SEWY work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


class SEWYForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SEWYModel(config)
        # Add learnable scaling parameter for logits
        self.s_z = nn.Parameter(
            torch.ones(config.vocab_size)
        )
        
        # Store init and scale values
        self.s_z_init = 1.0
        self.s_z_scale = 1.0 / (config.hidden_size ** 0.5)

        self.post_init()
    def normalize_embedding_matrices(self):
        """
        Normalizes all matrices along their embedding dimension after each training step.
        Should be called after optimizer.step()
        """
        with torch.no_grad():
            # Normalize input and output embeddings
            self.model.embed_tokens.weight.data = torch.nn.functional.normalize(
                self.model.embed_tokens.weight.data, dim=1
            )
            self.lm_head.weight.data = torch.nn.functional.normalize(
                self.lm_head.weight.data, dim=1
            )

            # Normalize attention matrices in each layer
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
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
    
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train SEWY models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    



        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        s_z_actual = self.s_z * (self.s_z_init / self.s_z_scale)
        logits = logits * s_z_actual

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride
                # during the decoding. Here, simply using `.contiguous()` is not sufficient as in the
                # batch size = 1 case, `position_ids` is already contiguous but with varying stride
                # which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class SEWYForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = SEWYModel(config)
        self.post_init()


class SEWYForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = SEWYModel(config)
        self.post_init()