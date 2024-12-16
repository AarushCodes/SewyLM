# SewyLM

SewyLM is a transformer model made by incorporating the following:
- Gemma2
- nGPT
- Differential Transformer
- LOCONUT (Limited COCONUT) (a varitation of COCONUT)
- NeuTRENO
- Muon optimizer
## FYI im a 15 yr old 
## It is compatible with the huggingface ecosystem.
## Installation

You can install SewyLM using pip:

```bash
pip install git+https://github.com/AarushCodes/SewyLM/
```

## Usage

Here is a simple example of how to use SewyLM:

```python
from sewylm.SewyModel import SEWYForCausalLM , SEWYConfig
config = SEWYConfig(Add ur config here)
model = SEWYForCausalLM(config)
print(model)
```

## I have also made a modified version of [Moun Optimizer](https://github.com/KellerJordan/Muon/tree/master) in accordance with nGPT paper and few optimizations.
```python
from sewylm import nMoun
## use as pytorch optimizer and also pass model to it like model=model
```
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the GNU GPL 3 License - see the [LICENSE](LICENSE) file for details.

## Author

Aarush Khilosia  
Email: aarushlikesllms@gmail.com

## Credit to the respective authors 
