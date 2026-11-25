# ANM-DCRNN

Basic PyTorch implementation of a graph recurrent encoder-decoder with fixed and learned graph supports and an edge readout.

## Run

```bash
python -m pip install -r requirements.txt
python demo.py --epochs 3
```

The demo trains and evaluates on generated tensors. Case-study data, trained weights, optimization-based labels, and physical feasibility correction are not included.

## Usage

`model.py` defines `GraphRuleNet`; `demo.py` contains scaling, training, validation selection, and prediction.

| Input | Shape |
| --- | --- |
| `adjacency` | `[N, N]` |
| `edges` | `[E, 2]` |
| `edge_features` | `[E, Fe]` |
| `history` | `[B, T, N, F]` |
| `forecasts` | `[B, H, N, Ff]` |
| `targets` / output | `[B, H, E]` |

Here, `B`, `T`, `H`, `N`, and `E` denote batch size, history length, forecast horizon, node count, and edge count; `F`, `Ff`, and `Fe` are feature dimensions.

Fit scalers on training data and reuse them for validation and testing. Forecasts must be available at the prediction origin. `alpha` scales the learned graph support; both fixed supports remain active.

## References

- Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR. https://openreview.net/forum?id=SJiHXGWAZ
- Wu et al. (2019), *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI. https://doi.org/10.24963/ijcai.2019/264
