"""Run the basic learning workflow using tensors generated in memory."""
import argparse
import json
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import GraphRuleNet

class Standardizer:
    def fit(self, training):
        axes = tuple(range(training.ndim - 1))
        self.mean = training.mean(axes, keepdim=True)
        self.scale = training.std(axes, keepdim=True, unbiased=False).clamp_min(1e-6)
        return self

    def transform(self, values):
        return (values - self.mean.to(values)) / self.scale.to(values)

    def inverse_transform(self, values):
        return values * self.scale.to(values) + self.mean.to(values)


@torch.no_grad()
def predict(model, history, forecasts):
    model.eval()
    return model(history, forecasts)


def fit(model, train_loader, validation_loader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, log = float("inf"), None, []
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        for history, forecasts, target in train_loader:
            history, forecasts, target = (v.to(device) for v in (history, forecasts, target))
            optimizer.zero_grad()
            loss = (model(history, forecasts) - target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        error, count = 0.0, 0
        with torch.no_grad():
            for history, forecasts, target in validation_loader:
                history, forecasts, target = (v.to(device) for v in (history, forecasts, target))
                error += (model(history, forecasts) - target).square().sum().item()
                count += target.numel()
        validation_loss = error / count
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, deepcopy(model.state_dict())
        log.append({"epoch": epoch + 1, "validation_mse": validation_loss})
    model.load_state_dict(best_state)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    nodes = 5
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]])
    adjacency = torch.zeros(nodes, nodes)
    adjacency[edges[:, 0], edges[:, 1]] = 1
    edge_features = torch.rand(len(edges), 2)
    history = torch.randn(24, 4, nodes, 3)
    forecasts = torch.randn(24, 3, nodes, 2)
    targets = (0.4 * history[:, -1, edges[:, 0], 0].unsqueeze(1)
               + 0.6 * forecasts[:, :, edges[:, 1], 0])
    xs = Standardizer().fit(history[:12])
    fs = Standardizer().fit(forecasts[:12])
    ys = Standardizer().fit(targets[:12])
    tensors = xs.transform(history), fs.transform(forecasts), ys.transform(targets)
    train = DataLoader(TensorDataset(*(t[:12] for t in tensors)), batch_size=4, shuffle=True)
    validation = DataLoader(TensorDataset(*(t[12:18] for t in tensors)), batch_size=4)
    model = GraphRuleNet(adjacency, edges, edge_features, input_dim=3, forecast_dim=2,
                         hidden_dim=12, embedding_dim=4)
    log = fit(model, train, validation, epochs=args.epochs)
    output = ys.inverse_transform(predict(model, tensors[0][18:], tensors[1][18:]))
    assert output.shape == targets[18:].shape and torch.isfinite(output).all()
    print(json.dumps({"purpose": "software demonstration with generated tensors",
                      "epochs": log, "prediction_shape": list(output.shape),
                      "demonstration_mse": ((output - targets[18:]) ** 2).mean().item()}, indent=2))


if __name__ == "__main__":
    main()
