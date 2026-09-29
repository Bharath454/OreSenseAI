"""
OreSense AI – Federated Learning Client
========================================
Each site trains a local shortfall prediction model on its OWN synthetic
data and only shares model weights (not raw data) with the FL server.

Site-specific data: slight variations in seasonality, equipment fleet,
and crew size to simulate different mine properties.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

try:
    import flwr as fl
    import torch
    import torch.nn as nn
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False

from simulators.production_sim import generate_production_records
from models.shortfall_model import _build_features, FEATURE_COLS

SITE_NAME = os.getenv("SITE_NAME", "Balaghat")
FL_SERVER = os.getenv("FL_SERVER", "fl_server:8080")
FL_ROUNDS = int(os.getenv("FL_ROUNDS", "5"))

# Site-specific random seed offsets (creates different local datasets)
SITE_SEEDS = {"Balaghat": 42, "Nagpur": 1337, "Gumgaon": 2025}


class ShortfallNet(nn.Module):
    """Small neural network for federated training."""
    def __init__(self, n_features: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    state = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state, strict=True)


def get_site_data() -> Tuple[np.ndarray, np.ndarray]:
    seed = SITE_SEEDS.get(SITE_NAME, 42)
    df_raw = generate_production_records(days=365, seed=seed)
    df = _build_features(df_raw)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["shortfall_pct"].values.astype(np.float32)
    return X, y


def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> Tuple[float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(loss_fn(model(X_t), y_t))
        # R2 as accuracy proxy
        ss_res = float(((y_t - model(X_t)) ** 2).sum())
        ss_tot = float(((y_t - y_t.mean()) ** 2).sum())
        r2 = max(0.0, 1 - ss_res / (ss_tot + 1e-9))
    return r2, len(X)


class OreSenseFlowerClient(fl.client.NumPyClient if HAS_FLWR else object):
    def __init__(self):
        self.model = ShortfallNet()
        self.X, self.y = get_site_data()
        print(f"[FL Client:{SITE_NAME}] Loaded {len(self.X)} samples")

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        accuracy, n = train_local(self.model, self.X, self.y)
        return get_weights(self.model), n, {"accuracy": accuracy, "site": SITE_NAME}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()
        X_t = torch.tensor(self.X)
        y_t = torch.tensor(self.y)
        with torch.no_grad():
            loss = float(nn.MSELoss()(self.model(X_t), y_t))
            ss_res = float(((y_t - self.model(X_t)) ** 2).sum())
            ss_tot = float(((y_t - y_t.mean()) ** 2).sum())
            accuracy = max(0.0, 1 - ss_res / (ss_tot + 1e-9))
        return loss, len(self.X), {
            "accuracy": accuracy,
            "num_examples": len(self.X),
            "site": SITE_NAME,
        }


def main():
    if not HAS_FLWR:
        print(f"[FL Client:{SITE_NAME}] Flower not available – sleeping")
        while True:
            time.sleep(30)
        return

    print(f"[FL Client:{SITE_NAME}] Connecting to {FL_SERVER}")
    # Retry until server is ready
    for attempt in range(30):
        try:
            fl.client.start_numpy_client(
                server_address=FL_SERVER,
                client=OreSenseFlowerClient(),
            )
            break
        except Exception as e:
            print(f"[FL Client:{SITE_NAME}] Attempt {attempt}: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
