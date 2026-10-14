"""Step 4 - Unsupervised anomaly detectors.

All detectors follow the same contract:

    model.fit(X_train_normal, X_val_normal, seed)
    scores = model.score(X)      # higher score = more anomalous

They only ever see NORMAL data during fitting. The StandardScaler lives inside
each model and is fitted on the training split only, so validation and test
statistics never influence preprocessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # the rest of the project works without PyTorch
    TORCH_AVAILABLE = False


class IsolationForestDetector:
    """Isolates points with random splits; anomalies are isolated in fewer splits."""

    name = "IsolationForest"
    representation = "features"

    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators

    def fit(self, X_train, X_val=None, seed: int = 0):
        self.pipe = make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=self.n_estimators, random_state=seed),
        ).fit(X_train)
        return self

    def score(self, X):
        return -self.pipe.decision_function(X)


class OneClassSVMDetector:
    """Learns a boundary that encloses most normal training points (RBF kernel)."""

    name = "OneClassSVM"
    representation = "features"

    def __init__(self, nu: float = 0.05, gamma="scale"):
        self.nu, self.gamma = nu, gamma

    def fit(self, X_train, X_val=None, seed: int = 0):  # deterministic: seed unused
        self.pipe = make_pipeline(StandardScaler(), OneClassSVM(nu=self.nu, gamma=self.gamma)).fit(X_train)
        return self

    def score(self, X):
        return -self.pipe.decision_function(X)


class SpectralAutoencoder:
    """Small fully-connected autoencoder on the pooled log-spectrum.

    It is trained to reconstruct normal spectra. A faulty bearing adds
    spectral content the network never learned, so the reconstruction error
    (mean squared error per window) becomes the anomaly score.
    Early stopping uses the reconstruction loss on VALIDATION NORMAL data only.
    """

    name = "SpectralAutoencoder"
    representation = "spectrum"

    def __init__(self, hidden=(64, 16), epochs=150, batch_size=64, lr=1e-3,
                 weight_decay=1e-5, patience=15):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the autoencoder.")
        self.hidden, self.epochs, self.batch_size = list(hidden), epochs, batch_size
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience

    def _build(self, n_in: int) -> "nn.Module":
        dims = [n_in] + self.hidden
        enc, dec = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        rev = dims[::-1]
        for i, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
            dec += [nn.Linear(a, b)] + ([nn.ReLU()] if i < len(rev) - 2 else [])
        return nn.Sequential(*enc, *dec)

    def _tensor(self, X):
        return torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)

    def fit(self, X_train, X_val, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.scaler = StandardScaler().fit(X_train)
        xtr, xva = self._tensor(X_train), self._tensor(X_val)
        self.net = self._build(xtr.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)

        best, best_state, bad_epochs = float("inf"), None, 0
        self.history = []
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(xtr), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss = loss_fn(self.net(xtr[idx]), xtr[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(xva), xva).item()
            self.history.append(val_loss)
            if val_loss < best - 1e-6:
                best, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def score(self, X):
        self.net.eval()
        x = self._tensor(X)
        with torch.no_grad():
            return ((self.net(x) - x) ** 2).mean(dim=1).numpy()


class RMSBaseline:
    """Simple anomaly detector using RMS vibration amplitude."""

    name = "RMSBaseline"
    representation = "features"

    def fit(self, X_train, X_val=None, seed=0):
        return self

    def score(self, X):
        # RMS is the third handcrafted feature (index 2)
        return X[:, 2]


def build_models(cfg: dict) -> list:
    m = cfg["models"]
    models = [
        RMSBaseline(),
        IsolationForestDetector(**m["isolation_forest"]),
        OneClassSVMDetector(**m["one_class_svm"]),
    ]
    ae_cfg = dict(m["autoencoder"])
    if ae_cfg.pop("enabled", True):
        if TORCH_AVAILABLE:
            models.append(SpectralAutoencoder(**ae_cfg))
        elif not getattr(build_models, "_warned", False):
            print("[warning] PyTorch not installed -> autoencoder skipped.")
            build_models._warned = True
    return models
