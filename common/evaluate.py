"""Official WILDS iWildCam metrics and the linear-probe baseline."""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from tqdm import tqdm

from common import resolve


def wilds_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Compute accuracy, macro recall and macro F1 exactly as WILDS does for iWildCam.

    Macro averages run over the classes present in `y_true` of the evaluated
    split; classes that are only predicted are ignored.

    Args:
        y_true: True class indices [N].
        y_pred: Predicted class indices [N].

    Returns:
        Dict with keys `n` (number of samples), `acc`, `recall_macro` and `f1_macro`.
    """
    labels = np.unique(y_true)
    return {
        "n": len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


class LinearProbe(nn.Module):
    """Standardized features followed by a linear layer (logistic regression).

    Like scikit-learn, it only learns weights for the classes seen in training
    (`classes`); all other classes get logit -inf, i.e. probability 0.

    Args:
        mean: Per-feature mean of the training features [D].
        std: Per-feature standard deviation of the training features [D].
        classes: Sorted class indices present in the training labels [K].
        num_classes: Total number of classes (output size).
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, classes: torch.Tensor, num_classes: int):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("classes", classes)
        self.num_classes = num_classes
        self.linear = nn.Linear(len(mean), len(classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map raw features [N, D] to logits [N, num_classes] (-inf for unseen classes)."""
        logits = features.new_full((len(features), self.num_classes), float("-inf"))
        logits[:, self.classes] = self.linear((features - self.mean) / self.std)
        return logits

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> np.ndarray:
        """Class probabilities [N, num_classes] for raw features [N, D] (on any device)."""
        return torch.softmax(self(features.to(self.mean.device)), dim=1).cpu().numpy()


def load_probe(cfg: dict, device: torch.device) -> LinearProbe:
    """Load the linear probe saved by `make baselines`.

    Args:
        cfg: Shared config; uses `paths.probe_checkpoint` and `model.num_classes`.
        device: Target device.

    Returns:
        The probe in eval mode; combine it with `load_imagenet_backbone` for a full
        image -> logits model (see `common.model.load_probe_model`).
    """
    state = torch.load(resolve(cfg["paths"]["probe_checkpoint"]), map_location="cpu", weights_only=True)
    probe = LinearProbe(state["mean"], state["std"], state["classes"], cfg["model"]["num_classes"])
    probe.load_state_dict(state)
    return probe.to(device).eval()


def fit_linear_probe(cfg: dict, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> LinearProbe:
    """Train a multinomial logistic regression on frozen features, on the GPU.

    Same model and objective as scikit-learn's `StandardScaler` +
    `LogisticRegression(C)`: one output per class seen in training, mean
    cross-entropy + ||W||^2 / (2 * C * N), bias not regularized, minimized with
    full-batch L-BFGS from zero weights until the largest gradient entry is
    below `tol` or `max_iter` iterations ran.

    Args:
        cfg: Shared config; uses `linear_probe.C`, `.max_iter`, `.tol` and `model.num_classes`.
        features: Training features [N, D], e.g. from `extract_features`.
        labels: Training class indices [N].
        device: Device to train on.

    Returns:
        The fitted probe on `device`, in eval mode.
    """
    params = cfg["linear_probe"]
    x = features.to(device)
    classes, y = torch.unique(labels.to(device), return_inverse=True)  # y: position in `classes`
    mean, std = x.mean(dim=0), x.std(dim=0, unbiased=False)
    std[std == 0] = 1.0  # constant features, as in StandardScaler
    x = (x - mean) / std

    probe = LinearProbe(mean, std, classes, cfg["model"]["num_classes"]).to(device)
    nn.init.zeros_(probe.linear.weight)
    nn.init.zeros_(probe.linear.bias)
    l2 = 1.0 / (2.0 * params["C"] * len(x))
    # All iterations run in one step() call: L-BFGS budgets its line search from
    # max_eval, so calling it one iteration at a time leaves the line search no
    # evaluations and the fit stalls. max_eval matches scipy's default (15000 for 1000 iterations).
    opt = torch.optim.LBFGS(
        probe.linear.parameters(),
        max_iter=params["max_iter"],
        max_eval=15 * params["max_iter"],
        tolerance_grad=params["tol"],
        line_search_fn="strong_wolfe",
    )

    def loss_fn() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(probe.linear(x), y) + l2 * probe.linear.weight.pow(2).sum()
        loss.backward()
        return loss

    with tqdm(total=params["max_iter"], desc="probe: fitting", unit="iter") as bar:

        def closure() -> torch.Tensor:  # called by L-BFGS for every evaluation; updates the bar
            loss = loss_fn()
            bar.update(opt.state[probe.linear.weight].get("n_iter", 0) - bar.n)
            bar.set_postfix(loss=f"{loss.item():.4f}")
            return loss

        opt.step(closure)

    final_loss = loss_fn().item()
    grad_max = max(p.grad.abs().max().item() for p in probe.linear.parameters())
    n_iter = opt.state[probe.linear.weight]["n_iter"]
    print(f"Linear probe: {n_iter} iterations, loss {final_loss:.4f}, max gradient {grad_max:.1e}")
    if grad_max > params["tol"]:
        print(f"WARNING: linear probe did not converge (max gradient {grad_max:.1e} > tol {params['tol']:.0e}); "
              "increase linear_probe.max_iter")
    return probe.eval()
