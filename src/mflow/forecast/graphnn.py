"""Spatio-temporal graph neural network baselines.

Three architectures, trained per site on a local training window, with the graph taken
from ``edges.csv`` rather than learned from correlations:

* **DCRNN** -- Li, Yu, Shahabi and Liu (2018), *Diffusion Convolutional Recurrent Neural
  Network: Data-Driven Traffic Forecasting*, ICLR. A GRU whose matrix multiplications are
  replaced by K-step random-walk diffusion over the forward and backward transition
  matrices.
* **STGCN** -- Yu, Yin and Zhu (2018), *Spatio-Temporal Graph Convolutional Networks*,
  IJCAI. Sandwiched gated temporal convolutions around a Chebyshev spectral graph
  convolution.
* **Graph WaveNet** -- Wu, Pan, Long, Jiang and Zhang (2019), IJCAI. Dilated causal
  convolutions with a learned adaptive adjacency added to the fixed one.

All three share the same training harness and the same head: a pinball loss over the nine
quantile levels, so they produce genuine predictive intervals rather than point forecasts
with an assumed noise model. Training is seeded and single-threaded by default so that
two runs with the same manifest give identical numbers.

The graph they run on is the series-level operator from
:meth:`mflow.graph.BuildingGraph.series_adjacency`, which covers occupancy and flow
series alike; running these baselines on occupancy only would leave half the results
table empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastError,
    fill_context,
    validate_quantiles,
)
from mflow.graph import BuildingGraph
from mflow.profiling import measure
from mflow.schema import Panel

# --------------------------------------------------------------------------- #
# Graph operators
# --------------------------------------------------------------------------- #


def random_walk_transitions(adjacency: np.ndarray) -> tuple[Tensor, Tensor]:
    """Forward and backward random-walk transition matrices used by DCRNN."""
    matrix = np.asarray(adjacency, dtype=np.float32)
    forward = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-8)
    backward = matrix.T / np.maximum(matrix.T.sum(axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(forward), torch.from_numpy(backward)


def scaled_laplacian(adjacency: np.ndarray) -> Tensor:
    """Symmetrically normalised Laplacian rescaled to ``[-1, 1]`` for Chebyshev filters."""
    matrix = np.asarray(adjacency, dtype=np.float64)
    degree = matrix.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-8)), 0.0)
    normalised = np.eye(matrix.shape[0]) - (inv_sqrt[:, None] * matrix * inv_sqrt[None, :])
    largest = float(np.max(np.abs(np.linalg.eigvalsh(normalised))))
    largest = largest if largest > 0 else 2.0
    return torch.from_numpy(
        (2.0 * normalised / largest - np.eye(matrix.shape[0])).astype(np.float32)
    )


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #


class DiffusionConvolution(nn.Module):
    """K-step diffusion convolution over the forward and backward transitions."""

    def __init__(self, in_dim: int, out_dim: int, k_hops: int) -> None:
        super().__init__()
        self.k_hops = k_hops
        self.linear = nn.Linear(in_dim * (1 + 2 * k_hops), out_dim)

    def forward(self, x: Tensor, forward: Tensor, backward: Tensor) -> Tensor:
        """Apply the convolution to ``(batch, n_nodes, in_dim)``."""
        supports = [x]
        for transition in (forward, backward):
            state = x
            for _ in range(self.k_hops):
                state = torch.einsum("ij,bjf->bif", transition, state)
                supports.append(state)
        return self.linear(torch.cat(supports, dim=-1))


class DCRNNCell(nn.Module):
    """A GRU cell whose gates are diffusion convolutions."""

    def __init__(self, in_dim: int, hidden_dim: int, k_hops: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = DiffusionConvolution(in_dim + hidden_dim, 2 * hidden_dim, k_hops)
        self.candidate = DiffusionConvolution(in_dim + hidden_dim, hidden_dim, k_hops)

    def forward(self, x: Tensor, hidden: Tensor, forward: Tensor, backward: Tensor) -> Tensor:
        """Advance the hidden state by one step."""
        joined = torch.cat([x, hidden], dim=-1)
        reset, update = torch.chunk(
            torch.sigmoid(self.gates(joined, forward, backward)), 2, dim=-1
        )
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * hidden], dim=-1), forward, backward)
        )
        return update * hidden + (1.0 - update) * candidate


class DCRNNModule(nn.Module):
    """Encoder-only DCRNN with a direct multi-horizon quantile head."""

    def __init__(
        self, n_nodes: int, hidden_dim: int, k_hops: int, horizon: int, n_quantiles: int
    ) -> None:
        super().__init__()
        self.cell = DCRNNCell(1, hidden_dim, k_hops)
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.head = nn.Linear(hidden_dim, horizon * n_quantiles)
        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x: Tensor, forward: Tensor, backward: Tensor) -> Tensor:
        """Map ``(batch, n_nodes, context)`` to ``(batch, n_nodes, horizon, n_quantiles)``."""
        batch, n_nodes, context = x.shape
        hidden = torch.zeros(batch, n_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(context):
            hidden = self.cell(x[:, :, t : t + 1], hidden, forward, backward)
        out = self.head(hidden)
        return out.view(batch, n_nodes, self.horizon, self.n_quantiles)


class TemporalGatedConv(nn.Module):
    """Gated 1-D causal convolution along time (the STGCN temporal block)."""

    def __init__(self, in_dim: int, out_dim: int, kernel: int = 3) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_dim, 2 * out_dim, kernel_size=(1, kernel))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated convolution to ``(batch, channels, n_nodes, time)``."""
        padded = nn.functional.pad(x, (self.kernel - 1, 0))
        projected, gate = torch.chunk(self.conv(padded), 2, dim=1)
        return projected * torch.sigmoid(gate)


class ChebyshevConv(nn.Module):
    """Chebyshev spectral graph convolution of order ``k``."""

    def __init__(self, in_dim: int, out_dim: int, k_order: int) -> None:
        super().__init__()
        self.k_order = k_order
        self.weight = nn.Parameter(torch.empty(k_order, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        """Apply the convolution to ``(batch, channels, n_nodes, time)``."""
        signal = x.permute(0, 2, 3, 1)  # (batch, n_nodes, time, channels)
        terms = [signal]
        if self.k_order > 1:
            terms.append(torch.einsum("ij,bjtc->bitc", laplacian, signal))
        for k in range(2, self.k_order):
            terms.append(
                2.0 * torch.einsum("ij,bjtc->bitc", laplacian, terms[k - 1]) - terms[k - 2]
            )
        stacked = torch.stack(terms, dim=0)
        out = torch.einsum("kbitc,kcf->bitf", stacked, self.weight) + self.bias
        return out.permute(0, 3, 1, 2)


class STGCNModule(nn.Module):
    """Two spatio-temporal blocks followed by a quantile head."""

    def __init__(
        self, hidden_dim: int, k_order: int, horizon: int, n_quantiles: int
    ) -> None:
        super().__init__()
        self.temporal_1 = TemporalGatedConv(1, hidden_dim)
        self.spatial = ChebyshevConv(hidden_dim, hidden_dim, k_order)
        self.temporal_2 = TemporalGatedConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon * n_quantiles)
        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        """Map ``(batch, n_nodes, context)`` to ``(batch, n_nodes, horizon, n_quantiles)``."""
        batch, n_nodes, _ = x.shape
        signal = x.unsqueeze(1)  # (batch, 1, n_nodes, time)
        signal = self.temporal_1(signal)
        signal = torch.relu(self.spatial(signal, laplacian))
        signal = self.temporal_2(signal)
        last = signal[..., -1].permute(0, 2, 1)  # (batch, n_nodes, channels)
        out = self.head(self.norm(last))
        return out.view(batch, n_nodes, self.horizon, self.n_quantiles)


class GraphWaveNetModule(nn.Module):
    """Dilated causal convolutions with a learned adaptive adjacency."""

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        n_layers: int,
        horizon: int,
        n_quantiles: int,
        embedding_dim: int = 10,
    ) -> None:
        super().__init__()
        self.start = nn.Conv2d(1, hidden_dim, kernel_size=(1, 1))
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.dilations = [2**i for i in range(n_layers)]
        for dilation in self.dilations:
            self.filters.append(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=(1, dilation))
            )
            self.gates.append(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=(1, dilation))
            )
            self.graph_convs.append(nn.Linear(hidden_dim * 3, hidden_dim))
        self.source = nn.Parameter(torch.randn(n_nodes, embedding_dim) * 0.01)
        self.target = nn.Parameter(torch.randn(embedding_dim, n_nodes) * 0.01)
        self.head = nn.Linear(hidden_dim, horizon * n_quantiles)
        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x: Tensor, forward: Tensor, backward: Tensor) -> Tensor:
        """Map ``(batch, n_nodes, context)`` to ``(batch, n_nodes, horizon, n_quantiles)``."""
        batch, n_nodes, _ = x.shape
        adaptive = torch.softmax(torch.relu(self.source @ self.target), dim=1)
        signal = self.start(x.unsqueeze(1))
        skip = torch.zeros(
            batch, signal.shape[1], n_nodes, 1, device=x.device, dtype=x.dtype
        )

        for index, dilation in enumerate(self.dilations):
            # Left padding by the dilation keeps the convolution causal and the time
            # axis unchanged, so the residual connection needs no cropping.
            padded = nn.functional.pad(signal, (dilation, 0))
            residual = torch.tanh(self.filters[index](padded)) * torch.sigmoid(
                self.gates[index](padded)
            )
            hidden = residual.permute(0, 2, 3, 1)  # (batch, n_nodes, time, channels)
            diffused = torch.cat(
                [
                    torch.einsum("ij,bjtc->bitc", forward, hidden),
                    torch.einsum("ij,bjtc->bitc", backward, hidden),
                    torch.einsum("ij,bjtc->bitc", adaptive, hidden),
                ],
                dim=-1,
            )
            hidden = self.graph_convs[index](diffused).permute(0, 3, 1, 2)
            signal = signal + hidden
            skip = skip + hidden[..., -1:]

        last = torch.relu(skip)[..., -1].permute(0, 2, 1)
        out = self.head(last)
        return out.view(batch, n_nodes, self.horizon, self.n_quantiles)


# --------------------------------------------------------------------------- #
# Training harness
# --------------------------------------------------------------------------- #


def pinball_loss(prediction: Tensor, target: Tensor, levels: Tensor) -> Tensor:
    """Mean pinball (quantile) loss.

    Args:
        prediction: ``(batch, n_nodes, horizon, n_quantiles)``.
        target: ``(batch, n_nodes, horizon)``.
        levels: ``(n_quantiles,)``.
    """
    errors = target.unsqueeze(-1) - prediction
    return torch.mean(torch.maximum(levels * errors, (levels - 1.0) * errors))


class _GraphForecaster(Forecaster):
    """Shared training loop for the three graph architectures.

    Args:
        graph: the building graph; the adjacency is derived from it, never learned from
            correlations between the evaluation series.
        context_length: steps of history fed to the model.
        horizon: the horizon the model is trained for. A model trained for one horizon
            must not be asked for a longer one, so this is checked at predict time.
        epochs, batch_size, learning_rate, weight_decay: optimisation settings.
        validation_fraction: tail of the training panel held out for early stopping.
        patience: early-stopping patience in epochs.
        device: torch device; ``auto`` prefers CUDA, then MPS.
    """

    requires_training = True
    supports_multivariate = True
    supports_future_covariates = False

    def __init__(
        self,
        graph: BuildingGraph,
        *,
        context_length: int = 96,
        horizon: int = 60,
        hidden_dim: int = 32,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        validation_fraction: float = 0.15,
        patience: int = 8,
        device: str = "cpu",
        torch_threads: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.graph = graph
        self.context_length = context_length
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.device = device
        self.torch_threads = torch_threads
        self._module: nn.Module | None = None
        self._series_ids: list[str] = []
        self._scale: Tensor | None = None
        self._offset: Tensor | None = None
        self._operators: dict[str, Tensor] = {}
        self._history: list[dict[str, float]] = []

    # -- to be provided by the concrete architectures ------------------------- #

    def _build_module(self, n_series: int, n_quantiles: int) -> nn.Module:
        raise NotImplementedError

    def _build_operators(self, adjacency: np.ndarray) -> dict[str, Tensor]:
        raise NotImplementedError

    def _apply(self, module: nn.Module, batch: Tensor) -> Tensor:
        raise NotImplementedError

    # -- shared ---------------------------------------------------------------- #

    def _resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _windows(self, series: np.ndarray) -> tuple[Tensor, Tensor]:
        """Cut the training panel into (context, target) windows."""
        total = series.shape[1]
        span = self.context_length + self.horizon
        if total < span + 1:
            raise ForecastError(
                f"{self.name} needs at least {span + 1} steps of training data "
                f"(context {self.context_length} + horizon {self.horizon}), got {total}"
            )
        starts = np.arange(0, total - span + 1)
        contexts = np.stack([series[:, s : s + self.context_length] for s in starts])
        targets = np.stack(
            [series[:, s + self.context_length : s + span] for s in starts]
        )
        return (
            torch.from_numpy(contexts.astype(np.float32)),
            torch.from_numpy(targets.astype(np.float32)),
        )

    def fit(self, panel: Panel) -> None:
        """Train on the given panel, holding out its tail for early stopping."""
        torch.manual_seed(self.seed)
        # Single-threaded by default for two reasons: intra-op parallelism makes
        # reduction order, and therefore the trained weights, machine-dependent, and on
        # macOS torch and LightGBM link separate OpenMP runtimes whose thread pools
        # crash the process when both are live. Raise torch_threads only when a run does
        # not need to be bit-reproducible.
        torch.set_num_threads(max(1, self.torch_threads))
        device = self._resolve_device()
        series = fill_context(panel.series)
        self._series_ids = list(panel.series_ids)

        # Per-series standardisation from the training window only. Ground rule 4: the
        # evaluation window contributes nothing, not even a mean.
        offset = series.mean(axis=1, keepdims=True)
        scale = np.maximum(series.std(axis=1, keepdims=True), 1e-3)
        # Stored as (n_series,) so that the caller decides how to broadcast; keeping a
        # trailing singleton here is what silently produced a four-dimensional forecast.
        self._offset = torch.from_numpy(offset.astype(np.float32).ravel()).to(device)
        self._scale = torch.from_numpy(scale.astype(np.float32).ravel()).to(device)
        normalised = (series - offset) / scale

        contexts, targets = self._windows(normalised)
        n_windows = contexts.shape[0]
        n_validation = max(1, int(n_windows * self.validation_fraction))
        # The split is by time, not at random: shuffling windows would leak the future
        # into the early-stopping decision.
        train_x, train_y = contexts[:-n_validation], targets[:-n_validation]
        valid_x = contexts[-n_validation:].to(device)
        valid_y = targets[-n_validation:].to(device)

        adjacency = self.graph.series_adjacency(self._series_ids)
        self._operators = {k: v.to(device) for k, v in self._build_operators(adjacency).items()}
        module = self._build_module(len(self._series_ids), len(DEFAULT_QUANTILES)).to(device)
        levels = torch.tensor(DEFAULT_QUANTILES, dtype=torch.float32, device=device)
        optimiser = torch.optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        generator = torch.Generator().manual_seed(self.seed)

        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale = 0
        self._history = []

        for epoch in range(self.epochs):
            module.train()
            order = torch.randperm(train_x.shape[0], generator=generator)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                batch_x = train_x[index].to(device)
                batch_y = train_y[index].to(device)
                optimiser.zero_grad(set_to_none=True)
                loss = pinball_loss(self._apply(module, batch_x), batch_y, levels)
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), 5.0)
                optimiser.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            module.eval()
            with torch.no_grad():
                validation_loss = float(
                    pinball_loss(self._apply(module, valid_x), valid_y, levels)
                )
            self._history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(n_batches, 1),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        self._module = module

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        if self._module is None or self._scale is None or self._offset is None:
            raise ForecastError(f"{self.name}.predict called before fit")
        if list(panel.series_ids) != self._series_ids:
            raise ForecastError(
                f"{self.name} was trained on a different series set; these models are "
                "per-site and cannot transfer"
            )
        if horizon > self.horizon:
            raise ForecastError(
                f"{self.name} was trained for a horizon of {self.horizon}, asked for {horizon}"
            )
        if tuple(levels) != DEFAULT_QUANTILES:
            raise ForecastError(
                f"{self.name} was trained with a pinball head on {DEFAULT_QUANTILES}; "
                f"asking for {tuple(levels)} would require retraining"
            )

        device = self._resolve_device()
        series = fill_context(panel.series)[:, -self.context_length :]
        if series.shape[1] < self.context_length:
            raise ForecastError(
                f"{self.name} needs a context of {self.context_length} steps, got "
                f"{series.shape[1]}"
            )
        offset = self._offset.view(1, -1, 1)
        scale = self._scale.view(1, -1, 1)
        batch = torch.from_numpy(series.astype(np.float32)).unsqueeze(0).to(device)
        batch = (batch - offset) / scale

        with measure(device.type) as usage, torch.no_grad():
            out = self._apply(self._module, batch)
        self._last_usage = usage

        denormalised = out * scale.unsqueeze(-1) + offset.unsqueeze(-1)
        array = denormalised.squeeze(0).cpu().numpy()[:, :horizon, :]
        return self._validate_output(
            np.maximum(array, 0.0), panel, horizon, levels, self.name
        )

    def describe(self) -> dict[str, Any]:
        """Configuration and training curve recorded in the run manifest."""
        return {
            **super().describe(),
            "context_length": self.context_length,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs_run": len(self._history),
            "best_validation_loss": (
                min((h["validation_loss"] for h in self._history), default=None)
            ),
        }


class DCRNN(_GraphForecaster):
    """Diffusion convolutional recurrent network (Li et al., 2018)."""

    name = "dcrnn"

    def __init__(self, graph: BuildingGraph, *, k_hops: int = 2, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)
        self.k_hops = k_hops

    def _build_operators(self, adjacency: np.ndarray) -> dict[str, Tensor]:
        forward, backward = random_walk_transitions(adjacency)
        return {"forward": forward, "backward": backward}

    def _build_module(self, n_series: int, n_quantiles: int) -> nn.Module:
        return DCRNNModule(n_series, self.hidden_dim, self.k_hops, self.horizon, n_quantiles)

    def _apply(self, module: nn.Module, batch: Tensor) -> Tensor:
        return module(batch, self._operators["forward"], self._operators["backward"])


class STGCN(_GraphForecaster):
    """Spatio-temporal graph convolutional network (Yu et al., 2018)."""

    name = "stgcn"

    def __init__(self, graph: BuildingGraph, *, k_order: int = 3, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)
        self.k_order = k_order

    def _build_operators(self, adjacency: np.ndarray) -> dict[str, Tensor]:
        return {"laplacian": scaled_laplacian(adjacency)}

    def _build_module(self, n_series: int, n_quantiles: int) -> nn.Module:
        del n_series
        return STGCNModule(self.hidden_dim, self.k_order, self.horizon, n_quantiles)

    def _apply(self, module: nn.Module, batch: Tensor) -> Tensor:
        return module(batch, self._operators["laplacian"])


class GraphWaveNet(_GraphForecaster):
    """Graph WaveNet with an adaptive adjacency (Wu et al., 2019)."""

    name = "graph_wavenet"

    def __init__(self, graph: BuildingGraph, *, n_layers: int = 4, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)
        self.n_layers = n_layers

    def _build_operators(self, adjacency: np.ndarray) -> dict[str, Tensor]:
        forward, backward = random_walk_transitions(adjacency)
        return {"forward": forward, "backward": backward}

    def _build_module(self, n_series: int, n_quantiles: int) -> nn.Module:
        return GraphWaveNetModule(
            n_series, self.hidden_dim, self.n_layers, self.horizon, n_quantiles
        )

    def _apply(self, module: nn.Module, batch: Tensor) -> Tensor:
        return module(batch, self._operators["forward"], self._operators["backward"])
