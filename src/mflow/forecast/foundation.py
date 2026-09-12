"""Zero-shot foundation model wrappers.

Every wrapper here was written against the vendor's source or model card, not against a
remembered signature:

* **TimesFM 3.0** -- ``timesfm3.TimesFM3Forecaster.from_pretrained(...)`` and
  ``predict_batch(contexts, horizon, past_only_covariates=..., past_future_covariates=...,
  return_quantiles=True, padding_mode=...)``, which yields
  :class:`timesfm3.ForecastOutput` objects whose ``quantiles`` field is
  ``(horizon, n_quantiles)`` for a 1-D context and ``(n_variates, horizon, n_quantiles)``
  for a 2-D one. Verified in ``src/timesfm3/torch/timesfm3_forecaster.py`` of
  https://github.com/google-research/timesfm.
* **TimesFM 2.5** -- ``timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)``, then
  ``compile(timesfm.ForecastConfig(...))`` before ``forecast(horizon, inputs)``, which
  returns ``(point (B, H), quantiles (B, H, 10))`` where column 0 is the mean and columns
  1..9 are the deciles q10..q90.
* **Chronos-2** -- ``chronos.Chronos2Pipeline.from_pretrained(...).predict_df(...)``,
  per https://github.com/amazon-science/chronos-forecasting.
* **Toto 2.0** -- ``toto2.Toto2Model.from_pretrained(...).forecast({"target", "target_mask",
  "series_ids"}, horizon=...)`` returning ``(9, batch, n_variates, horizon)``, per the
  https://huggingface.co/Datadog/Toto-2.0-2.5B model card.

Three conventions are shared by all of them.

*Normalisation.* The series are wildly heterogeneous in scale -- a bottleneck corridor
sees hundreds of crossings per minute while a side gallery sees three. Each model's own
per-series normalisation is used and no extra scaling is layered on top, because an extra
transform would confound the comparison. The relevant switch is exposed
(``use_znorm`` for TimesFM 3, ``normalize_inputs`` for TimesFM 2.5) so that the choice can
be ablated rather than assumed.

*Batching.* Multivariate modes send the whole site as one request. Batching by series
would defeat the cross-series attention that those modes exist to use.

*Cost.* Every call is wrapped in :func:`mflow.profiling.measure`, because the edge
deployment claim needs measured latency and peak memory, not estimates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, ClassVar, Final

import numpy as np

from mflow.forecast.base import (
    DEFAULT_QUANTILES,
    ForecastError,
    Forecaster,
    validate_quantiles,
)
from mflow.profiling import measure
from mflow.schema import Panel

#: Quantile levels the three model families emit natively.
NATIVE_DECILES: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

TIMESFM3_REPO: Final[str] = "google/timesfm-3.0-pytorch"
TIMESFM25_REPO: Final[str] = "google/timesfm-2.5-200m-pytorch"
CHRONOS2_REPO: Final[str] = "amazon/chronos-2"
TOTO2_REPO: Final[str] = "Datadog/Toto-2.0-313m"


def _hf_token() -> str | None:
    """Hugging Face token from the environment, if the user configured one.

    ``.env`` in the repository root carries ``HUGGINGFACE_API_KEY``; it is loaded by
    :func:`mflow.cli.load_environment` at start-up. Gated repositories need it, public
    ones do not, and it is never written into a manifest.
    """
    for key in ("HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _torch_device(requested: str | None) -> str:
    """Resolve ``auto`` to the best available torch device."""
    if requested and requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def map_quantiles(
    fan: np.ndarray, native: Sequence[float], requested: Sequence[float]
) -> np.ndarray:
    """Resample a quantile fan from the model's native levels onto the requested ones.

    Levels that the model emits directly are copied; anything else is linearly
    interpolated in level space, which is the standard reading of a discrete quantile
    function and keeps the fan monotone. Extrapolation beyond the outermost native level
    is refused: inventing a 99th percentile from a model that only reports up to the 90th
    would put a fabricated number into a coverage table.

    Args:
        fan: ``(..., len(native))`` ascending in the last axis.
        native: the model's own levels, ascending.
        requested: the levels wanted, ascending.

    Returns:
        ``(..., len(requested))``.
    """
    native_levels = np.asarray(native, dtype=np.float64)
    wanted = np.asarray(requested, dtype=np.float64)
    if wanted.min() < native_levels.min() - 1e-9 or wanted.max() > native_levels.max() + 1e-9:
        raise ForecastError(
            f"requested quantile levels {list(requested)} fall outside the levels the "
            f"model reports ({list(native)}); extrapolating them would fabricate an "
            "uncertainty estimate the model never produced"
        )
    if len(native_levels) == len(wanted) and np.allclose(native_levels, wanted):
        return np.asarray(fan, dtype=np.float64)
    flat = np.asarray(fan, dtype=np.float64).reshape(-1, len(native_levels))
    out = np.stack([np.interp(wanted, native_levels, row) for row in flat], axis=0)
    return out.reshape(*fan.shape[:-1], len(wanted))


class _ZeroShotForecaster(Forecaster):
    """Shared behaviour of the zero-shot wrappers.

    Attributes:
        repo_id: the Hugging Face repository the weights come from.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        max_context: context truncation, in steps.
    """

    requires_training = False
    native_quantiles: ClassVar[tuple[float, ...]] = NATIVE_DECILES

    def __init__(
        self,
        *,
        repo_id: str,
        device: str = "auto",
        max_context: int = 2048,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.repo_id = repo_id
        self.device = device
        self.max_context = max_context
        self._model: Any = None

    def fit(self, panel: Panel) -> None:
        """Refuse training data.

        A zero-shot claim is only meaningful if the model never touches anything outside
        the context window, so being handed a training panel is an error rather than
        something to ignore politely.
        """
        self._assert_zero_shot(panel)

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in the run manifest."""
        return {
            **super().describe(),
            "repo_id": self.repo_id,
            "device": self.device,
            "max_context": self.max_context,
        }

    def _context(self, panel: Panel) -> np.ndarray:
        """Truncate the context to what the model accepts.

        NaNs are left in place: all four models document their own missing-value handling
        (leading NaNs stripped, interior NaNs interpolated), and duplicating that here
        would mean the wrapper and the model disagree about what was observed.
        """
        series = np.asarray(panel.series, dtype=np.float32)
        return series[:, -self.max_context :] if series.shape[1] > self.max_context else series

    def unload(self) -> None:
        """Release the weights, for sweeps that cycle through several models."""
        self._model = None


# --------------------------------------------------------------------------- #
# TimesFM 3.0
# --------------------------------------------------------------------------- #


class _TimesFM3Base(_ZeroShotForecaster):
    """Common loading and decoding for the three TimesFM 3 modes.

    Args:
        per_core_batch_size: inference batch size passed to the forecaster config.
        use_znorm: let the wrapper z-normalise before the model sees the data. Off by
            default; TimesFM 3 performs its own reversible instance normalisation, and
            stacking another one is exactly the kind of undeclared preprocessing that
            makes a zero-shot comparison unreproducible.
        make_positive: clamp forecasts of non-negative inputs to zero. Visitor counts
            cannot be negative, so this is on.
    """

    def __init__(
        self,
        *,
        repo_id: str = TIMESFM3_REPO,
        device: str = "auto",
        max_context: int = 2048,
        per_core_batch_size: int = 8,
        use_znorm: bool = False,
        make_positive: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(repo_id=repo_id, device=device, max_context=max_context, seed=seed)
        self.per_core_batch_size = per_core_batch_size
        self.use_znorm = use_znorm
        self.make_positive = make_positive

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from timesfm3 import TimesFM3Forecaster
        except ImportError as exc:
            raise ForecastError(
                "TimesFM 3 is not installed. Install it from the checkout with "
                "`pip install -e /path/to/timesfm` or `pip install 'mflow[timesfm]'`."
            ) from exc
        self._model = TimesFM3Forecaster.from_pretrained(
            self.repo_id,
            device=_torch_device(self.device),
            per_core_batch_size=self.per_core_batch_size,
            token=_hf_token(),
        )
        native = tuple(float(q) for q in self._model.config.quantiles)
        if native != self.native_quantiles:
            # The checkpoint decides its own quantile head; trust it over our constant.
            type(self).native_quantiles = native
        return self._model

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in the run manifest."""
        return {
            **super().describe(),
            "per_core_batch_size": self.per_core_batch_size,
            "use_znorm": self.use_znorm,
            "make_positive": self.make_positive,
        }


class TimesFM3Univariate(_TimesFM3Base):
    """TimesFM 3, one independent request per series.

    This is the reference point for H2: whatever the multivariate mode gains, it gains
    over this.
    """

    name = "timesfm3_univariate"
    supports_multivariate = False
    supports_future_covariates = False

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        model = self._load()
        context = self._context(panel)
        contexts = [context[i].astype(np.float32) for i in range(context.shape[0])]

        with measure(_torch_device(self.device)) as usage:
            outputs = list(
                model.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    ts_ids=list(panel.series_ids),
                    return_quantiles=True,
                    make_positive=self.make_positive,
                    sort_quantiles=True,
                    use_znorm=self.use_znorm,
                )
            )
        self._last_usage = usage

        fan = np.stack([np.asarray(out.quantiles, dtype=np.float64) for out in outputs], axis=0)
        mapped = map_quantiles(fan, self.native_quantiles, levels)
        return self._validate_output(mapped, panel, horizon, levels, self.name)


class TimesFM3Multivariate(_TimesFM3Base):
    """TimesFM 3 with variate attention across every series of the site.

    The whole site goes in as a single ``(n_series, T)`` context. Splitting it into
    batches of series would silence the cross-series attention this mode exists for.
    """

    name = "timesfm3_multivariate"
    supports_multivariate = True
    supports_future_covariates = False

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        model = self._load()
        context = self._context(panel).astype(np.float32)

        with measure(_torch_device(self.device)) as usage:
            outputs = list(
                model.predict_batch(
                    contexts=[context],
                    horizon=horizon,
                    ts_ids=[panel.site_id],
                    return_quantiles=True,
                    make_positive=self.make_positive,
                    sort_quantiles=True,
                    use_znorm=self.use_znorm,
                )
            )
        self._last_usage = usage

        fan = np.asarray(outputs[0].quantiles, dtype=np.float64)
        mapped = map_quantiles(fan, self.native_quantiles, levels)
        return self._validate_output(mapped, panel, horizon, levels, self.name)


class TimesFM3MultivariateCovariates(_TimesFM3Base):
    """TimesFM 3 with variate attention and known-future covariates.

    Covariate handling
    ------------------
    The panel's ``future_covariates`` are handed to ``past_future_covariates``, which the
    model expects as ``(n_covariates, context + horizon)``. Because TimesFM rounds the
    horizon up to a multiple of its output patch length, ``padding_mode="edge"`` extends
    the covariate window to the rounded horizon by repeating its last value.

    Past-only covariates (CO2, temperature, Wi-Fi device counts) go to
    ``past_only_covariates``.

    Static per-series attributes -- node kind, floor area, capacity -- are *not* future
    covariates and are deliberately not smuggled in as constant channels. The TimesFM 3
    inference API exposes no static covariate input (unlike TimesFM 1's
    ``forecast_with_covariates``, which took ``static_categorical_covariates``), so this
    configuration simply does without them; that limitation is reported rather than
    worked around.

    Args:
        covariate_subset: restrict the known-future covariates to these ids, for the E2
            subset ablation. None uses all of them.
        use_past_covariates: include the past-only covariate channels.
    """

    name = "timesfm3_multivariate_covariates"
    supports_multivariate = True
    supports_future_covariates = True

    def __init__(
        self,
        *,
        covariate_subset: Sequence[str] | None = None,
        use_past_covariates: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.covariate_subset = tuple(covariate_subset) if covariate_subset else None
        self.use_past_covariates = use_past_covariates

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in the run manifest."""
        return {
            **super().describe(),
            "covariate_subset": list(self.covariate_subset) if self.covariate_subset else None,
            "use_past_covariates": self.use_past_covariates,
        }

    def _future_block(self, panel: Panel, horizon: int) -> np.ndarray | None:
        if panel.future_covariates is None:
            raise ForecastError(
                f"{self.name} needs known-future covariates but the panel carries none; "
                "use TimesFM3Multivariate for sites without them"
            )
        available = panel.horizon_covered
        if available < horizon:
            raise ForecastError(
                f"{self.name} needs {horizon} steps of known-future covariates, the panel "
                f"provides {available}"
            )
        rows = range(len(panel.future_covariate_ids))
        if self.covariate_subset is not None:
            missing = set(self.covariate_subset) - set(panel.future_covariate_ids)
            if missing:
                raise ForecastError(f"{self.name}: covariates {sorted(missing)} are not in the panel")
            rows = [panel.future_covariate_ids.index(c) for c in self.covariate_subset]
        block = panel.future_covariates[list(rows), :]
        # Trim the context part to the truncated window, then keep exactly `horizon`
        # future columns so the model can infer the horizon from the covariate width.
        context_len = min(panel.n_timesteps, self.max_context)
        start = panel.n_timesteps - context_len
        return np.asarray(
            block[:, start : panel.n_timesteps + horizon], dtype=np.float32
        )

    def _past_block(self, panel: Panel) -> np.ndarray | None:
        if not self.use_past_covariates or panel.past_covariates is None:
            return None
        context_len = min(panel.n_timesteps, self.max_context)
        return np.asarray(panel.past_covariates[:, -context_len:], dtype=np.float32)

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        model = self._load()
        context = self._context(panel).astype(np.float32)

        with measure(_torch_device(self.device)) as usage:
            outputs = list(
                model.predict_batch(
                    contexts=[context],
                    horizon=horizon,
                    past_only_covariates=[self._past_block(panel)],
                    past_future_covariates=[self._future_block(panel, horizon)],
                    ts_ids=[panel.site_id],
                    return_quantiles=True,
                    make_positive=self.make_positive,
                    sort_quantiles=True,
                    use_znorm=self.use_znorm,
                    padding_mode="edge",
                )
            )
        self._last_usage = usage

        fan = np.asarray(outputs[0].quantiles, dtype=np.float64)
        mapped = map_quantiles(fan, self.native_quantiles, levels)
        return self._validate_output(mapped, panel, horizon, levels, self.name)


# --------------------------------------------------------------------------- #
# TimesFM 2.5
# --------------------------------------------------------------------------- #


class TimesFM25Univariate(_ZeroShotForecaster):
    """TimesFM 2.5, univariate, for the generational ablation.

    The 2.5 API differs from 3.0: the model must be compiled with a
    :class:`timesfm.ForecastConfig` before ``forecast`` may be called, and the returned
    quantile array has ten columns whose first entry is the *mean*, not a quantile, with
    the nine deciles in columns 1..9.
    """

    name = "timesfm25_univariate"
    supports_multivariate = False
    supports_future_covariates = False

    def __init__(
        self,
        *,
        repo_id: str = TIMESFM25_REPO,
        device: str = "auto",
        max_context: int = 2048,
        max_horizon: int = 256,
        per_core_batch_size: int = 32,
        normalize_inputs: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(repo_id=repo_id, device=device, max_context=max_context, seed=seed)
        self.max_horizon = max_horizon
        self.per_core_batch_size = per_core_batch_size
        self.normalize_inputs = normalize_inputs

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import timesfm
        except ImportError as exc:
            raise ForecastError(
                "TimesFM is not installed; `pip install 'mflow[timesfm]'`"
            ) from exc
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=self.normalize_inputs,
                per_core_batch_size=self.per_core_batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        return model

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        if horizon > self.max_horizon:
            raise ForecastError(
                f"{self.name} was compiled for horizons up to {self.max_horizon}, got {horizon}"
            )
        model = self._load()
        context = self._context(panel)
        inputs = [context[i].astype(np.float32) for i in range(context.shape[0])]

        with measure(_torch_device(self.device)) as usage:
            _, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)
        self._last_usage = usage

        fan = np.asarray(quantile_forecast, dtype=np.float64)
        if fan.shape[-1] != 10:
            raise ForecastError(
                f"{self.name}: expected 10 quantile columns (mean + 9 deciles) from "
                f"TimesFM 2.5, got {fan.shape[-1]}"
            )
        deciles = fan[..., 1:]  # drop the mean column
        mapped = map_quantiles(deciles, NATIVE_DECILES, levels)
        return self._validate_output(np.maximum(mapped, 0.0), panel, horizon, levels, self.name)


# --------------------------------------------------------------------------- #
# Chronos-2
# --------------------------------------------------------------------------- #


class Chronos2(_ZeroShotForecaster):
    """Chronos-2, the independent covariate-aware comparator.

    Chronos-2's public inference entry point is ``predict_df``, a long-format dataframe
    API. In multivariate mode the whole site is one item with one column per series, so
    the model attends across them; in univariate mode each series is its own item.

    Args:
        multivariate: put every series in one item.
        use_future_covariates: append the panel's known-future covariates to ``future_df``.
    """

    name = "chronos2"

    def __init__(
        self,
        *,
        repo_id: str = CHRONOS2_REPO,
        device: str = "auto",
        max_context: int = 2048,
        multivariate: bool = True,
        use_future_covariates: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(repo_id=repo_id, device=device, max_context=max_context, seed=seed)
        self.multivariate = multivariate
        self.use_future_covariates = use_future_covariates
        self.supports_multivariate = multivariate
        self.supports_future_covariates = use_future_covariates
        self.name = "chronos2_multivariate" if multivariate else "chronos2_univariate"
        if use_future_covariates:
            self.name += "_covariates"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ForecastError(
                "Chronos is not installed; `pip install 'mflow[chronos]'`"
            ) from exc
        self._model = Chronos2Pipeline.from_pretrained(
            self.repo_id, device_map=_torch_device(self.device)
        )
        return self._model

    def _frames(self, panel: Panel, horizon: int) -> tuple[Any, Any, list[str]]:
        import pandas as pd

        context = self._context(panel)
        context_len = context.shape[1]
        stamps = panel.timestamps[-context_len:]
        future_stamps = pd.date_range(
            stamps[-1] + pd.Timedelta(seconds=panel.interval_seconds),
            periods=horizon,
            freq=pd.Timedelta(seconds=panel.interval_seconds),
            tz=stamps.tz,
        )

        if self.multivariate:
            targets = list(panel.series_ids)
            context_df = pd.DataFrame(
                {"id": panel.site_id, "timestamp": stamps}
                | {sid: context[i] for i, sid in enumerate(panel.series_ids)}
            )
            future_df = pd.DataFrame({"id": panel.site_id, "timestamp": future_stamps})
        else:
            targets = ["target"]
            context_df = pd.concat(
                [
                    pd.DataFrame(
                        {"id": sid, "timestamp": stamps, "target": context[i]}
                    )
                    for i, sid in enumerate(panel.series_ids)
                ],
                ignore_index=True,
            )
            future_df = pd.concat(
                [
                    pd.DataFrame({"id": sid, "timestamp": future_stamps})
                    for sid in panel.series_ids
                ],
                ignore_index=True,
            )

        if self.use_future_covariates:
            if panel.future_covariates is None or panel.horizon_covered < horizon:
                raise ForecastError(
                    f"{self.name} needs {horizon} steps of known-future covariates"
                )
            start = panel.n_timesteps - context_len
            for row, cov_id in enumerate(panel.future_covariate_ids):
                column = cov_id.replace("|", "_")
                past_values = panel.future_covariates[row, start : panel.n_timesteps]
                future_values = panel.future_covariates[
                    row, panel.n_timesteps : panel.n_timesteps + horizon
                ]
                context_df[column] = np.tile(past_values, len(context_df) // context_len)
                future_df[column] = np.tile(future_values, len(future_df) // horizon)

        return context_df, future_df, targets

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        levels = validate_quantiles(quantiles)
        pipeline = self._load()
        context_df, future_df, targets = self._frames(panel, horizon)

        with measure(_torch_device(self.device)) as usage:
            predictions = pipeline.predict_df(
                context_df,
                future_df=future_df if self.use_future_covariates else None,
                prediction_length=horizon,
                quantile_levels=list(levels),
                id_column="id",
                timestamp_column="timestamp",
                target=targets if self.multivariate else "target",
            )
        self._last_usage = usage

        out = np.empty((panel.n_series, horizon, len(levels)), dtype=np.float64)
        columns = [str(level) for level in levels]
        if self.multivariate:
            for i, sid in enumerate(panel.series_ids):
                rows = predictions[predictions["target"] == sid] if "target" in predictions else predictions
                out[i] = rows[columns].to_numpy()[:horizon]
        else:
            for i, sid in enumerate(panel.series_ids):
                rows = predictions[predictions["id"] == sid]
                out[i] = rows[columns].to_numpy()[:horizon]
        return self._validate_output(np.maximum(out, 0.0), panel, horizon, levels, self.name)


# --------------------------------------------------------------------------- #
# Toto 2.0
# --------------------------------------------------------------------------- #


class Toto2(_ZeroShotForecaster):
    """Toto 2.0, the independent multivariate comparator.

    The model card's inference signature takes a dict of tensors shaped
    ``(batch, n_variates, time)`` and returns quantiles shaped
    ``(9, batch, n_variates, horizon)`` at levels 0.1 through 0.9. The whole site is one
    batch element so that Toto's variate attention sees every room at once.

    Args:
        decode_block_size: parallel decoding block size from the model card.
        checkpoint sizes: the 4m and 22m checkpoints are the ones relevant to the edge
            deployment argument; 313m is the default general-purpose choice.
    """

    name = "toto2"
    supports_multivariate = True
    supports_future_covariates = False

    def __init__(
        self,
        *,
        repo_id: str = TOTO2_REPO,
        device: str = "auto",
        max_context: int = 2048,
        decode_block_size: int = 768,
        seed: int = 0,
    ) -> None:
        super().__init__(repo_id=repo_id, device=device, max_context=max_context, seed=seed)
        self.decode_block_size = decode_block_size

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from toto2 import Toto2Model
        except ImportError as exc:
            raise ForecastError(
                "Toto is not installed; `pip install 'mflow[toto]'` (package `toto-models`)"
            ) from exc
        import torch

        model = Toto2Model.from_pretrained(self.repo_id)
        self._model = model.to(torch.device(_torch_device(self.device))).eval()
        return self._model

    def predict(
        self,
        panel: Panel,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> np.ndarray:
        import torch

        levels = validate_quantiles(quantiles)
        model = self._load()
        device = torch.device(_torch_device(self.device))
        context = self._context(panel)

        observed = ~np.isnan(context)
        target = torch.from_numpy(np.nan_to_num(context, nan=0.0).astype(np.float32))
        target = target.unsqueeze(0).to(device)
        mask = torch.from_numpy(observed).unsqueeze(0).to(device)
        series_ids = torch.zeros(1, context.shape[0], dtype=torch.long, device=device)

        with measure(_torch_device(self.device)) as usage, torch.inference_mode():
            fan = model.forecast(
                {"target": target, "target_mask": mask, "series_ids": series_ids},
                horizon=horizon,
                decode_block_size=self.decode_block_size,
                has_missing_values=bool((~observed).any()),
            )
        self._last_usage = usage

        # (9, batch, n_variates, horizon) -> (n_variates, horizon, 9)
        array = np.asarray(fan.detach().cpu().numpy(), dtype=np.float64)
        array = np.transpose(array[:, 0], (1, 2, 0))
        mapped = map_quantiles(array, NATIVE_DECILES, levels)
        return self._validate_output(np.maximum(mapped, 0.0), panel, horizon, levels, self.name)
