"""Diebold-Mariano tests on paired loss differentials (M6).

A table of MAE values does not say whether one method is better than another. The rule
this project follows is that an improvement which is not significant is not claimed, so
every headline comparison carries a p-value from here.

The test is Diebold and Mariano (1995), *Comparing predictive accuracy*, Journal of
Business and Economic Statistics 13(3), 253-263, with the small-sample correction of
Harvey, Leybourne and Newbold (1997), *Testing the equality of prediction mean squared
errors*, International Journal of Forecasting 13(2), 281-291.

Two details matter for this setting and are handled explicitly:

*Multi-step horizons.* Forecasts made at overlapping origins for a horizon of ``h`` steps
share information, so their loss differentials are autocorrelated up to lag ``h - 1``.
The long-run variance therefore uses a Newey-West estimator truncated at ``h - 1`` rather
than the sample variance, which would otherwise understate the standard error and
manufacture significance.

*Small numbers of origins.* A rolling-origin run over a few days of test data gives tens
of origins, not thousands. The Harvey-Leybourne-Newbold factor corrects the statistic and
the reference distribution is Student's t with ``n - 1`` degrees of freedom rather than
the normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Alternative = Literal["two-sided", "less", "greater"]


class SignificanceError(ValueError):
    """Raised when a test cannot be computed from the inputs given."""


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of one paired comparison.

    Attributes:
        statistic: the corrected DM statistic. Negative means the first method has the
            lower loss.
        p_value: p-value under Student's t with ``n_origins - 1`` degrees of freedom.
        mean_difference: mean of ``loss_a - loss_b``, in the units of the loss.
        n_origins: number of paired observations.
        lag: truncation lag used for the long-run variance.
    """

    statistic: float
    p_value: float
    mean_difference: float
    n_origins: int
    lag: int

    @property
    def favours(self) -> Literal["a", "b", "tie"]:
        """Which method has the lower mean loss, ignoring significance."""
        if self.mean_difference < 0:
            return "a"
        if self.mean_difference > 0:
            return "b"
        return "tie"

    def marker(self, alpha: float = 0.05) -> str:
        """Significance marker for a results table: ``*`` when the difference holds up."""
        return "*" if self.p_value < alpha else ""


def newey_west_variance(differences: np.ndarray, lag: int) -> float:
    """Long-run variance of the mean of ``differences``, truncated at ``lag``.

    Args:
        differences: the paired loss differentials, one per origin.
        lag: largest autocovariance included. For an ``h``-step forecast this is
            ``h - 1``.

    Raises:
        SignificanceError: if the lag is longer than the sample.
    """
    n = differences.shape[0]
    if lag < 0:
        raise SignificanceError(f"lag must not be negative, got {lag}")
    if lag >= n:
        raise SignificanceError(f"lag {lag} needs more than {n} origins")
    centred = differences - differences.mean()
    variance = float(np.dot(centred, centred) / n)
    for k in range(1, lag + 1):
        autocovariance = float(np.dot(centred[k:], centred[:-k]) / n)
        # Bartlett weights, so that the estimator stays non-negative in finite samples.
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * autocovariance
    return variance


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    alternative: Alternative = "two-sided",
) -> DieboldMarianoResult:
    """Test whether two methods have equal expected loss.

    Args:
        loss_a: per-origin loss of the first method.
        loss_b: per-origin loss of the second method, paired by origin.
        horizon: the forecast horizon in steps, which sets the truncation lag.
        alternative: ``less`` tests that method A is better.

    Raises:
        SignificanceError: if the inputs are not paired, are too short, or produce a
            non-positive variance estimate. A non-positive variance means the differentials
            carry no usable information, and reporting p = 1 for it would look like a
            tested result rather than an untestable one.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape:
        raise SignificanceError(f"losses must be paired, got {a.shape} and {b.shape}")
    if a.ndim != 1:
        raise SignificanceError(f"losses must be one value per origin, got {a.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = a.shape[0]
    if n < 3:
        raise SignificanceError(f"need at least 3 paired origins, got {n}")

    differences = a - b
    mean_difference = float(differences.mean())
    lag = min(max(horizon - 1, 0), n - 1)
    variance = newey_west_variance(differences, lag)
    if variance <= 0.0:
        raise SignificanceError(
            "the long-run variance of the loss differential is not positive, so the two "
            "methods cannot be compared; this happens when their losses are identical"
        )

    statistic = mean_difference / np.sqrt(variance / n)
    # Harvey, Leybourne and Newbold (1997) small-sample correction. The factor goes
    # non-positive once the horizon is long relative to the number of origins, which is
    # the arithmetic saying the sample cannot support a test at that horizon. Reporting
    # the resulting statistic of zero, and its p-value of one, would read as strong
    # evidence of no difference when it is in fact no evidence at all.
    inner = (n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n
    if inner <= 0.0:
        raise SignificanceError(
            f"{n} origins cannot support a Diebold-Mariano test at horizon {horizon}: the "
            f"Harvey-Leybourne-Newbold correction requires roughly more than 2h origins. "
            "Shorten the stride or lengthen the test window."
        )
    statistic *= np.sqrt(inner)

    distribution = stats.t(df=n - 1)
    if alternative == "two-sided":
        p_value = float(2.0 * distribution.sf(abs(statistic)))
    elif alternative == "less":
        p_value = float(distribution.cdf(statistic))
    elif alternative == "greater":
        p_value = float(distribution.sf(statistic))
    else:
        raise SignificanceError(f"unknown alternative {alternative!r}")

    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=p_value,
        mean_difference=mean_difference,
        n_origins=n,
        lag=lag,
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down correction over a family of comparisons.

    A results table that compares a dozen methods against a baseline will produce a
    significant p-value by chance. Holm controls the family-wise error rate without the
    conservatism of plain Bonferroni, and unlike it is uniformly more powerful.

    Args:
        p_values: one p-value per comparison, keyed however the caller likes.
        alpha: family-wise error rate.

    Returns:
        Whether each comparison survives the correction, keyed the same way.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    verdict: dict[str, bool] = {}
    rejected_so_far = True
    for index, (key, p) in enumerate(ordered):
        # Once one hypothesis fails, every larger p-value fails too: that step-down rule
        # is what makes Holm valid.
        rejected_so_far = rejected_so_far and p <= alpha / (n - index)
        verdict[key] = rejected_so_far
    return verdict
