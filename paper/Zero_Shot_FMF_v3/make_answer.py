"""Build answer.docx and answer.pdf, the response to the second review.

Both files are rendered from the single content list below, so the Word document and the
PDF cannot drift apart. The Word file is written with python-docx and the PDF is typeset
with pdflatex from the same blocks.

The reviewer's comments are quoted verbatim from the decision letter. Every number quoted
in a response is a number of the revised manuscript, where ``verify_round2.py`` checks it
against the logged analysis; nothing here is computed afresh.

Inline markup understood by both renderers::

    **bold**    *italic*    `monospace`    ~subscript~

Usage:
    python make_answer.py            # writes answer.docx and answer.pdf beside this file
    python make_answer.py --no-pdf   # the Word file only
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.shared import Inches, Pt, RGBColor
from docx.table import _Cell
from docx.text.paragraph import Paragraph
from docx.text.run import Run

HERE = Path(__file__).resolve().parent
DOCX = HERE / "answer.docx"
PDF = HERE / "answer.pdf"
PDFLATEX = Path("/Library/TeX/texbin/pdflatex")

ACCENT_HEX = "8C0C0C"
ACCENT = RGBColor(0x8C, 0x0C, 0x0C)
QUOTE_FILL = "F4F1EC"
HEAD_FILL = "E9E4DC"
BODY_FONT = "Times New Roman"

TITLE = "Response to the Reviewer"
SUBTITLE = "Second revision"
MANUSCRIPT = (
    "Topology-Constrained Reconciliation for Zero-Shot Forecasting of Building Occupancy "
    "and Doorway Flow"
)
AUTHORS = "A. A. Jafari, C. Ozcinar and G. Anbarjafari"


class AnswerError(RuntimeError):
    """Raised when the response cannot be built as specified."""


# --------------------------------------------------------------------------- #
# Content model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Heading:
    text: str
    level: int = 1


@dataclass(frozen=True)
class Label:
    """A run-in label such as 'Response' that introduces the paragraphs after it."""

    text: str


@dataclass(frozen=True)
class Para:
    text: str


@dataclass(frozen=True)
class Quote:
    """The reviewer's own words, reproduced verbatim."""

    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Bullets:
    items: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    caption: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    #: Relative column widths; they are normalised to the text width.
    widths: tuple[float, ...]
    #: Column alignment, one of ``l``, ``c`` or ``r`` per column.
    align: str = ""
    #: Rows, by index, that span the table as a panel title.
    panels: tuple[int, ...] = field(default_factory=tuple)


Block = Heading | Label | Para | Quote | Bullets | Table


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #

CONTENT: list[Block] = [
    Para(
        "We thank the reviewer for a second careful reading of the manuscript and for the "
        "constructive tone of the report. All seven comments are accepted. Four of them "
        "identified errors of ours and not only points of presentation: the "
        "matched-coverage table had been computed inconsistently, one percentage had been "
        "computed on a different loss and with a different denominator from every other "
        "percentage in the paper, the explanation we gave for the per-quantile residuals "
        "was wrong, and the public repository had not been brought into line with the "
        "manuscript. In each case we traced the cause, corrected it, and state below "
        "exactly what was wrong."
    ),
    Para(
        "Two headline numbers are withdrawn as a result: the 4.9-fold sharpness advantage "
        "at matched coverage and the 28.9% advantage of the local model. Two conclusions "
        "are narrowed: the sharpness claim on metro flow, which is not established out of "
        "sample, and the risk-sensitive reading of the peak error, which the detection "
        "results do not support. No conclusion has been strengthened in this round."
    ),
    Heading("How to read the revised manuscript", 2),
    Bullets(
        (
            "**Marking.** Changes made in this round are in red. The changes of the first "
            "revision, which were red in the previous version, are now in black, so that "
            "red marks only what is new.",
            "**Numbering.** Section numbers are unchanged. Three tables have been added "
            "in Section 7.7 (Tables 11, 12 and 13), so the later tables are renumbered: "
            "the table of coherence by quantile level, which the reviewer cites as Table "
            "11, is now Table 14. Tables 4, 8 and 10 keep their numbers. Two equations "
            "have been added, Equations (16) and (17), and two references, Künsch (1989) "
            "for the moving-block bootstrap and Vovk, Gammerman and Shafer (2005) for "
            "split conformal calibration.",
            "**Provenance of the new numbers.** No forecaster was rerun. Every number "
            "added in this round is recomputed from the predictions logged for the first "
            "revision by one script, `analysis_round2.py`, which writes "
            "`summary_round2.json`. A second script, `verify_round2.py`, rebuilds 126 "
            "table rows and quoted values from that file, formats each as it is printed, "
            "and fails if any of them is absent from the manuscript source. Running the "
            "analysis twice reproduces the summary file byte for byte.",
        )
    ),
    Table(
        caption="Summary of the comments and of what was done.",
        header=("Comment", "Assessment", "Principal change", "Where"),
        rows=(
            (
                "1. Matched coverage",
                "Correct. The table was computed inconsistently.",
                "Table and procedure replaced. The 4.9-fold claim is withdrawn. Factors "
                "are now reported both retrospectively and out of sample.",
                "Section 7.5, Table 8, Equation (17); abstract, Sections 1, 5, 8, 9",
            ),
            (
                "2. Percentage against LightGBM",
                "Correct. The figure is 16.6%.",
                "One definition of relative improvement, stated as an equation and "
                "applied everywhere. 28.9% is explained and withdrawn.",
                "Section 6.4, Equation (16); Section 7.2; abstract, Sections 1, 5, 8, 9",
            ),
            (
                "3. Statistical interpretation",
                "Correct. The two statistics answer different questions.",
                "Mean and typical-origin effects are tested separately, with a sign test "
                "and a bootstrap interval for the median. Block lengths 1 to 40 reported.",
                "Sections 6.4, 7.7 and 7.10; Tables 10, 11 and 12",
            ),
            (
                "4. Per-quantile residuals",
                "Correct. Solver tolerance was not the cause.",
                "The residual is produced by the monotone rearrangement. Tolerances "
                "reported, identical implementation confirmed, table split into before "
                "and after the rearrangement.",
                "Sections 4.4 and 7.8; Table 14",
            ),
            (
                "5. Repository",
                "Correct. The corrections had not been published.",
                "README rewritten, configurations mapped to tables, version tagged "
                "`manuscript-r2`.",
                "Repository; Limitations; Reproducibility statement",
            ),
            (
                "6. Seasonal-strength hypothesis",
                "Correct.",
                "\u201cRefuted\u201d replaced by \u201cnot supported by the within-dataset "
                "analysis\u201d and \u201cnot statistically established\u201d throughout.",
                "Section 7.4; abstract, Sections 1, 5, 8, 9",
            ),
            (
                "7. Risk-sensitive evaluation",
                "Correct. The missing results are now reported and they narrow the claim.",
                "High-occupancy MAE and detection scores reported with thresholds and "
                "sample sizes. Three levels of claim separated. Operational statements "
                "withdrawn.",
                "Sections 6.4 and 7.7; Table 13; abstract, Sections 1, 8, 9",
            ),
        ),
        widths=(1.25, 1.35, 2.6, 1.6),
        align="llll",
    ),
    # ------------------------------------------------------------------ 1
    Heading("Comment 1. Matched-coverage interval comparison (Section 7.5, Table 8)", 1),
    Label("Reviewer comment"),
    Quote(
        (
            "I appreciate the authors' effort to separate calibration from sharpness and "
            "to compare interval widths at matched empirical coverage. However, the "
            "rescaling result for seasonal naive on ROBOD appears inconsistent.",
            "Table 8 reports empirical coverage of 0.839 for seasonal naive, but applies a "
            "scaling factor of 1.60 to obtain nominal 0.80 coverage. For fixed interval "
            "centres, increasing interval width cannot reduce empirical coverage. Since "
            "the original coverage is already above 0.80, the required factor should not "
            "exceed one.",
            "This is important because the manuscript's reported 4.9-fold sharpness "
            "advantage depends on the resulting width of 3.75 persons.",
            "Could the authors verify the rescaling calculation, correct the table if "
            "necessary, and update the corresponding claims in the abstract and "
            "conclusion?",
            "Please also clarify whether the rescaling factors were estimated using a "
            "separate calibration period or the evaluation observations themselves. The "
            "latter would provide a retrospective comparison rather than evidence of "
            "out-of-sample calibrated intervals.",
        )
    ),
    Label("Response"),
    Para(
        "The reviewer is right on both points. The first exposed an error in our "
        "computation and not in the typesetting of the table, and the second identified "
        "something the earlier text should have stated and did not."
    ),
    Para(
        "**What was wrong.** For fixed interval centres a factor above one can only add "
        "outcomes to an interval, so a method that already covers 0.839 cannot need a "
        "factor of 1.60 to reach 0.80. The two columns of the earlier table had been "
        "computed on different observations by different rules. The coverage was computed "
        "on all observations. The scale factor was the 80th percentile of the absolute "
        "error divided by half the interval width, computed only on the observations "
        "whose interval had positive width, which for seasonal naive on ROBOD is 57% of "
        "them, and with half the total width used in place of the half-width on the side "
        "of the error. A factor obtained in that way does not bring the coverage of the "
        "full sample to 0.80. We checked this directly: applied to all observations, the "
        "earlier factors give a coverage of 0.844 for seasonal naive and 0.836 for "
        "Toto-2.0. The width of 3.75 persons, and the ratio of 4.9 derived from it, were "
        "therefore not widths at a matched coverage. Both are withdrawn, together with "
        "the corresponding statement in our previous response (R2.26)."
    ),
    Para(
        "**The corrected procedure.** Section 7.5 and Table 8 are replaced, and the "
        "rescaling rule is now stated as Equation (17). Each side of the interval is "
        "scaled separately about the median, so that an asymmetric interval keeps its "
        "shape. For each observation the score is the smallest factor at which the "
        "outcome is covered; it is zero when the outcome equals the median and infinite "
        "when the outcome lies on a side of zero width. The empirical coverage as a "
        "function of the factor is then the empirical distribution function of the scores "
        "over all observations. It is non-decreasing in the factor by construction, and "
        "at a factor of one it equals the published coverage identically. The analysis "
        "script asserts that identity for every method before it reports anything, so "
        "the inconsistency the reviewer found cannot recur silently."
    ),
    Para(
        "**Calibration period or evaluation observations.** The factors of the earlier "
        "table were estimated on the evaluation observations themselves. The comparison "
        "was therefore retrospective, as the reviewer suspected, and the earlier text did "
        "not say so. Table 8 now reports both kinds of estimate side by side and labels "
        "them:"
    ),
    Bullets(
        (
            "**Retrospective.** The smallest factor whose coverage on the evaluation "
            "observations reaches the target. It is descriptive and could not have been "
            "applied in operation.",
            "**Out of sample, single split.** The factor is the split conformal quantile "
            "of the scores on the first half of the test origins and is applied unchanged "
            "to the second half. Enough origins are removed between the two halves that "
            "no calibration target window overlaps an evaluation window. This gives 130 "
            "calibration and 130 evaluation origins on ROBOD, and 93 and 91 on HZMetro.",
            "**Out of sample, expanding window.** The factor applied at an origin is "
            "estimated from all earlier origins outside the same gap, after a burn-in of "
            "one quarter of the origins (195 origins scored on ROBOD, 138 on HZMetro).",
        )
    ),
    Para(
        "The foundation models remain zero-shot in both out-of-sample schemes: a single "
        "scalar per method is estimated, never from the observations on which it is "
        "evaluated. The new paragraph \u201cWhere the factors come from\u201d in Section "
        "7.5 answers the reviewer's question in these terms."
    ),
    Para(
        "**What the corrected analysis shows on ROBOD.** The corrected table contains the "
        "correction the reviewer anticipated and a finding we had not anticipated. "
        "Horizon 60 minutes, 15,660 observations from 261 origins."
    ),
    Bullets(
        (
            "**The factor at the nominal 80% level is exactly 1.00, and exact matching is "
            "not attainable.** Room occupancy is zero on 69.4% of the scored "
            "observations. When a room is empty and a method's lower quantile is zero "
            "while its median is not, the outcome lies exactly on the lower endpoint of "
            "the interval and its score is exactly one. Between 8.0% (seasonal naive) and "
            "35.7% (Toto-2.0) of outcomes lie on an endpoint, so the coverage curve has a "
            "large step at a factor of one: the coverage of Toto-2.0 falls from 0.901 at "
            "a factor of one to 0.544 immediately below it, and that of seasonal naive "
            "from 0.839 to 0.759. The published interval is therefore already the "
            "narrowest rescaled interval that covers at least 80%, for every method with "
            "a non-degenerate interval, and no factor produces a coverage of exactly "
            "0.80. The factor for seasonal naive is 1.00 and not 1.60.",
            "**Against the seasonal baseline no rescaling is needed, because the "
            "comparison is one of dominance.** Toto-2.0 covers 0.901 of outcomes with a "
            "mean width of 1.17 persons; seasonal naive covers 0.839 with 2.35 persons. "
            "The seasonal interval is 2.0 times as wide and covers less. The same holds "
            "for all four zero-shot configurations.",
            "**At the coverage the foundation models attain, near 90%, the gap is "
            "larger.** The table below gives the comparison at a target of 0.90. We "
            "describe the seasonal interval as needing to be six to eight times wider and "
            "attach no weight to the decimal: on count data the scores of a seasonal "
            "forecast concentrate on a few values, so its factor moves in steps (it is "
            "exactly 3 and exactly 5 here).",
            "**Against the trained LightGBM no comparison at a common coverage is "
            "possible.** Its interval is wider than that of Toto-2.0 (1.90 against 1.17 "
            "persons) and covers more (0.944 against 0.901), and the step at a factor of "
            "one prevents it from being narrowed to 0.90. Neither dominates the other, "
            "and the manuscript now says so.",
        )
    ),
    Table(
        caption=(
            "ROBOD, 60 minutes, target coverage 0.90 (from Table 8 of the revised "
            "manuscript). Widths in persons."
        ),
        header=(
            "Method",
            "Published: coverage, width",
            "Retrospective: factor, coverage, width",
            "Out of sample, split: factor, coverage, width",
        ),
        rows=(
            ("Toto-2.0", "0.901, 1.17", "1.00, 0.901, 1.17", "1.06, 0.921, 1.47"),
            ("Seasonal naive", "0.839, 2.35", "3.00, 0.902, 7.04", "5.00, 0.919, 12.28"),
            ("Width ratio, seasonal to Toto-2.0", "2.0", "6.0", "8.4"),
        ),
        widths=(1.7, 1.5, 1.9, 2.1),
        align="lccc",
    ),
    Para(
        "**What it shows on HZMetro.** There the scores are effectively continuous (at "
        "most 2.4% of outcomes lie on an endpoint for any method) and exact matching is "
        "attainable. Retrospectively, TimesFM-3 in multivariate mode is narrower than "
        "seasonal naive by 15.6% at a common coverage of 0.80 (99.7 against 118.2 "
        "passengers per interval) and by 25.1% at 0.90 (133.5 against 178.2). Out of "
        "sample the advantage is not reproduced cleanly. In the single split at a target "
        "of 0.80 the two methods attain the same coverage on the evaluation period, 0.820 "
        "and 0.821, and it is seasonal naive that is narrower, 83.6 against 92.8 "
        "passengers, by 9.9%. In the other three out-of-sample comparisons TimesFM-3 is "
        "narrower but also covers less, by 0.02 to 0.03, so the widths are not compared "
        "at a common coverage. The direction of the out-of-sample comparison depends on "
        "the calibration scheme, and we conclude that a sharpness advantage for the "
        "foundation models on HZMetro is not established."
    ),
    Para(
        "**Consequences for the claims.** The claim of a 4.9-fold advantage is removed "
        "from the abstract, the introduction and the conclusion, as the reviewer "
        "requested, and also from the statement of the outcome of Hypothesis 3 in Section "
        "5, the discussion and the limitations. What the manuscript now claims is this. "
        "The foundation models are not better calibrated than the seasonal baseline. On "
        "room occupancy the best foundation model's interval is half as wide as the "
        "seasonal one while covering more outcomes, and the seasonal interval must be "
        "widened 6.0 times retrospectively, or 8.4 times with a factor estimated on a "
        "preceding calibration period, to reach a common coverage near 90%. On metro flow "
        "a sharpness advantage holds retrospectively and is not established out of "
        "sample. Hypothesis 3 remains \u201cpartly refuted\u201d. The probabilistic claim "
        "of the paper now rests principally on CRPS, a proper scoring rule that rewards "
        "calibration and sharpness jointly and needs no rescaling, and which is "
        "unaffected by this correction: 18.79 against 25.93 for seasonal naive at 60 "
        "minutes on HZMetro, with the foundation model leading at every horizon."
    ),
    Para(
        "We also disclose two degenerate rows that the corrected procedure makes "
        "visible. Last value on ROBOD is exactly right on 81.5% of observations with no "
        "interval at all, so on the literal criterion of width at 80% coverage it is the "
        "sharpest method in the table, which is a statement about the target and not "
        "about the forecaster. Historical average reaches 0.80 only at a factor of 21.5. "
        "We draw the general caution, in Section 7.5, that on zero-inflated count data "
        "neither coverage near the nominal level nor width at a fixed coverage is by "
        "itself evidence of a useful predictive distribution."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 7.5: new paragraphs \u201cWhat was wrong with the earlier "
            "table\u201d, \u201cThe rescaling rule\u201d with Equation (17), \u201cROBOD: "
            "exact matching at the nominal level is not attainable, and the reason is the "
            "data\u201d, \u201cROBOD: what can be said about sharpness\u201d, \u201cThe "
            "degenerate rows\u201d, \u201cHZMetro: an advantage in retrospect, not "
            "established out of sample\u201d and \u201cWhere the factors come from\u201d; "
            "the verdict on Hypothesis 3 is rewritten.",
            "Table 8 is replaced. It has four panels (ROBOD and HZMetro, targets 0.80 and "
            "0.90) and reports, for every method, the published coverage and width, the "
            "retrospective factor, coverage and width, and the out-of-sample factor, "
            "coverage and width under the single split and the expanding window.",
            "Abstract, Section 1 (two places), Section 5, Section 8 (discussion and "
            "limitations) and Section 9 (conclusion) are updated to the corrected claim.",
        )
    ),
    # ------------------------------------------------------------------ 2
    Heading("Comment 2. Percentage improvement against LightGBM (Section 7.2, Table 4)", 1),
    Label("Reviewer comment"),
    Quote(
        (
            "The revised LightGBM comparison is a valuable addition and changes how we "
            "interpret the forecasting results.",
            "However, the manuscript reports that the per-series LightGBM model achieves "
            "0.511 MASE against 0.613 for the best zero-shot configuration on HZMetro, "
            "describing this as a 28.9% advantage.",
            "Using the zero-shot model as the reference, these values imply approximately "
            "a 16.6% reduction in MASE.",
            "Could the authors verify the percentage calculation and ensure the reported "
            "improvement is consistent across the abstract, results, discussion and "
            "conclusion? If the 28.9% figure uses another denominator or aggregation, that "
            "definition should be stated explicitly.",
        )
    ),
    Label("Response"),
    Para(
        "The reviewer's arithmetic is correct: 100 \u00d7 (1 \u2212 0.511 / 0.613) = "
        "16.6%. The figure of 28.9% did use another loss and another denominator, and the "
        "text did not say so."
    ),
    Para(
        "**Where 28.9% came from.** The significance test for this comparison is computed "
        "on the paired per-origin MAE, which is 23.61 passengers per interval for the "
        "local model and 30.43 for the zero-shot model over the 196 origins. The figure of "
        "28.9% was the excess of the larger of these over the smaller, 100 \u00d7 (30.43 / "
        "23.61 \u2212 1). It therefore differed from every other percentage in the paper "
        "in two respects, the loss (MAE and not MASE) and the denominator (the better "
        "method and not the worse), and it was printed beside MASE values with which it "
        "could not be reconciled."
    ),
    Para(
        "**The definition, now stated explicitly.** Section 6.4 has a new paragraph, "
        "\u201cDefinition of a relative improvement\u201d, with Equation (16). For a loss "
        "L that is lower for the better method, the relative improvement of a method A "
        "over a reference method B is 100 \u00d7 (1 \u2212 L~A~ / L~B~). The reference is "
        "always the worse of the two methods, and the loss is the MASE unless another "
        "loss is named. Under this definition the advantage of the local model on HZMetro "
        "at 120 minutes is **16.6% on MASE**, the quantity tabulated in Table 4, and "
        "22.4% on the paired MAE. The two differ because MASE divides each series by its "
        "own scale before averaging, whereas the mean MAE is dominated by the busiest "
        "stations. We report the MASE figure throughout. The test statistic and its "
        "p-value (p = 0.014, significant under Holm control) are unaffected, since they "
        "never depended on the percentage."
    ),
    Para(
        "**Consistency.** The figure is now 16.6% in the abstract, the introduction, the "
        "statement of the outcome of Hypothesis 1 in Section 5, Section 7.2, the "
        "discussion and the conclusion, and each of the abstract and the conclusion "
        "carries the definition in words. Section 7.2 also reports the advantage at each "
        "horizon under the same definition: 8.1%, 9.4%, 13.9% and 16.6% at 15, 30, 60 and "
        "120 minutes. Section 7.2 retains one explanatory sentence stating that an "
        "earlier version gave 28.9% and why, so that a reader who compares versions is "
        "not left to guess."
    ),
    Para(
        "**The companion figure was affected in the same way and is also corrected.** The "
        "advantage of the best zero-shot model over the best local model on ROBOD was "
        "given as 26.8%, which was likewise the figure on the paired MAE (0.450 against "
        "0.615 persons). On the MASE of Table 4 (0.374 against 0.512) it is 26.9%, and "
        "that is what the manuscript now reports. We checked the remaining percentages "
        "that compare two methods against Equation (16) and found them consistent with "
        "it."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 6.4: new paragraph \u201cDefinition of a relative improvement\u201d "
            "with Equation (16).",
            "Section 7.2: 16.6% with its definition, the per-horizon figures, and the "
            "explanation of the earlier 28.9%; 26.9% for ROBOD with the MAE figure given "
            "beside it.",
            "Abstract, Section 1, Section 5, Section 8 and Section 9: 28.9% replaced by "
            "16.6% and 26.8% by 26.9%.",
        )
    ),
    # ------------------------------------------------------------------ 3
    Heading(
        "Comment 3. Statistical interpretation of the reconciliation improvement "
        "(Section 7.7, Table 10)",
        1,
    ),
    Label("Reviewer comment"),
    Quote(
        (
            "The addition of moving-block bootstrap confidence intervals is helpful. "
            "Nevertheless, the interpretation of the proposed method's improvement over "
            "the raw forecast requires clarification.",
            "For TimesFM-3, Table 10 reports a mean MAE change of -0.98% with a 95% "
            "bootstrap confidence interval of [-1.41%, -0.61%]. This interval excludes "
            "zero. At the same time, the paired signed-rank test reports p = 0.18, which "
            "the manuscript uses to conclude that the improvement is not statistically "
            "established.",
            "These findings are not necessarily contradictory because they concern "
            "different aspects of the paired-difference distribution. The bootstrap "
            "interval evaluates the mean, while the signed-rank test has different "
            "assumptions and does not generally provide a valid test of the median under "
            "an asymmetric distribution.",
            "I suggest distinguishing evidence for a mean improvement from evidence for a "
            "typical-origin improvement, using a sign test or bootstrap interval for the "
            "median if the latter is of interest.",
            "Given the reported strong serial dependence, it would also be useful to "
            "indicate whether the bootstrap conclusions are sensitive to the block length "
            "of ten origins.",
        )
    ),
    Label("Response"),
    Para(
        "We agree with the reviewer's analysis and have adopted both suggestions. The "
        "earlier text had in fact gone further than the reviewer says: it described the "
        "signed-rank result as a statement about the median, which it is not. That "
        "description is removed. The mean and the typical origin are now tested "
        "separately, each by a method suited to it, and the result is more informative "
        "than either statistic was alone."
    ),
    Para(
        "**What is reported.** Section 6.4 has a new paragraph, \u201cMean and "
        "typical-origin effects\u201d, which defines three quantities for every paired "
        "comparison of the reconciliation study: the mean paired change with a "
        "moving-block bootstrap interval; the median paired change with a moving-block "
        "bootstrap interval of its own; and the number of origins improved, worsened and "
        "unchanged, with an exact binomial sign test on the untied origins. The new Table "
        "11 reports all three for every comparison. The rows for the proposed projection "
        "against the raw forecast are reproduced below."
    ),
    Table(
        caption=(
            "Proposed projection against the raw forecast, 300 origins per configuration "
            "(from Table 11). Changes in MAE as a percentage of the raw MAE; negative is "
            "an improvement. Intervals are 95% moving-block bootstrap intervals with "
            "blocks of ten origins."
        ),
        header=(
            "Configuration, forecaster",
            "Mean change, 95% interval",
            "Median change, 95% interval",
            "Improved / worsened / tied",
            "Sign test p",
            "Signed-rank p",
        ),
        rows=(
            (
                "Baseline, TimesFM-3",
                "\u22120.98 [\u22121.41, \u22120.61]",
                "+0.013 [+0.011, +0.015]",
                "89 / 211 / 0",
                "1.4\u00a0\u00d7\u00a010^\u221212^",
                "0.18",
            ),
            (
                "Baseline, LightGBM",
                "\u22120.10 [\u22120.39, +0.19]",
                "0.000 [0.000, 0.000]",
                "69 / 46 / 185",
                "0.040",
                "0.11",
            ),
            (
                "Crowded, TimesFM-3",
                "\u22121.25 [\u22121.69, \u22120.82]",
                "+0.006 [+0.003, +0.007]",
                "121 / 179 / 0",
                "9.7\u00a0\u00d7\u00a010^\u22124^",
                "0.47",
            ),
            (
                "Crowded, LightGBM",
                "+0.17 [\u22120.21, +0.41]",
                "0.000 [0.000, 0.000]",
                "59 / 65 / 176",
                "0.65",
                "0.30",
            ),
        ),
        widths=(1.45, 1.55, 1.8, 1.15, 0.95, 0.7),
        align="lccccc",
    ),
    Bullets(
        (
            "**The mean.** For TimesFM-3 the interval for the mean excludes zero at both "
            "operating points, so a reduction in mean error is statistically established "
            "for this forecaster on the simulator. For LightGBM the interval contains "
            "zero at both, so no change in mean error is established in either "
            "direction.",
            "**The typical origin.** For TimesFM-3 the interval for the median also "
            "excludes zero, on the other side. The projection worsens 211 of the 300 "
            "origins and improves 89, and the sign test rejects equal odds at "
            "p\u00a0=\u00a01.4\u00a0\u00d7\u00a010^\u221212^. It is therefore established, and "
            "not merely suggested, that the typical origin is made worse, by an amount about "
            "two orders of magnitude smaller than the mean improvement. For LightGBM the "
            "median change is exactly zero, because the projection leaves the forecast "
            "unchanged on 185 of the 300 origins.",
            "**Why the signed-rank test rejects neither.** The paired differences for "
            "TimesFM-3 are strongly left-skewed (sample skewness \u22124.9): many small "
            "positive differences and a minority of large negative ones. The numerous "
            "low-ranked positive differences and the few high-ranked negative ones "
            "nearly balance in the rank sum, and the test returns p = 0.18. Its null "
            "hypothesis is symmetry about zero, which is false here in a way the "
            "statistic happens not to detect. Its failure to reject is evidence neither "
            "that the mean is zero nor that the median is. We keep the statistic in Table "
            "10 for continuity and draw no conclusion from it, and the sentence that used "
            "it to conclude that the improvement is not statistically established has "
            "been removed.",
        )
    ),
    Para(
        "The manuscript now states the result at the strength the evidence supports: for "
        "TimesFM-3 the projection reduces the mean error by about one percent, which is "
        "established on the simulator, while making the typical origin worse by about one "
        "hundredth of one percent, which is also established; for LightGBM no effect on "
        "the mean is established. The reduction is a statement about the mean and not "
        "about most forecasts, and the abstract-level summary in the conclusion says so "
        "in those words."
    ),
    Para(
        "**Sensitivity to the block length.** The reviewer refers to the strong serial "
        "dependence we reported. That dependence, a lag-one autocorrelation of 0.57 on "
        "ROBOD and 0.92 on HZMetro, belongs to the loss differential between two "
        "forecasters on the real datasets (Section 7.10). The paired difference between "
        "the reconciled and the raw forecast on the simulator is a different quantity and "
        "is nearly uncorrelated between origins: its lag-one autocorrelation is \u22120.09 "
        "for TimesFM-3 and \u22120.02 for LightGBM. We therefore examined the block length "
        "in both places, at 1, 5, 10, 20 and 40 origins, the first being the ordinary "
        "bootstrap that assumes independent origins."
    ),
    Table(
        caption=(
            "95% bootstrap intervals against the block length. The first four rows are "
            "from the new Table 12 (paired change in MAE, percent). The last two rows are "
            "the interval of the headline real-data test of Table 16 (mean loss "
            "differential, in persons on ROBOD and passengers per interval on HZMetro), "
            "now reported in Section 7.10."
        ),
        header=("Quantity", "1", "5", "10", "20", "40"),
        rows=(
            ("Reconciliation against the raw forecast, simulator", "", "", "", "", ""),
            (
                "Baseline, TimesFM-3, mean",
                "[\u22121.48, \u22120.54]",
                "[\u22121.41, \u22120.59]",
                "[\u22121.41, \u22120.61]",
                "[\u22121.33, \u22120.61]",
                "[\u22121.32, \u22120.62]",
            ),
            (
                "Baseline, TimesFM-3, median",
                "[+0.010, +0.015]",
                "[+0.011, +0.015]",
                "[+0.011, +0.015]",
                "[+0.011, +0.015]",
                "[+0.011, +0.015]",
            ),
            (
                "Baseline, LightGBM, mean",
                "[\u22120.43, +0.22]",
                "[\u22120.43, +0.23]",
                "[\u22120.39, +0.19]",
                "[\u22120.33, +0.12]",
                "[\u22120.32, +0.15]",
            ),
            (
                "Crowded, TimesFM-3, mean",
                "[\u22121.81, \u22120.75]",
                "[\u22121.76, \u22120.78]",
                "[\u22121.69, \u22120.82]",
                "[\u22121.62, \u22120.83]",
                "[\u22121.51, \u22120.89]",
            ),
            ("Headline zero-shot advantage, real data", "", "", "", "", ""),
            (
                "ROBOD, Toto-2.0 against last value",
                "[\u22120.276, \u22120.106]",
                "[\u22120.343, \u22120.061]",
                "[\u22120.364, \u22120.045]",
                "[\u22120.370, \u22120.060]",
                "[\u22120.393, \u22120.067]",
            ),
            (
                "HZMetro, TimesFM-3 against seasonal naive",
                "[\u22127.37, \u22122.13]",
                "[\u221210.59, +0.14]",
                "[\u221211.70, +1.46]",
                "[\u221212.06, +2.71]",
                "[\u221210.44, +2.61]",
            ),
        ),
        widths=(2.1, 1.05, 1.05, 1.05, 1.05, 1.05),
        align="lccccc",
        panels=(0, 5),
    ),
    Bullets(
        (
            "**Reconciliation comparisons.** No conclusion changes at any block length. "
            "For TimesFM-3 the interval for the mean excludes zero and the interval for "
            "the median lies above zero at every length and at both operating points; "
            "for LightGBM the interval for the mean contains zero at every length. The "
            "six ablation intervals of Table 10 exclude zero at every block length.",
            "**Headline real-data test.** This is where the block length could matter, "
            "and we had not reported it. On ROBOD the interval excludes zero at every "
            "length. On HZMetro it contains zero at every block length from five "
            "upwards and excludes zero only at a block of one, that is, only if the "
            "origins are treated as independent. Both conclusions of Table 16 are "
            "therefore stable once dependence is allowed for at all, and the HZMetro "
            "row shows what the adjustment is for. We add one caution in the text: the "
            "autocorrelation of the HZMetro differential is still 0.56 at a lag of ten "
            "origins, so a block of ten does not exhaust the dependence, which is why "
            "the longer blocks are reported.",
        )
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 6.4: new paragraph \u201cMean and typical-origin effects\u201d.",
            "Section 7.7: the paragraph on the whole layer against the raw forecast is "
            "rewritten under the headings \u201cThe mean\u201d, \u201cThe typical "
            "origin\u201d, \u201cWhy the signed-rank test rejects neither\u201d and "
            "\u201cSensitivity to the block length\u201d; new Tables 11 and 12.",
            "Table 10: the caption now states what the signed-rank test does and does not "
            "test, and that its p-values are one-sided in the three ablation blocks and "
            "two-sided in the last block.",
            "Section 7.10: new paragraph \u201cSensitivity to the block length\u201d for "
            "the headline real-data test.",
            "Section 9: the conclusion states that the reduction is established for the "
            "mean and not at the typical origin.",
        )
    ),
    # ------------------------------------------------------------------ 4
    Heading(
        "Comment 4. Conservation residuals under per-quantile projection "
        "(Section 7.8, Table 11, now Table 14)",
        1,
    ),
    Label("Reviewer comment"),
    Quote(
        (
            "The correction restricting exact probabilistic coherence claims to the "
            "reconciled median is appropriate, and the revised theoretical explanation is "
            "much clearer.",
            "However, Table 11 reports a median conservation residual of approximately "
            "zero for the shift method and 0.006 persons for the per-quantile method. "
            "Since both strategies project the same median forecast onto the same "
            "admissible set, I would expect their median projections to agree under "
            "identical weights and solver settings.",
            "The per-quantile residuals are also described as solver-tolerance effects, "
            "although the median projection elsewhere achieves residuals several orders "
            "of magnitude smaller.",
            "Could the authors explain this discrepancy, report the relevant solver "
            "tolerances and confirm that the implementations use identical constraints "
            "and weighting at the median?",
            "I appreciate that Section 4.4 now explicitly acknowledges that individually "
            "feasible marginal quantile paths do not constitute a coherent joint "
            "predictive distribution. This important distinction should remain consistent "
            "throughout the manuscript and response.",
        )
    ),
    Label("Response"),
    Para(
        "The reviewer's expectation is correct and the suspicion was justified. Two "
        "statements in the earlier text were wrong: that the per-quantile residuals were "
        "an effect of solver tolerance, and that the crossing rate before any monotone "
        "repair was zero. The table concerned is now Table 14."
    ),
    Para(
        "**The implementations are identical at the median.** Both strategies call the "
        "same routine on the same raw median, with the same weights, obtained from the "
        "same quantile spread, the same conservation equalities, the same box and the "
        "same solver settings. Before any further processing the two medians agree to "
        "within 4.2\u00a0\u00d7\u00a010^\u22125^ persons at every coordinate of every origin."
    ),
    Para(
        "**Solver tolerances.** They are now stated in Section 4.4. The quadratic program "
        "is solved with OSQP through CVXPY with absolute and relative tolerances of 10^\u22127^, "
        "a budget of 50,000 iterations and solution polishing. The returned point is "
        "clipped to the box, and a solve is rejected with an error if the clipped point "
        "leaves a conservation residual above 10^\u22124^ persons in any equation."
    ),
    Para(
        "**Solver tolerance was not the explanation.** As projected, the mean absolute "
        "residual at each of the nine quantile levels is below 10^\u22129^ persons, the median "
        "included, and the largest residual in any single equation at any origin is "
        "3.8\u00a0\u00d7\u00a010^\u22126^ persons. These are the magnitudes the tolerances imply, "
        "and the mean residuals are more than six orders of magnitude below the 0.003 to 0.012 "
        "persons of the earlier table."
    ),
    Para(
        "**The explanation is the monotone rearrangement.** Nine independent projections "
        "do not preserve the order of the quantile levels: 11.3% of adjacent quantile "
        "pairs cross, and 73% of the origin and forecaster pairs contain at least one "
        "crossing. The implementation therefore sorts the nine projected values of every "
        "coordinate into ascending order. That step moves between 9.4% and 21.7% of the "
        "coordinates of a level to a different level, including 19.6% of the coordinates "
        "of the median. Each rearranged path is then a mixture, coordinate by coordinate, "
        "of several paths that were each coherent, and such a mixture is not coherent, "
        "because the conservation constraint couples coordinates within a path. The "
        "residuals of 0.003 to 0.012 persons, including the 0.006 at the median, are "
        "produced entirely by this step. The rearranged median differs from the projected "
        "median by up to 2.2 persons at a single coordinate. The bound violation stays at "
        "zero, because a coordinatewise sort of values inside an interval remains inside "
        "it."
    ),
    Para(
        "**Our error on the crossing rate.** The crossing rate of zero that we reported "
        "had been measured after the rearrangement, where it is zero by construction. "
        "The statement that it was measured before any repair was an error in the "
        "analysis script. It is corrected, and the corrected analysis measures every "
        "quantity both before and after the rearrangement."
    ),
    Table(
        caption=(
            "Conservation residual in persons by quantile level (from Table 14), on 120 "
            "origin and forecaster pairs of the simulated building."
        ),
        header=("Strategy", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"),
        rows=(
            (
                "Shift",
                "0.044",
                "0.041",
                "0.044",
                "0.039",
                "0.000",
                "0.033",
                "0.058",
                "0.068",
                "0.085",
            ),
            (
                "Per quantile, as projected",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
                "0.000",
            ),
            (
                "Per quantile, rearranged",
                "0.012",
                "0.009",
                "0.008",
                "0.008",
                "0.006",
                "0.004",
                "0.004",
                "0.003",
                "0.004",
            ),
            (
                "Share of coordinates moved by the rearrangement, %",
                "21.7",
                "21.0",
                "20.4",
                "20.3",
                "19.6",
                "17.9",
                "16.7",
                "13.4",
                "9.4",
            ),
        ),
        widths=(2.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
        align="lccccccccc",
    ),
    Para(
        "**Consistency of the distinction between marginal paths and a joint "
        "distribution.** The measurement makes the distinction sharper than the earlier "
        "text did, and the manuscript now states it in the same terms in Section 4.4, "
        "Section 7.8, the limitations and the conclusion. Projecting quantiles "
        "independently yields either exactly coherent and feasible paths that cross, or "
        "ordered and feasible paths that are only approximately coherent, but not both. "
        "Neither is a coherent joint predictive distribution: the unsorted paths are not "
        "the quantiles of any law, because they cross, and for neither output is there a "
        "joint law over the series of which the paths are the marginal quantiles. The "
        "guarantee of the paper covers the projected median path under the default shift "
        "strategy and nothing more. In particular, the per-quantile strategy as "
        "implemented does not return a coherent median, whereas the shift does, which is "
        "a further reason for keeping the shift as the default. A coherent predictive "
        "distribution requires projecting sample paths and is left to future work, as "
        "before. We have checked that nothing in the manuscript or in this response "
        "describes the output as a coherent distribution."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 4.4: the solver tolerances are stated; the description of "
            "per-quantile reconciliation now includes the rearrangement and its "
            "consequence for coherence.",
            "Section 7.8: new paragraph \u201cWhy the per-quantile residual was not "
            "zero\u201d, with the measurements above and the three consequences that "
            "follow.",
            "Table 14 (formerly Table 11): the single per-quantile row is replaced by "
            "\u201cas projected\u201d and \u201crearranged\u201d rows for the residual and "
            "the bound violation, and a row giving the share of coordinates moved.",
            "Section 9: the sentence on per-quantile projection is rewritten accordingly.",
        )
    ),
    # ------------------------------------------------------------------ 5
    Heading("Comment 5. Repository consistency and reproducibility (Response R2.20)", 1),
    Label("Reviewer comment"),
    Quote(
        (
            "The authors state that the public repository has been updated to reflect all "
            "completed experiments and revised results.",
            "However, the publicly accessible README on the repository's main branch still "
            "indicates that E1-E6 have not been executed and retains earlier forecasting "
            "results and statements concerning three-seed evaluation.",
            "This does not appear consistent with the revised manuscript or the authors' response.",
            "Could the authors ensure that the public repository contains the updated "
            "documentation, experiment configurations and a clearly identified version "
            "corresponding to the revised manuscript?",
        )
    ),
    Label("Response"),
    Para(
        "The reviewer is right, and we apologise. The corrections described in our "
        "response R2.20 had been made in our working copy and had not been published to "
        "the public branch, so the public README was still that of the original "
        "submission. What we wrote in R2.20 was true of the working copy and not of the "
        "repository a reader can check, which is the only one that matters. The "
        "repository has now been brought into line with the manuscript, and the version "
        "that corresponds to it is identified by a tag."
    ),
    Bullets(
        (
            "**A clearly identified version.** The tag `manuscript-r2` on "
            "https://github.com/001shahab/zero-shot-FMF marks the commit that corresponds "
            "to this manuscript. The README opens with a \u201cVersion of record\u201d "
            "section that names the tag, the manuscript source, the analysis scripts and "
            "the configuration directory.",
            "**Documentation.** The README has been rewritten. It carries the title of the "
            "manuscript, states the five hypotheses with the outcome the manuscript "
            "reports for each, and reports the results of this manuscript and no others; "
            "the tables of the original submission are removed. Its experiment registry "
            "lists every experiment family with its configuration file, whether it was "
            "run, and whether the manuscript uses it. The three families that remain "
            "unexecuted, the covariate ablation (E2), the sensor-degradation study (E5) "
            "and the decision-utility study (E6), are exactly those on which the "
            "limitations section of the manuscript makes no claim. The registry also "
            "states that E1 and E3 were never executed as separate experiments, because "
            "the questions they asked are answered by E11 and E12, and that E4 was run "
            "and superseded by E11.",
            "**Seeds.** A section headed \u201cSeeds are not replicates\u201d states what "
            "Section 6.3 of the manuscript states: every forecaster except the "
            "gradient-boosted model is deterministic, repeated seeds reproduce identical "
            "forecasts, all inference runs on distinct origins from a single seed, and no "
            "result is a three-seed result. The counts given are the counts of distinct "
            "origins used in the manuscript: 261 on ROBOD, 181 to 196 on HZMetro and 300 "
            "on each simulator configuration.",
            "**Experiment configurations.** All configurations are in "
            "`configs/experiments/`. The README has a table, \u201cThe configuration "
            "behind every table\u201d, that maps each table and figure of the manuscript "
            "to the configuration file that produced its runs and to the script that "
            "generates it.",
            "**The analyses of this revision.** The repository contains "
            "`analysis_round2.py`, its logged output `summary_round2.json`, the per-origin "
            "tables it writes, and `verify_round2.py`. The summary file is committed, so "
            "the check of the manuscript against the logged numbers can be run without "
            "the model weights or the datasets. The source of the revised manuscript is "
            "included at the tag, since it is the text the verification script reads.",
        )
    ),
    Para(
        "The manuscript records this in two places. The limitations section states that "
        "the correction announced in the first revision had not reached the public branch "
        "when that revision was reviewed, and the Reproducibility statement identifies "
        "the tagged version and describes what the repository contains at that tag."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 8 (limitations): one sentence added on the repository.",
            "Reproducibility statement: new paragraph identifying the tag "
            "`manuscript-r2` and the scripts of this revision.",
        )
    ),
    # ------------------------------------------------------------------ 6
    Heading("Comment 6. Interpretation of the seasonal-strength hypothesis (Section 7.4)", 1),
    Label("Reviewer comment"),
    Quote(
        (
            "The within-dataset test of the seasonal-strength hypothesis is a useful "
            "improvement over the original three-dataset comparison.",
            "However, the manuscript continues to describe the hypothesis as "
            "\u201crefuted\u201d, although the estimated slope is negative and its 95% "
            "confidence interval includes both zero and potentially meaningful negative "
            "effects.",
            "The result indicates insufficient evidence to establish the proposed "
            "relationship, rather than evidence that no such relationship exists.",
            "I suggest describing the hypothesis as \u201cnot supported by the "
            "within-dataset analysis\u201d or \u201cnot statistically established\u201d "
            "rather than refuted.",
        )
    ),
    Label("Response"),
    Para(
        "We agree and have adopted the reviewer's wording. Hypothesis 2 is now described "
        "as \u201cnot supported by the within-dataset analysis\u201d, and the dependence as "
        "\u201cnot statistically established\u201d, everywhere it is mentioned: the "
        "abstract, the introduction, the list of contributions, the statement of the "
        "outcome in Section 5, the title and the verdict of Section 7.4, the discussion, "
        "the limitations and the conclusion."
    ),
    Para(
        "Section 7.4 also explains the choice of words, so that the distinction the "
        "reviewer draws is visible to the reader. The falsification criterion we had "
        "stated in advance, a slope not distinguishable from zero, is met, so the "
        "hypothesis fails the test we set for it. That is a failure to establish the "
        "relationship and not a demonstration that it is absent. The interval for the "
        "slope, [\u221238, +4] percentage points per unit of seasonal strength, is wide. "
        "Across the range of seasonal strength observed on this panel, 0.57 to 0.89, its "
        "lower end corresponds to a decline of about twelve percentage points in the "
        "zero-shot advantage, which would matter against a median advantage of 14.3%, and "
        "its upper end to a rise of about one point. The data are compatible both with no "
        "dependence and with a meaningful negative dependence, and a single panel in "
        "which the predictor varies over a range of only 0.32 cannot decide between "
        "them."
    ),
    Para(
        "In the same spirit we now disclose a figure that points in the reviewer's "
        "direction and that the earlier text reported without its p-value. The rank "
        "correlation between advantage and seasonal strength is \u22120.17, which taken "
        "alone carries p = 0.032. We do not read it as confirmation, since the slope and "
        "not the rank correlation was the test specified in advance and the pair is not "
        "corrected for multiplicity, but it is a further reason not to describe the "
        "hypothesis as disproved."
    ),
    Para(
        "The word \u201crefuted\u201d now appears in the manuscript only where the evidence "
        "contradicts a claim and does not merely fail to support it: for Hypothesis 1, "
        "which required the foundation models to beat every local baseline on both "
        "datasets and is contradicted on HZMetro, and for the calibration half of "
        "Hypothesis 3, where the seasonal baseline is measurably closer to nominal "
        "coverage."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 7.4: title, verdict and closing remarks reworded; the explanation of "
            "the wording and the rank-correlation p-value added.",
            "Abstract, Section 1 (two places), Section 5, Section 8 (discussion and "
            "limitations) and Section 9: \u201crefuted\u201d replaced as described.",
        )
    ),
    # ------------------------------------------------------------------ 7
    Heading("Comment 7. Risk-sensitive evaluation (Sections 7.6 and 7.7)", 1),
    Label("Reviewer comment"),
    Quote(
        (
            "I appreciate the addition of the crowded simulation scenario and "
            "peak-occupancy error analysis. These steps make the reconciliation "
            "evaluation more relevant to crowd management.",
            "The reduction in peak error is encouraging, but a lower average peak error "
            "does not necessarily indicate improved identification of unsafe or "
            "near-capacity conditions.",
            "The authors' response also refers to high-occupancy MAE and Brier scores for "
            "capacity exceedance, but I could not identify the corresponding numerical "
            "results in the revised manuscript.",
            "If these analyses were completed, I encourage the authors to report their "
            "results, including the threshold definition and sample size. Otherwise, the "
            "response should be aligned with the evidence actually presented.",
            "The distinction between average forecasting accuracy, risk-sensitive "
            "prediction and demonstrated operational decision benefits should remain "
            "explicit.",
        )
    ),
    Label("Response"),
    Para(
        "The reviewer is right on all three points. The high-occupancy MAE and the Brier "
        "score had been computed and logged, and our response referred to them, but they "
        "were not printed in the manuscript. They are now reported in full, with "
        "thresholds and sample sizes, and they do not support the conclusion that the "
        "earlier text drew from the peak error. We have aligned both the manuscript and "
        "this response with the evidence."
    ),
    Para(
        "**Definitions and thresholds.** Section 6.4 has a new paragraph, "
        "\u201cRisk-sensitive quantities\u201d. All three quantities are defined on room "
        "occupancy, with c the capacity of the room."
    ),
    Bullets(
        (
            "**Peak error.** The absolute difference between the maximum of the median "
            "forecast and the maximum of the observed occupancy over the horizon, for one "
            "room, averaged over the rooms of an origin. It is an error of magnitude and "
            "carries no information about timing or about any threshold.",
            "**High-occupancy MAE.** The mean absolute error over the room-steps at which "
            "the observed occupancy is at or above 0.8 c.",
            "**Event detection.** The binary forecast that the median is at or above a "
            "forecast threshold is scored against the binary outcome that the observation "
            "is at or above an outcome threshold, by hit rate, number of false alarms and "
            "Brier score, which for a binary forecast equals the misclassification rate. "
            "Two events are scored. *Near capacity*: both thresholds are 0.8 c. *At "
            "capacity*: the outcome threshold is c and the forecast threshold is c "
            "\u2212 0.5, so that a forecast counts as predicting a full room when it "
            "rounds to the capacity as an integer count. Observed occupancy never exceeds "
            "capacity in the simulator, so the second event is a room being full and not "
            "capacity being exceeded, and the manuscript no longer speaks of "
            "exceedance.",
        )
    ),
    Para(
        "**Sample sizes.** Each configuration scores 300 origins for each of two "
        "forecasters; each origin contributes 8 rooms over 60 forecast steps, which is "
        "144,000 room-steps per forecaster. In the baseline configuration 5,914 of these "
        "(4.1%) are at or above 80% of capacity, spread over 82 origins, and 3,316 (2.3%) "
        "are at capacity, in 75 origins. In the crowded configuration the counts are "
        "15,307 (10.6%) in 103 origins and 11,460 (8.0%) in 99 origins. The table below "
        "pools the two forecasters; Table 13 of the manuscript also gives the paired "
        "change for each forecaster separately with a block-bootstrap interval."
    ),
    Table(
        caption=(
            "Risk-sensitive quantities for the raw forecast and the proposed projection "
            "(from Table 13), pooled over two forecasters: 600 origins and 288,000 "
            "room-steps per configuration."
        ),
        header=("Quantity", "Baseline raw", "Baseline proposed", "Crowded raw", "Crowded proposed"),
        rows=(
            ("Peak error, persons", "1.298", "1.217", "1.113", "1.068"),
            ("High-occupancy MAE, mean over origins", "4.319", "4.280", "2.827", "2.849"),
            ("High-occupancy MAE, pooled over room-steps", "3.408", "3.350", "1.855", "1.828"),
            (
                "Near capacity: 11,828 events (baseline), 30,614 events (crowded)",
                "",
                "",
                "",
                "",
            ),
            ("Hit rate", "43.7%", "45.0%", "80.1%", "80.6%"),
            ("False alarms", "2,095", "2,219", "3,643", "3,658"),
            ("Brier score", "0.0304", "0.0303", "0.0338", "0.0334"),
            (
                "At capacity: 6,632 events (baseline), 22,920 events (crowded)",
                "",
                "",
                "",
                "",
            ),
            ("Hit rate", "39.1%", "39.4%", "68.4%", "68.6%"),
            ("False alarms", "512", "469", "2,407", "2,312"),
            ("Brier score", "0.0158", "0.0156", "0.0335", "0.0330"),
        ),
        widths=(3.0, 1.0, 1.15, 1.0, 1.15),
        align="lcccc",
        panels=(3, 7),
    ),
    Bullets(
        (
            "**Peak error.** The reduction, 6.3% on the baseline configuration and 4.0% "
            "on the crowded one, is statistically established: all four paired "
            "block-bootstrap intervals exclude zero. It is the one risk-related effect "
            "that is established, and it is a statement about the magnitude of the "
            "predicted peak. It says nothing about whether the peak is predicted at the "
            "right time, in the right room, or on the right side of a threshold.",
            "**Error at high occupancy.** The evidence is mixed. Averaged over origins "
            "the error improves on the baseline configuration and deteriorates on the "
            "crowded one; pooled over room-steps it improves on both. Of the four paired "
            "intervals, two exclude zero in the favourable direction and two contain "
            "zero. An improvement in the error at high occupancy is not established.",
            "**Detection of near-capacity and at-capacity conditions.** This is the "
            "quantity the reviewer's objection concerns, and the projection changes it "
            "very little. Hit rates move by between 0.2 and 1.3 percentage points and the "
            "Brier scores change in the fourth decimal place. Of the eight paired "
            "intervals for the change in Brier score, four exclude zero, all in the "
            "favourable direction, with relative reductions between 0.7% and 1.8%; the "
            "other four contain zero. A forecast that misses six in ten full-room "
            "room-steps on the baseline configuration before the projection misses six "
            "in ten after it. We conclude, in the manuscript, that reconciliation does "
            "not materially improve the identification of unsafe conditions in these "
            "experiments.",
        )
    ),
    Para(
        "**The score our earlier response referred to.** Our response R2.28 mentioned a "
        "Brier score for predicted capacity exceedance. That score had been computed "
        "with a strict rule, under which a forecast predicts a full room only if it is "
        "numerically at or above capacity. We report it in Table 13 because it is the "
        "score that was logged, and we rely on it in neither direction, because for a "
        "box-constrained forecast it is not numerically well defined. The projection "
        "places every forecast that would exceed capacity on the bound, and the solver "
        "returns the bound only to within its tolerance, so whether the strict "
        "inequality holds is decided by the last digits of the solution. The evidence is "
        "direct: solving the identical problems from a warm start instead of a cold "
        "start changes the MAE by less than 10^\u22126^ persons at every origin, and changes "
        "the number of strict-rule hits from 3,991 to 4,510 on the crowded configuration, "
        "while every count under the half-person rule and under the near-capacity rule "
        "is identical. Under the strict rule the projection appears to make detection "
        "worse (hit rate 16.2% to 11.7% and 29.8% to 17.4%), which is an artefact of the "
        "same kind. This is why the at-capacity event is now defined with a margin of "
        "half a person, and the reference to an exceedance score should not have been "
        "made without that qualification."
    ),
    Para(
        "**Three levels of claim, kept explicit.** Section 7.7 closes with a paragraph, "
        "\u201cThree levels of claim\u201d, and the same separation is carried into the "
        "abstract, the discussion and the conclusion."
    ),
    Bullets(
        (
            "**Average accuracy.** The projection reduces the mean error of TimesFM-3 by "
            "about one percent and leaves that of LightGBM unchanged (Comment 3).",
            "**Risk-sensitive prediction.** The projection reduces the error in the "
            "magnitude of the predicted peak, which is established, and does not "
            "materially change the error at high occupancy or the detection of "
            "near-capacity and full-room conditions.",
            "**Operational decision benefit.** This was not evaluated at all. No decision "
            "rule was run, no lead time was measured and no cost was assigned to a missed "
            "event or a false alarm, because the decision-utility study of our original "
            "design remains unrun. No operational benefit is claimed.",
        )
    ),
    Para(
        "**Statements withdrawn.** The manuscript no longer says that the gain lands on "
        "the origins a crowd-protection rule needs, or that the peak result is "
        "operationally stronger than the MAE result. For the same reason we withdraw two "
        "sentences of our previous response, in R1.2 and R2.28, which described the peak "
        "result as the operationally relevant effect and as eleven times the mean "
        "effect. The comparison of the two percentages is arithmetically correct, but "
        "the inference to operational relevance was not supported. The abstract now "
        "states that reconciliation leaves the detection of near-capacity and "
        "at-capacity conditions essentially unchanged, so that no risk-sensitive or "
        "operational benefit is claimed. What the layer provides on the present evidence "
        "is a feasibility guarantee, a small and established reduction in mean error for "
        "one forecaster, and a more accurate predicted peak magnitude."
    ),
    Label("Changes in the manuscript"),
    Bullets(
        (
            "Section 6.4: new paragraph \u201cRisk-sensitive quantities\u201d with the "
            "definitions and thresholds.",
            "Section 7.7: the earlier paragraph on where the gain lands is replaced by "
            "\u201cRisk-sensitive evaluation: what is shown and what is not\u201d, with the "
            "sample sizes, the three sets of results, the strict rule and the three "
            "levels of claim; new Table 13.",
            "Abstract, Section 1, Section 8 and Section 9: the risk-sensitive and "
            "operational statements are narrowed as described.",
        )
    ),
    # ------------------------------------------------------------------ extra
    Heading("Further corrections made while preparing this revision", 1),
    Para(
        "Checking every quoted number against the logged analysis turned up five smaller "
        "inconsistencies that the reviewer did not raise. None changes a conclusion. We "
        "list them so that the reviewer does not have to find them."
    ),
    Bullets(
        (
            "**Table 10, last column.** For the comparison of the unbounded projection "
            "with MinT on TimesFM-3, the share of origins made worse was printed as 20%. "
            "On 29 of the 300 origins the two arms agree to within 10^\u221212^ persons, and "
            "those origins had been classified by the sign of rounding noise. Counting "
            "them as ties, 33 origins worsen, which is 11%. The entry is corrected.",
            "**Table 9, crowded configuration.** The peak error of the proposed "
            "projection was printed as 1.069 where the logged value is 1.0685 and the "
            "text already gave 1.068. The table entry is corrected.",
            "**Section 4.4, solve time.** The text retained a solve time of 10.2 ms from "
            "the original submission, whereas Table 9 and Section 7.12 report 6.2 ms "
            "from the runs of the first revision. The text now gives 6.2 ms, and Section "
            "7.12 states that the figure is a median.",
            "**Sections 8 and 9, units.** The effect sizes of the ablations were "
            "described as percentages of MASE. They are percentages of MAE, as in Table "
            "10, and the text now says so.",
            "**Section 9, quantile level.** The residual of 0.085 persons under the "
            "shift strategy was attributed to the 95th percentile. The highest level "
            "evaluated is the 90th, as in Table 14, and the text is corrected.",
        )
    ),
    Para(
        "We are grateful to the reviewer. Each of the seven comments either corrected an "
        "error or removed a claim that the evidence did not carry, and the manuscript is "
        "more accurate for it."
    ),
]


# --------------------------------------------------------------------------- #
# Inline markup
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|~[^~\s]+~|\^[^^\s]+\^)")


def spans(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into (style, content) pairs; style is '', b, i, code, sub or sup."""
    out: list[tuple[str, str]] = []
    for piece in _TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            out.append(("b", piece[2:-2]))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(("i", piece[1:-1]))
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            out.append(("code", piece[1:-1]))
        elif piece.startswith("~") and piece.endswith("~") and len(piece) > 2:
            out.append(("sub", piece[1:-1]))
        elif piece.startswith("^") and piece.endswith("^") and len(piece) > 2:
            out.append(("sup", piece[1:-1]))
        else:
            out.append(("", piece))
    return out


def all_text() -> str:
    """Every string of the response, for the checks that apply to all of it."""
    parts: list[str] = [TITLE, SUBTITLE, MANUSCRIPT, AUTHORS]
    for block in CONTENT:
        match block:
            case Heading(text=text) | Label(text=text) | Para(text=text):
                parts.append(text)
            case Quote(paragraphs=paragraphs):
                parts.extend(paragraphs)
            case Bullets(items=items):
                parts.extend(items)
            case Table(caption=caption, header=header, rows=rows):
                parts.append(caption)
                parts.extend(header)
                for row in rows:
                    parts.extend(row)
    return "\n".join(parts)


def check_content() -> None:
    """Refuse to build a response that breaks its own rules."""
    text = all_text()
    for forbidden, name in (("\u2014", "em dash"), ("\u2013", "en dash"), ("--", "double hyphen")):
        if forbidden in text:
            where = text[max(0, text.index(forbidden) - 40) : text.index(forbidden) + 40]
            raise AnswerError(f"the response contains an {name}: ...{where!r}...")
    for block in CONTENT:
        if isinstance(block, Table):
            columns = len(block.header)
            if len(block.widths) != columns or (block.align and len(block.align) != columns):
                raise AnswerError(f"table {block.caption!r}: widths or alignment do not match")
            for row in block.rows:
                if len(row) != columns:
                    raise AnswerError(f"table {block.caption!r}: a row has {len(row)} cells")
    headings = [b.text for b in CONTENT if isinstance(b, Heading) and b.level == 1]
    numbered = [h for h in headings if h.startswith("Comment ")]
    if [h.split(".")[0] for h in numbered] != [f"Comment {i}" for i in range(1, 8)]:
        raise AnswerError(f"the seven comments are not all answered in order: {numbered}")
    if sum(isinstance(b, Quote) for b in CONTENT) != 7:
        raise AnswerError("each of the seven comments must be quoted exactly once")


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #


def _shade(element: BaseOxmlElement, fill: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    element.append(shading)


def _left_rule(paragraph: Paragraph) -> None:
    """A vertical accent rule and a tint, marking the reviewer's own words."""
    properties = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), ACCENT_HEX)
    borders.append(left)
    properties.append(borders)
    _shade(properties, QUOTE_FILL)


def _set_font(run: Run, name: str) -> None:
    """Set the typeface for Latin and East Asian text alike, so Word cannot substitute."""
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)


def _add_runs(
    paragraph: Paragraph,
    text: str,
    *,
    size: float | None = None,
    italic: bool = False,
    bold: bool = False,
) -> None:
    for style, content in spans(text):
        run = paragraph.add_run(content)
        _set_font(run, "Menlo" if style == "code" else BODY_FONT)
        if size is not None:
            run.font.size = Pt(size - 1 if style == "code" else size)
        elif style == "code":
            run.font.size = Pt(9.5)
        run.bold = bold or style == "b"
        run.italic = italic or style == "i"
        if style == "sub":
            run.font.subscript = True
        elif style == "sup":
            run.font.superscript = True


def _page_number_field(paragraph: Paragraph) -> None:
    run = paragraph.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            char = OxmlElement("w:fldChar")
            char.set(qn("w:fldCharType"), kind)
            run._element.append(char)
        else:
            instruction = OxmlElement("w:instrText")
            instruction.set(qn("xml:space"), "preserve")
            instruction.text = text
            run._element.append(instruction)
    run.font.size = Pt(9)
    run.font.name = BODY_FONT


def _cell_text(
    cell: _Cell, text: str, *, bold: bool, align: str, size: float, italic: bool = False
) -> None:
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(1)
    paragraph.paragraph_format.space_before = Pt(1)
    paragraph.alignment = {
        "l": WD_ALIGN_PARAGRAPH.LEFT,
        "c": WD_ALIGN_PARAGRAPH.CENTER,
        "r": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    _add_runs(paragraph, text, size=size, bold=bold, italic=italic)


def _docx_table(document: DocumentType, block: Table, text_width: float) -> None:
    caption = document.add_paragraph()
    caption.paragraph_format.space_before = Pt(8)
    caption.paragraph_format.space_after = Pt(3)
    caption.paragraph_format.keep_with_next = True
    _add_runs(caption, block.caption, size=9.5, italic=True)

    columns = len(block.header)
    table = document.add_table(rows=1 + len(block.rows), cols=columns)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    total = sum(block.widths)
    align = block.align or "l" * columns
    for index, width in enumerate(block.widths):
        for row in table.rows:
            row.cells[index].width = Inches(text_width * width / total)
    for index, text in enumerate(block.header):
        cell = table.rows[0].cells[index]
        _cell_text(cell, text, bold=True, align="l" if index == 0 else "c", size=9)
        _shade(cell._tc.get_or_add_tcPr(), HEAD_FILL)
    for r, row in enumerate(block.rows):
        cells = table.rows[r + 1].cells
        if r in block.panels:
            merged = cells[0].merge(cells[-1])
            _cell_text(merged, row[0], bold=False, italic=True, align="l", size=9)
            _shade(merged._tc.get_or_add_tcPr(), QUOTE_FILL)
            continue
        for index, text in enumerate(row):
            _cell_text(cells[index], text, bold=False, align=align[index], size=9)
    # A row is never split across two pages, and the header row repeats on a new page.
    for r, row in enumerate(table.rows):
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
        if r == 0:
            properties.append(OxmlElement("w:tblHeader"))
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def build_docx() -> None:
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(1))
    text_width = 6.5

    normal = document.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _page_number_field(footer)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(0)
    run = title.add_run(TITLE)
    run.font.size, run.bold, run.font.color.rgb = Pt(20), True, ACCENT
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run(SUBTITLE)
    run.font.size, run.italic = Pt(13), True
    for label, value in (("Manuscript", MANUSCRIPT), ("Authors", AUTHORS)):
        line = document.add_paragraph()
        line.paragraph_format.space_after = Pt(2)
        head = line.add_run(f"{label}: ")
        head.bold = True
        line.add_run(value)
    rule = document.add_paragraph()
    rule.paragraph_format.space_after = Pt(8)
    properties = rule._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("val", "single"), ("sz", "8"), ("space", "1"), ("color", ACCENT_HEX)):
        bottom.set(qn(f"w:{key}"), value)
    borders.append(bottom)
    properties.append(borders)

    for block in CONTENT:
        match block:
            case Heading(text=text, level=level):
                # A real heading style, so that the comments appear in the navigation
                # pane and in the PDF outline; only its appearance is overridden.
                paragraph = document.add_heading(level=level)
                paragraph.paragraph_format.space_before = Pt(16 if level == 1 else 10)
                paragraph.paragraph_format.space_after = Pt(6)
                paragraph.paragraph_format.keep_with_next = True
                run = paragraph.add_run(text)
                run.bold = True
                run.italic = False
                _set_font(run, BODY_FONT)
                run.font.size = Pt(13.5 if level == 1 else 12)
                run.font.color.rgb = ACCENT
            case Label(text=text):
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_before = Pt(6)
                paragraph.paragraph_format.space_after = Pt(3)
                paragraph.paragraph_format.keep_with_next = True
                run = paragraph.add_run(text.upper())
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            case Para(text=text):
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                _add_runs(paragraph, text)
            case Quote(paragraphs=paragraphs):
                for index, text in enumerate(paragraphs):
                    paragraph = document.add_paragraph()
                    paragraph.paragraph_format.left_indent = Inches(0.25)
                    paragraph.paragraph_format.right_indent = Inches(0.15)
                    last = index == len(paragraphs) - 1
                    paragraph.paragraph_format.space_after = Pt(8 if last else 0)
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    _left_rule(paragraph)
                    _add_runs(paragraph, text, size=10.5, italic=True)
                    if not last:
                        gap = document.add_paragraph()
                        gap.paragraph_format.left_indent = Inches(0.25)
                        gap.paragraph_format.right_indent = Inches(0.15)
                        gap.paragraph_format.space_after = Pt(0)
                        gap.paragraph_format.line_spacing = Pt(6)
                        _left_rule(gap)
            case Bullets(items=items):
                for text in items:
                    paragraph = document.add_paragraph(style="List Bullet")
                    paragraph.paragraph_format.space_after = Pt(4)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    _add_runs(paragraph, text)
            case Table():
                _docx_table(document, block, text_width)

    document.core_properties.title = f"{TITLE}: {SUBTITLE}"
    document.core_properties.author = AUTHORS
    document.core_properties.subject = MANUSCRIPT
    document.save(str(DOCX))


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\u2212": r"$-$",
    "\u00d7": r"$\times$",
    "\u201c": "``",
    "\u201d": "''",
    "\u2019": "'",
    "\u00fc": r"\"u",
    "\u00a0": "~",
}


def tex_escape(text: str) -> str:
    out = "".join(_ESCAPES.get(character, character) for character in text)
    if not out.isascii():
        bad = sorted({c for c in out if ord(c) > 127})
        raise AnswerError(f"no LaTeX mapping for {bad!r}")
    # A plus or minus directly before a digit is a sign, and reads better in math mode.
    return re.sub(r"(?<![\w.$])\+(?=\d)", r"$+$", out)


def tex_inline(text: str) -> str:
    parts: list[str] = []
    for style, content in spans(text):
        if style == "code":
            escaped = tex_escape(content)
            for mark in ("/", r"\_", "."):
                escaped = escaped.replace(mark, mark + r"\allowbreak{}")
            parts.append(r"{\small\texttt{" + escaped + "}}")
            continue
        escaped = tex_escape(content)
        if style == "b":
            parts.append(r"\textbf{" + escaped + "}")
        elif style == "i":
            parts.append(r"\emph{" + escaped + "}")
        elif style == "sub":
            parts.append(r"\textsubscript{" + escaped + "}")
        elif style == "sup":
            parts.append(r"\textsuperscript{" + escaped + "}")
        else:
            parts.append(escaped)
    joined = "".join(parts)
    return (
        re.sub(r"(https://[^\s,;]+[^\s,;.])", r"\\url{\1}", joined) if "https" in joined else joined
    )


def _tex_table(block: Table) -> str:
    columns = len(block.header)
    total = sum(block.widths)
    align = block.align or "l" * columns
    # Leave room for the column padding, which tabular adds outside each p column.
    usable = 1.0 - 0.0195 * columns
    spec = []
    for index, width in enumerate(block.widths):
        share = usable * width / total
        how = {"l": r"\raggedright", "c": r"\centering", "r": r"\raggedleft"}[align[index]]
        spec.append(rf">{{{how}\arraybackslash}}p{{{share:.4f}\textwidth}}")
    lines = [
        r"\begin{center}",
        r"\begin{minipage}{\textwidth}",
        r"\small\emph{" + tex_inline(block.caption) + r"}\par\vspace{3pt}",
        r"\footnotesize\renewcommand{\arraystretch}{1.18}\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{" + "".join(spec) + "}",
        r"\toprule",
        " & ".join(
            (r"\raggedright\arraybackslash " if index == 0 else r"\centering\arraybackslash ")
            + r"\textbf{"
            + tex_inline(cell)
            + "}"
            for index, cell in enumerate(block.header)
        )
        + r" \\",
        r"\midrule",
    ]
    for r, row in enumerate(block.rows):
        if r in block.panels:
            if r:
                lines.append(r"\addlinespace[2pt]")
            lines.append(
                rf"\multicolumn{{{columns}}}{{@{{\hspace{{3.5pt}}}}l}}{{\emph{{"
                + tex_inline(row[0])
                + r"}} \\"
            )
            continue
        lines.append(" & ".join(tex_inline(cell) for cell in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}", r"\end{center}"]
    return "\n".join(lines)


def build_tex() -> str:
    head = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[scaled=0.92]{helvet}
\usepackage[margin=1in]{geometry}
\usepackage{xcolor,array,booktabs,fancyhdr,microtype,url,ragged2e}
\usepackage[hidelinks]{hyperref}
\definecolor{accent}{HTML}{8C0C0C}
\definecolor{quotefill}{HTML}{F4F1EC}
\definecolor{labelgrey}{HTML}{555555}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{2em}
\urlstyle{same}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\small\thepage}
% Start a new page unless at least #1 of the current one is left, so that a heading or a
% label is never stranded at the foot of a page (the needspace package is not assumed).
\newcommand{\need}[1]{\par\begingroup\dimen0=\pagegoal\advance\dimen0 by -\pagetotal
  \ifdim\pagetotal>0pt\ifdim\dimen0<#1\relax\newpage\fi\fi\endgroup}
\newcommand{\respsection}[1]{\need{9\baselineskip}\vspace{12pt}%
  {\large\bfseries\color{accent}\raggedright #1\par}\nopagebreak\vspace{2pt}}
\newcommand{\respsubsection}[1]{\need{6\baselineskip}\vspace{6pt}%
  {\normalsize\bfseries\color{accent}\raggedright #1\par}\nopagebreak}
\newcommand{\resplabel}[1]{\need{5\baselineskip}\vspace{4pt}%
  {\footnotesize\bfseries\color{labelgrey}\MakeUppercase{#1}\par}\nopagebreak\vspace{-2pt}}
% The reviewer's words: tinted, with an accent rule on the left, and free to break
% across pages paragraph by paragraph.
\newcommand{\reviewerpar}[1]{%
  \par\noindent{\color{accent}\vrule width 2.2pt}%
  \colorbox{quotefill}{\parbox{\dimexpr\textwidth-2.2pt-2\fboxsep\relax}{%
    \small\itshape\strut #1\strut}}\par}
\newenvironment{respitems}{\begin{list}{\textbullet}{\setlength{\leftmargin}{1.4em}%
  \setlength{\labelwidth}{0.8em}\setlength{\labelsep}{0.6em}\setlength{\itemsep}{3pt}%
  \setlength{\parsep}{0pt}\setlength{\topsep}{2pt}}}{\end{list}}
\begin{document}
"""
    body: list[str] = [
        r"{\LARGE\bfseries\color{accent} " + tex_escape(TITLE) + r"\par}",
        r"\vspace{2pt}{\large\itshape " + tex_escape(SUBTITLE) + r"\par}\vspace{8pt}",
        r"\textbf{Manuscript:} " + tex_escape(MANUSCRIPT) + r"\par",
        r"\vspace{-3pt}\textbf{Authors:} " + tex_escape(AUTHORS) + r"\par",
        r"\vspace{2pt}{\color{accent}\hrule height 0.8pt}\vspace{6pt}",
    ]
    for block in CONTENT:
        match block:
            case Heading(text=text, level=level):
                command = "respsection" if level == 1 else "respsubsection"
                body.append(rf"\{command}{{{tex_inline(text)}}}")
            case Label(text=text):
                body.append(rf"\resplabel{{{tex_inline(text)}}}")
            case Para(text=text):
                body.append(tex_inline(text) + r"\par")
            case Quote(paragraphs=paragraphs):
                body.append(r"\vspace{5pt}{\setlength{\parskip}{0pt}\offinterlineskip")
                for text in paragraphs:
                    body.append(rf"\reviewerpar{{{tex_inline(text)}}}")
                body.append(r"}\vspace{4pt}")
            case Bullets(items=items):
                body.append(r"\begin{respitems}")
                body.extend(r"\item " + tex_inline(text) for text in items)
                body.append(r"\end{respitems}")
            case Table():
                body.append(_tex_table(block))
    return head + "\n\n".join(body) + "\n\\end{document}\n"


def build_pdf() -> None:
    if not PDFLATEX.is_file():
        raise AnswerError(f"pdflatex was not found at {PDFLATEX}")
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "answer.tex").write_text(build_tex(), encoding="utf-8")
        for _ in range(2):
            done = subprocess.run(
                [str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "answer.tex"],
                cwd=work,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                log = (work / "answer.log").read_text(encoding="utf-8", errors="replace")
                lines = log.splitlines()
                first = next((i for i, line in enumerate(lines) if line.startswith("!")), None)
                detail = "\n".join(lines[first : first + 12]) if first is not None else log[-1500:]
                raise AnswerError(f"pdflatex failed:\n{detail}")
        log = (work / "answer.log").read_text(encoding="utf-8", errors="replace")
        overfull = [line for line in log.splitlines() if line.startswith("Overfull \\hbox")]
        if overfull:
            raise AnswerError(f"the PDF has {len(overfull)} overfull lines: {overfull[:5]}")
        shutil.copyfile(work / "answer.pdf", PDF)


def main() -> int:
    check_content()
    build_docx()
    print(f"wrote {DOCX}")
    if "--no-pdf" not in sys.argv[1:]:
        build_pdf()
        print(f"wrote {PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
