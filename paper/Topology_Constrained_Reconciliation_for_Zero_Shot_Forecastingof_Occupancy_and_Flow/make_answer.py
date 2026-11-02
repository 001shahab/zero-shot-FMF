"""Build answer.docx, the point-by-point response to the reviewers.

Reviewer 2's comments are quoted verbatim from the embedded annotations in
``paper/2003109reviewsupportingfile1354161.PDF``; the extraction is done here rather
than transcribed by hand so that the quotations cannot drift from the source. Reviewer
1's points arrived as prose in the decision letter rather than as PDF annotations, so
they are restated in our own words and labelled as such.
"""

from __future__ import annotations

import pathlib

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from pypdf import PdfReader

HERE = pathlib.Path(__file__).resolve().parent
REVIEW_PDF = HERE.parent / "2003109reviewsupportingfile1354161.PDF"
OUTPUT = HERE / "answer.docx"

ACCENT = RGBColor(0x8C, 0x0C, 0x0C)


class AnswerError(RuntimeError):
    """Raised when the response document cannot be built from its sources."""


def reviewer_two_comments() -> list[tuple[int, str]]:
    """Every annotation with a comment body, in page order, as (page, text)."""
    reader = PdfReader(REVIEW_PDF)
    found: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        for annotation in page.get("/Annots", []) or []:
            contents = annotation.get_object().get("/Contents")
            if contents:
                found.append((page_number, " ".join(str(contents).split())))
    if len(found) != 30:
        raise AnswerError(f"expected 30 embedded comments, found {len(found)}")
    return found


# --------------------------------------------------------------------------- #
# Reviewer 1
# --------------------------------------------------------------------------- #

REVIEWER_ONE: list[tuple[str, str, str]] = [
    (
        "R1.1 Framing: no real museum data, and the foundation model loses to simpler "
        "baselines on the authors' own simulator",
        "The paper is framed around heritage museums but contains no museum data, and on "
        "the simulator the paper itself releases, the foundation model is beaten by "
        "seasonal naive and by LightGBM. Either validate the simulator against real "
        "museum traces, or drop the heritage museum claim from the title and "
        "contributions.",
        "Accepted in the second form, since the first is not available to us. We have no "
        "real museum data and will not pretend otherwise.\n\n"
        "The title has changed. It was \u201cTopology-Constrained Reconciliation for "
        "Zero-Shot Forecasting of Heritage Museum Occupancy and Flow\u201d and is now "
        "\u201cTopology-Constrained Reconciliation for Zero-Shot Forecasting of Building "
        "Occupancy and Doorway Flow\u201d. The museum setting now appears only where it "
        "is accurate: as the motivating application in the introduction, and as the "
        "subject of the simulator we release. The contribution list calls the simulated "
        "site a \u201cmuseum-like building\u201d rather than a validated museum model, and "
        "a new contribution item states the evidential boundary explicitly.\n\n"
        "On the simulator result, we agree it is real and we would rather explain it than "
        "bury it, so we have kept it and said what we think it means. The simulator was "
        "built to have known ground truth on every doorway, which is what a conservation "
        "constraint needs, and it was not built to reproduce the difficulty structure of "
        "real visitor behaviour. Its generative process is close enough to a fixed daily "
        "profile that a seasonal baseline is nearly optimal on it, which is exactly the "
        "regime in which Section 7.2 predicts zero-shot models should not win. We now "
        "state in the limitations that the simulator's data-efficiency result contradicts "
        "both real datasets, and we read that as a limitation of the generator's realism "
        "rather than as evidence about the models. Correspondingly, no transferability "
        "claim in the paper rests on the simulator; Remark 3 forbids it, and the "
        "forecasting comparisons use only the two real datasets.",
    ),
    (
        "R1.2 The reconciliation gains are small and are tested only on simulated data",
        "The accuracy improvement from reconciliation is a fraction of a percent, and the "
        "whole reconciliation evaluation is on simulation. The framework is therefore not "
        "shown to work end to end in a real building.",
        "Accepted, and this is now the paper's stated principal limitation and its "
        "principal item of future work. We have separated what is proved from what is "
        "measured, and we have stopped presenting the two at equal strength.\n\n"
        "What we have not done is manufacture real data. We searched for a public dataset "
        "giving occupancy and doorway flow for the same building on a common clock and did "
        "not find one. Neither of our real datasets can substitute: ROBOD reports room "
        "occupancy without doorway counts, HZMetro reports station flow without station "
        "occupancy, so in each case one side of the continuity equation is missing and the "
        "constraint cannot even be formed. The harness enforces this rather than leaving it "
        "to a reader to notice, by making any reconciler other than \u201cnone\u201d an "
        "error on such a site. The ATC indoor tracking dataset we cite comes closest, but "
        "an occupancy figure derived from the same tracks that give the flows would make "
        "the constraint hold by construction and test nothing, so it is unusable for this "
        "purpose without an independent occupancy measurement.\n\n"
        "What we have done instead is strengthen the simulated evidence and label it "
        "honestly. We added a second operating point, a crowded scenario in which room "
        "capacity actually binds, so that the inequality constraints are exercised rather "
        "than being almost always slack: raw bound violation rises from 0.006 to 0.451 "
        "persons and MinT's violation from 2.06 to 2.94. We added moving-block bootstrap "
        "confidence intervals for every paired change. We added risk-sensitive metrics, "
        "which produced the most useful result of the revision: the error in the predicted "
        "peak occupancy falls by 6.3 percent against 0.56 percent for mean absolute error, "
        "so the operationally relevant effect is eleven times the mean effect. And we have "
        "rewritten the recommendation so that the constrained projection is recommended on "
        "the strength of the feasibility guarantee, which is mathematics, while the "
        "uncertainty weighting is offered as the better-performing variant in our "
        "simulations rather than as an established improvement. See also R2.27 and R2.29.",
    ),
    (
        "R1.3 The theoretical novelty is overstated; the theorems are standard projection "
        "results already present in the reconciliation literature",
        "Uniqueness, coherence and non-expansiveness of a projection onto a closed convex "
        "set are textbook facts, and constrained reconciliation has been studied before. "
        "The theoretical contribution should not be presented as new.",
        "Accepted. We have removed the novelty framing from the theory and replaced it "
        "with attribution.\n\n"
        "We now say plainly that uniqueness and non-expansiveness are the standard "
        "consequences of projecting onto a non-empty closed convex set under a positive "
        "definite metric, that we claim no novelty for them, and that we state and prove "
        "them because the specific admissible set and the specific metric are ours and the "
        "reader is entitled to see that they satisfy the hypotheses. We have added "
        "citations to the prior work the reviewer is pointing at, including Wickramasuriya "
        "et al. on MinT, Wickramasuriya et al. on non-negative reconciliation, Girolimetto "
        "and Di Fonzo on reconciliation under general linear equality constraints, and "
        "Panagiotelis et al. on probabilistic coherence. Proposition 7, which derives MinT "
        "as the special case that drops the inequality constraints, is positioned as "
        "locating our method inside that literature rather than as an independent result.\n\n"
        "What we do claim as new is narrower and we have stated it that way: the "
        "particular admissible set, which couples a discrete continuity equation on a "
        "building graph with occupancy capacity and doorway throughput; the use of a "
        "foundation model's own predictive quantile spread as the reconciliation metric, "
        "for which we are not aware of prior direct evidence; and the empirical finding "
        "that this spread is informative enough about the model's own errors to turn the "
        "accuracy cost of coherence into a gain, which the uniform-weighting arm shows is "
        "not automatic.",
    ),
    (
        "R1.4 The quantile shift does not preserve coherence except at the median",
        "Shifting every quantile by the median correction cannot preserve conservation or "
        "the bounds at non-median levels, so the claim that the whole predictive "
        "distribution is coherent is not justified.",
        "Accepted; the claim was wrong and is withdrawn. Because Reviewer 2 raised the same "
        "point in more detail at page 13, the full response including the new measurements "
        "is given at R2.16, and Section 7.8 and Table 11 are new.",
    ),
    (
        "R1.5 The evaluation design is limited: too few context lengths, and the local "
        "baselines are short relative to the metro's weekly seasonality",
        "Only two context lengths are compared, and the baselines see at most two days of "
        "history although metro ridership has strong weekly structure, so the baselines may "
        "be handicapped by the design rather than by their own limitations.",
        "Accepted on both counts, and both produced new experiments that changed "
        "conclusions.\n\n"
        "On context length, we extended the sweep from two points to five, at one, two, "
        "three, seven and fourteen daily seasons, on both real datasets, comparing on the "
        "subset of origins that every context length can score so that the comparison is "
        "not confounded by origin attrition. Two seasons is not where the curve flattens. "
        "On HZMetro at 120 minutes the best foundation model goes 1.037, 0.677, 0.557, "
        "0.424, 0.414 MASE across the five context lengths, so accuracy continues to "
        "improve to about seven seasons and two seasons is a reporting convention rather "
        "than a saturation point. Section 7.3 and Table 5 are rewritten accordingly, and "
        "the abstract and conclusion now say so. See also R2.24.\n\n"
        "On the weekly baseline, we implemented a weekly seasonal naive forecaster and ran "
        "it on both sites with a context long enough to support it. The reviewer's "
        "intuition is correct in direction and does not change the conclusion. On HZMetro "
        "the weekly baseline is much stronger than the daily one, 0.598 against 0.774 MASE, "
        "and it is still 30.8 percent behind the best foundation model. On ROBOD it is "
        "worse than the daily baseline, 0.887, which we read as the weekly profile of an "
        "office building being less stable than its daily profile. Section 7.1 reports both.",
    ),
    (
        "R1.6 Housekeeping: funding disclosure, the inconsistent series counts, the "
        "pre-registration claim, and the data availability wording",
        "The funding statement and the acknowledgement contradict each other. Table 1 and "
        "Figure 2 give different series counts for HZMetro. The hypotheses are described as "
        "pre-registered. Data availability is declared not applicable although a code "
        "repository is provided.",
        "All four accepted and corrected.\n\n"
        "Funding. There was no funding for this work. The manuscript now carries an "
        "explicit Funding section stating that the research received no specific grant from "
        "any agency in the public, commercial or not-for-profit sectors, a new "
        "Acknowledgment section that thanks the dataset and model authors and the reviewers "
        "and makes no financial claim whatsoever, and a separate Declaration of competing "
        "interest recording that one author is affiliated with 3S Holding O\u00dc and that "
        "the company provided no financial support and had no role in the study. The "
        "contradiction the reviewers saw is removed because the financial-support sentence "
        "is gone. See also R2.1.\n\n"
        "Series counts. Both numbers were right and the paper never said so. The HZMetro "
        "panel is 240 series, being 80 station occupancies plus 160 directional flows, and "
        "the within-dataset regression of Section 7.3 uses the 160 flow series only, "
        "because HZMetro provides no independent occupancy ground truth against which the "
        "80 occupancy series could be scored. Table 1 now has separate Panel and Scored "
        "columns for every dataset, and a new paragraph states the distinction and notes "
        "that the submitted table had reported the scored count for ROBOD and the panel "
        "count for HZMetro, which is what made them look inconsistent. See also R2.18.\n\n"
        "Pre-registration. The word was wrong and is removed throughout. Our hypotheses "
        "were written into the repository before the corresponding experiments ran, and the "
        "commit history evidences that, but they were never deposited with a registry. "
        "\u201cStated a priori\u201d replaces \u201cpre-registered\u201d everywhere, and the "
        "reproducibility statement explains the change rather than making it silently.\n\n"
        "Data availability. The manuscript now carries a Data and code availability section "
        "giving the repository URL, stating that the simulated benchmark is released as its "
        "generator plus seed and configuration so that any scenario reproduces bit for bit, "
        "stating that the two real datasets are third-party public releases that we do not "
        "redistribute but for which we provide checksummed fetch scripts, and stating that "
        "run manifests, per-origin predictions and metrics are included so that every table "
        "and figure can be regenerated without re-running the forecasters.",
    ),
]


# --------------------------------------------------------------------------- #
# Reviewer 2, keyed by the order of the extracted annotations
# --------------------------------------------------------------------------- #

REVIEWER_TWO_REPLIES: list[tuple[str, str]] = [
    (
        "Funding and acknowledgement contradict each other",
        "Accepted and corrected, and the resolution is that there was no funding. The "
        "financial-support sentence has been deleted. The manuscript now carries three "
        "separate statements that cannot conflict: a Funding section declaring that the "
        "research received no specific grant from any agency in the public, commercial or "
        "not-for-profit sectors and that no organisation provided financial support; an "
        "Acknowledgment section that thanks the authors of the ROBOD and PVCGN releases, "
        "the developers of TimesFM, Chronos and Toto, and the reviewers, and makes no "
        "financial claim; and a Declaration of competing interest recording the 3S Holding "
        "O\u00dc affiliation together with the fact that the company provided no financial "
        "support and had no role in the study design, in the collection, analysis or "
        "interpretation of data, in the writing, or in the decision to submit.",
    ),
    (
        "The three applications act on different temporal scales; say which decisions the "
        "horizons support",
        "Accepted, and a new paragraph in the introduction answers it directly, because the "
        "reviewer is right that the answer constrains the design.\n\n"
        "Crowd safety is the fastest of the three. A steward redirected to a bottleneck, or "
        "a timed-entry slot held back, needs 15 to 30 minutes of warning, which is why that "
        "is the range Hypothesis 1 is stated over, and the indicator is peak occupancy "
        "relative to room capacity rather than mean error, which is why the revision reports "
        "peak error as a separate metric. Conservation acts on hours: the quantity that "
        "damages a hygroscopic object is cumulative moisture and temperature load over a "
        "visiting day, so the relevant horizon is one to two hours and the indicator is an "
        "integral rather than a peak. Security sits in between and is a detection problem "
        "rather than a control problem, where the forecast supplies an expected pattern and "
        "the indicator is the size of the deviation from it. The horizons we evaluate, 5 to "
        "120 minutes, span the first two.\n\n"
        "The paragraph also states what we do not do: we evaluate none of the three decision "
        "rules themselves, because the decision-utility experiment remains unrun. It says "
        "what the forecast is for, not what it achieves operationally.",
    ),
    (
        "The cold-start claim is too general; present it as a practical challenge",
        "Accepted, and the sentence has been rewritten. The reviewer's objection is "
        "well-founded and, as it turns out, our own data agrees with the reviewer rather "
        "than with us.\n\n"
        "We now say that on day one of instrumentation there is no history at all, so a "
        "method requiring site-specific training cannot produce a forecast whereas a "
        "zero-shot method can. That is a statement about availability, not about how long "
        "adequacy takes to arrive, and it is presented as the practical challenge motivating "
        "the work rather than as a universal limitation. The claim that a newly instrumented "
        "site yields no usable data for weeks is gone, and we say explicitly that how long a "
        "building takes to yield usable data depends on its traffic volume, its sensors and "
        "the model being fitted, none of which we have measured for a real museum.\n\n"
        "We also tested the stronger version of the claim and it failed. The new "
        "training-budget sweep on real ROBOD data (Section 7.8) shows that going from three "
        "days of local history to sixty, a twentyfold increase, buys the trained local model "
        "at most 4.3 percent, whereas re-specifying that model bought 14.8 percent. Data "
        "scarcity is therefore not the dominant factor even on our own data, and the paper "
        "now says so.",
    ),
    (
        "Do all the foundation models support future covariates, and were they actually "
        "used?",
        "A fair challenge, and the submitted introduction did imply more than the "
        "experiments deliver. The introduction now states the position explicitly.\n\n"
        "On capability, the position differs by model and the paper now says which is which. "
        "TimesFM 3.0 and Chronos-2 accept known-future covariates through documented "
        "interfaces, which we verified against the live repositories rather than against "
        "remembered signatures, and our wrappers for both expose that pathway as the "
        "configurations named in the registry. Our Toto-2.0 wrapper exposes no known-future "
        "covariate pathway and declares so, since the inference signature we verified on its "
        "model card takes only the variate tensor; we did not establish whether the model "
        "could accept covariates by some route we did not find, so we state what our wrapper "
        "does rather than what the model cannot do. TimesFM 2.5 is used univariately here.\n\n"
        "On what was evaluated, the honest answer is nothing. Our data contract carries "
        "known-future covariates, the forecaster interface accepts them, and the simulator "
        "emits them, so the pathway is implemented and unit-tested, but the covariate "
        "ablation (E2) was never executed. Every accuracy number in the paper comes from "
        "runs with no covariates supplied. The introduction now says that the covariate "
        "channel is part of the formulation rather than part of the evidence, the "
        "limitations section lists E2 among the unrun experiments, and the repository's "
        "experiment registry has been corrected to agree (see R2.20).",
    ),
    (
        "The components are evaluated on different datasets and reconciliation is never "
        "tested on real occupancy and flow from the same building; make this explicit in the "
        "contribution statement",
        "Accepted without reservation. This is the single most important correction in the "
        "revision and we have made it unmissable rather than merely present.\n\n"
        "The introduction now carries a dedicated paragraph headed \u201cWhat is tested on "
        "real data and what is not\u201d, which states that the forecasting claims are "
        "tested on two real datasets over 261 and 181 distinct origins, that the "
        "reconciliation claims are proved mathematically and measured only on the "
        "simulator, and that no experiment in the paper reconciles real occupancy against "
        "real doorway flow from one building. The contribution list has a new final item "
        "saying the same thing, and the reconciliation contribution item carries the "
        "qualifier in place. The limitations section explains why no public dataset can "
        "currently supply this and states what acquiring it would require, namely "
        "synchronised counting at every doorway including those to outside plus a per-room "
        "occupancy estimate that is independent of the doorway counts, since a "
        "doorway-derived occupancy would make the constraint hold by construction. The "
        "conclusion repeats the boundary. The title changed for the same reason (R1.1).",
    ),
    (
        "State explicitly that forecasting improvement is tested on real data whereas "
        "reconciliation is evaluated on simulation",
        "Accepted; this is the same correction as R2.5 and is addressed by the same new "
        "material. The related-work gap statement now separates the two claims where it "
        "states the gap, so that a reader meets the distinction at the point where the "
        "contribution is motivated rather than only at the point where the results are "
        "reported.",
    ),
    (
        "Revisit the statement that conservation is exact only when no visitor crosses more "
        "than one doorway within an interval",
        "The reviewer is right and we have rewritten the assumption. Our statement was both "
        "too strong and aimed at the wrong thing.\n\n"
        "If every crossing is counted, the node-level balance holds regardless of how many "
        "doorways one person traverses within the interval, because each traversal "
        "contributes to both the outflow of one room and the inflow of the next and the sum "
        "telescopes. What matters is what the reviewer identifies. The revised text replaces "
        "our single condition with three: complete observation, meaning every crossing of "
        "every edge in the graph is counted and no movement bypasses a counted doorway; a "
        "consistent clock, meaning occupancy snapshots and flow counts refer to the same "
        "interval boundaries so that a person in transit at a boundary is attributed "
        "consistently on both sides; and a closed boundary, meaning the virtual outside node "
        "absorbs all exchange with the exterior so that the graph has no unmodelled source "
        "or sink. We also note what the multiple-crossing case does affect, which is not "
        "exactness but interpretation: a visitor who enters and leaves a room within one "
        "interval raises both the inflow and outflow of that room while leaving its "
        "occupancy snapshot unchanged, so the constraint is satisfied while the within-"
        "interval dynamics are invisible at this resolution. That is a statement about "
        "temporal aliasing, not about whether the identity holds.",
    ),
    (
        "Why are flow-capacity limits excluded from the admissible set?",
        "They are not excluded, and the paper was wrong to suggest otherwise. This was a "
        "documentation error rather than a missing feature, and we are grateful it was "
        "caught because the formulation as written understated what the implementation does.\n\n"
        "The implementation has always imposed a per-edge throughput bound, computed as the "
        "doorway's capacity in persons per minute times the interval length in minutes, with "
        "the per-minute rate taken from the edge's clear width using the pedestrian flow "
        "literature we cite. Equation (4) in the revision now states this bound explicitly, "
        "as a term in the admissible set alongside the occupancy bounds, with the mapping "
        "from clear width to capacity given. Section 7.11, which is new, reports the simulator's "
        "doorway widths of 0.9 to 2.2 metres and the resulting 14 to 50 persons per minute, "
        "so the values are inspectable rather than asserted. This also answers the reviewer's "
        "bottleneck-management point: the constraint that makes the formulation relevant to "
        "bottlenecks was present in the solver and absent only from the paper.",
    ),
    (
        "The admissible set is described as bounded, but flow coordinates have no finite "
        "upper limit and a directed cycle admits arbitrarily large circulating flow",
        "The reviewer's analysis is exactly correct, and with the correction of R2.8 in place "
        "the conclusion changes in our favour, which is why we are glad the point was pressed "
        "rather than conceded loosely.\n\n"
        "As Equation (4) stood in the submitted paper, the set was not bounded. A directed "
        "cycle can carry an arbitrarily large circulating flow that satisfies conservation at "
        "every node and leaves every occupancy unchanged, so the feasible set contained an "
        "entire unbounded ray and the word \u201cbounded\u201d was false. Once the per-edge "
        "throughput bound that the implementation always applied is written into the "
        "formulation, every coordinate is confined to a closed interval, the set is contained "
        "in a product of compact intervals, and it is genuinely compact. Proposition 2 now "
        "says \u201ccompact\u201d with that argument given, and a remark records both the "
        "reviewer's counterexample and the fact that uniqueness and non-expansiveness never "
        "needed boundedness in the first place, since projection onto a non-empty closed "
        "convex set under a positive definite metric is unique and non-expansive whether or "
        "not the set is bounded. The correction therefore strengthens a statement without any "
        "downstream result depending on it.",
    ),
    (
        "Is the Gaussian approximation in Equation (6) necessary, and how sensitive is the "
        "result to the interval width?",
        "The reviewer's instinct is right, the approximation is unnecessary, and chasing this "
        "point uncovered a genuine defect in our solver configuration which is now fixed.\n\n"
        "The reviewer notes that the optimisation needs only positive relative weights. That "
        "is correct: scaling the weight matrix by any positive constant leaves the argmin "
        "untouched, so the constant 2.5631 that converts an 80 percent interval width into a "
        "standard deviation divides out and the Gaussian reading of the interval carries no "
        "mathematical content. When we tried to verify this empirically, it failed. Scaling "
        "every predictive spread by a thousand moved the projection by 1.6e-2, because the "
        "solver's absolute convergence tolerance is compared against an unnormalised "
        "residual, so the scale of the weights did reach the solution through the numerics "
        "even though it cannot reach it through the mathematics. We fixed this at the source "
        "by normalising the weight matrix to unit mean diagonal before the solve, which is a "
        "mathematical no-op that makes the invariance exact rather than approximate, and we "
        "added two unit tests, one checking invariance to the scale of the spread and one "
        "checking that a degenerate spread is refused loudly rather than silently normalised. "
        "With the normalisation in place, the Gaussian reading is provably irrelevant, and we "
        "say so in the text instead of claiming a distributional assumption we do not need.\n\n"
        "On sensitivity to interval width, Section 7.9 and Table 12 are new. We re-ran the "
        "projection with the weights derived four ways: the Gaussian mapping of the 80 "
        "percent interval, the raw 80 percent width with no mapping at all, and Gaussian "
        "mappings of the 60 and 40 percent intervals. The raw width is now bit-identical to "
        "the Gaussian mapping of the same interval, with a maximum coordinate shift of 1e-13, "
        "which is the invariance confirmed numerically. Narrowing the interval does change the "
        "answer slightly, from 0.31760 to 0.31809 at 60 percent and 0.31898 at 40 percent in "
        "MASE, because a narrower interval is more often clipped by the lower bound on the "
        "spread: the share of coordinates sitting at that floor rises from 63.6 to 69.6 "
        "percent. The conclusion we draw and state is that the choice of interval matters only "
        "through how much of the spread it leaves resolvable, and the Gaussian label on it "
        "matters not at all.",
    ),
    (
        "The positive-definiteness argument has the inequality backwards, and the "
        "eigenvalue bound and strong-convexity modulus need correcting",
        "The reviewer is right, the error is ours, and it is corrected. From a lower bound "
        "on the spread it follows that the inverse square is bounded above, so the weight "
        "matrix satisfies an upper eigenvalue bound and not the lower one we wrote. The "
        "corrected statement, now an explicit displayed bound in Section 4, brackets the "
        "weight matrix between an upper bound set by the floor on the spread and a lower "
        "bound set by the largest spread actually observed, which is finite because the "
        "predictive quantiles are finite. Positive definiteness follows from every spread "
        "being finite and strictly positive, which the floor guarantees, so uniqueness is "
        "unaffected. The strong-convexity modulus is restated as the smallest eigenvalue, "
        "which is the correct quantity and which the corrected bracket supplies, and the "
        "non-expansiveness result is re-derived from it. No result changes direction; the "
        "derivation that reached them was invalid and now is not.",
    ),
    (
        "Corollary 5 claims an equality where Theorem 4 establishes only an inequality",
        "Accepted. Corollary 5 is restated as an inequality, with the direction the theorem "
        "actually supports, namely that the expected reduction in weighted squared error is "
        "at least the expected squared correction distance. The interpretation in the "
        "surrounding text is rewritten to match, so that the corollary is now read as a "
        "guaranteed floor on the improvement rather than as an exact accounting of it. We "
        "also state the condition under which equality does hold, which is the affine case "
        "where no inequality constraint is active and the correction is orthogonal to the "
        "feasible directions, and we note that this is precisely the MinT setting of "
        "Proposition 7, which makes the relationship between the two results legible rather "
        "than coincidental.",
    ),
    (
        "A counterexample showing the claimed equality fails",
        "The counterexample is correct and we have put it in the paper, as a remark "
        "immediately after the corrected corollary, with the reviewer's arithmetic "
        "reproduced: on the non-negative half-line, with a raw forecast of -1, a projection "
        "to 0, an admissible truth of 1 and unit weight, the original squared error is 4, the "
        "projected squared error is 1, the improvement is 3, and the squared correction "
        "distance is 1. Three exceeds one, so the inequality holds strictly and the equality "
        "does not. We kept it rather than paraphrasing it because it shows in two lines what "
        "an active inequality constraint does to the geometry, and because a reader who "
        "remembers the submitted version is entitled to see the refutation rather than a "
        "quiet amendment.",
    ),
    (
        "Equation (8) is affine rather than linear when the constraint has a non-zero "
        "right-hand side",
        "Accepted. The map is affine, not linear, and Proposition 7 now says so. The "
        "right-hand side of the continuity equation is not zero in our setting, since it "
        "carries the previous occupancy level, so the general case is genuinely affine and "
        "calling it linear was imprecise. The revision distinguishes the two explicitly: the "
        "general affine conservation constraint with a non-zero right-hand side, which is our "
        "case, and the homogeneous zero-right-hand-side formulation standard in hierarchical "
        "reconciliation, for which Lemma 8 is stated. The lemma's hypotheses now name the "
        "homogeneous case rather than leaving a reader to infer it, and the proposition "
        "records that the affine case reduces to the homogeneous one by translating the "
        "origin, which is why the MinT correspondence survives the distinction.",
    ),
    (
        "The inequality constraints are the whole difference between MinT and the proposed "
        "method only if both use the same weighting matrix; specify how W was obtained and "
        "separate the two effects",
        "Accepted on both halves, and the new experiment changes the attribution we had made.\n\n"
        "On specification, Section 4 now states that MinT(Shrink) estimates the forecast-error "
        "covariance from in-sample one-step residuals using the Sch\u00e4fer and Strimmer "
        "shrinkage estimator towards a diagonal target, with the shrinkage intensity chosen by "
        "their analytic rule rather than tuned, and that it uses the equality constraints "
        "only. The citation is added. The proposed method's weight matrix is diagonal and "
        "derived per origin from the predictive quantile spread, so the two differ in two "
        "respects and the submitted paper's logic conflated them.\n\n"
        "On separation, we added a third reconciler arm that applies the proposed diagonal "
        "weight matrix with the equality constraints only, which holds the constraint set "
        "fixed at MinT's and varies only the weighting, and we report three ablations in the "
        "new Table 10. Holding the weighting fixed and removing only the inequality "
        "constraints costs 0.28 and 0.30 percent of mean absolute error for the two "
        "forecasters. Holding the constraint set at equalities only and replacing MinT's "
        "shrunk covariance with the proposed diagonal matrix gains 1.07 and 0.69 percent. "
        "The weighting matrix is therefore the larger of the two effects by a factor of "
        "roughly three, and the submitted paper's attribution of the entire difference to "
        "the inequality constraints was wrong in emphasis as well as in logic. The text now "
        "says this in those terms.",
    ),
    (
        "Shifting quantiles by the median correction does not preserve conservation or the "
        "bounds at other quantile levels",
        "Accepted; the claim was wrong and is withdrawn. We have replaced it with the "
        "restricted claim the reviewer asks for and, because an alternative existed in the "
        "harness, with a measurement of what the alternative costs.\n\n"
        "The theory is now restricted to the median. Section 4.4 gives the algebra: because "
        "the constraint map is affine with a non-zero right-hand side, translating a "
        "non-median quantile path by the median's correction leaves a residual equal to the "
        "constraint matrix applied to the difference between that quantile path and the "
        "median path, which is not zero in general, and the bounds can likewise be broken. "
        "Only the median path is coherent, and the text says only that.\n\n"
        "Section 7.8 and Table 11 are new and measure the damage. Under the shift strategy "
        "the residual at the extreme quantiles reaches 0.085 persons and the bound violation "
        "reaches 9.832 persons at the 5th percentile, the latter being large because a "
        "downward-shifted low quantile goes negative. Under the per-quantile strategy, which "
        "projects each quantile level separately and which the harness already implemented, "
        "the residual falls to at most 0.012 and the bound violation is exactly zero at every "
        "level. The quantile crossing rate is zero under both, so per-quantile projection "
        "does not reorder the quantiles in practice on this data. The cost is in the quantile "
        "loss, which we report, and in solve time, which scales with the number of levels. "
        "We now present shift as the default that guarantees a coherent median cheaply and "
        "per-quantile as the option that delivers a feasible predictive distribution, and we "
        "state the trade-off rather than claiming the first does the second. The conclusion "
        "carries the same distinction (see R2.30).",
    ),
    (
        "A single globally trained LightGBM is not enough; give the training window, lags, "
        "tuning and forecast strategy, and add a per-series or direct multi-horizon variant",
        "Accepted in full, and this is the reviewer comment that changed the paper's headline "
        "result. The reviewer's suspicion was correct: the competitor was too weak to carry "
        "the weight the hypothesis placed on it.\n\n"
        "On disclosure, Section 7.2 now states for every variant the training window, the lag "
        "set, whether daily and weekly seasonal lags are included, the tuning procedure and "
        "its search space, the forecast strategy, and whether the model is pooled across the "
        "panel or fitted per series. The submitted configuration, which we now describe as "
        "the weak specification it was, used short recent lags only, default "
        "hyper-parameters, one model pooled over all series, and direct multi-horizon "
        "prediction.\n\n"
        "On the variants, we built the ladder the reviewer asks for and report every rung in "
        "the new Table 4: the submitted global model, plus daily and weekly seasonal lags, "
        "plus hyper-parameter tuning, and finally fitted per series. The conclusion reverses "
        "on one of the two sites. On HZMetro at 120 minutes the per-series model reaches "
        "0.511 MASE against 0.613 for the best zero-shot configuration, an advantage of 28.9 "
        "percent with a Diebold-Mariano p-value of 1.44e-2 that survives Holm correction. On "
        "ROBOD at 60 minutes the zero-shot ordering holds, 0.374 against 0.512, an advantage "
        "of 26.8 percent that is not individually significant after correction. We checked "
        "for leakage before accepting the reversal: the local model's training window ends "
        "well before the test period begins, with a gap, and the first scored origin lies "
        "after the gap.\n\n"
        "Everything downstream has been changed to match. Hypothesis 1 is reported as holding "
        "on ROBOD and failing on HZMetro rather than as holding generally. The abstract, the "
        "introduction, the deployment recommendation and the conclusion all now state "
        "transferability as a competitive cold-start default whose direction depends on the "
        "site. Figure 1 overlays the strong competitor so that the reversal is visible rather "
        "than only described, and its caption says so. We have kept the submitted claim in the "
        "text with its retraction attached, so that a reader can see exactly how much of it "
        "was an artefact of the competitor.",
    ),
    (
        "Clarify the 240 versus 160 series counts",
        "Both numbers were right and the paper never explained them, which is our fault. "
        "HZMetro has 80 stations. The panel supplied to the forecasters is 240 series, being "
        "80 station occupancies plus 160 directional flows at two per station. The "
        "within-dataset regression of Section 7.3 uses the 160 flow series only, because "
        "HZMetro publishes no occupancy ground truth independent of the flows and the 80 "
        "occupancy series therefore cannot be scored. They remain in the panel because the "
        "multivariate models condition on them. Table 1 now has separate Panel and Scored "
        "columns for every dataset, and a new paragraph states which analysis uses which "
        "subset. The same paragraph records that ROBOD's panel is 15 series of which 5 are "
        "scored room occupancies, and that the submitted table had reported the scored count "
        "for ROBOD and the panel count for HZMetro, which is what made the two look "
        "inconsistent. We verified both counts by loading the sites rather than by "
        "recomputing them on paper.",
    ),
    (
        "Give more detail on the simulator's parameters and show the generated distributions "
        "are plausible",
        "Accepted. Section 7.11 and Figure 6 are new and document the generator rather than "
        "describing it.\n\n"
        "The text now gives, for both operating points, the room count and the capacity of "
        "each room (8 rooms, capacities 10 to 30), the doorway inventory with clear widths "
        "and the throughput each width implies (8 pairs, 0.9 to 2.2 metres, 14 to 50 persons "
        "per minute), the arrival process and its peak rate (1.6 against 7.0 arrivals per "
        "minute), the route choice model, the dwell time distribution per room, and the "
        "simulated span (508 against 653 visitors per day over 479 against 119 days). It also "
        "states where the parameters came from: the doorway throughput mapping is from the "
        "pedestrian flow literature we cite, the room capacities are set from floor area at "
        "conventional occupant densities, and the arrival and dwell parameters were chosen to "
        "produce the peak-to-mean ratios reported in museum visitor studies rather than fitted "
        "to any dataset. We say the last part plainly, because the simulator is not calibrated "
        "to a real museum and claiming otherwise is what the reviewers are right to guard "
        "against.\n\n"
        "On plausibility, the new figure shows the generated occupancy and flow distributions "
        "with the diagnostics that make them checkable: mean daily peak occupancy of 69.6 "
        "against 84.4 persons, capacity binding on 2.7 against 8.0 percent of room-steps, "
        "the entrance moving from 1.4 to 20.5 percent of its capacity between the two "
        "operating points and the parlour from 12.1 to 28.4, a 22-fold spread in mean "
        "occupancy between the busiest and quietest rooms, and per-series peak-to-mean ratios "
        "from 4.5 to 51. The crowded configuration exists because of this comment and R1.2: "
        "at the baseline operating point the capacity constraints were almost always slack, "
        "so the inequality constraints were barely exercised, and the second operating point "
        "makes them bind.",
    ),
    (
        "The repository describes several experiments as not executed while the manuscript "
        "reports results from those families",
        "The reviewer's inference is correct: the repository documentation had not been "
        "updated. No result in the manuscript was ever unlogged, and every number comes from "
        "a manifest under the results directory, but the README's status section still "
        "described the state of the project at an earlier point and said that only one "
        "experiment family had been run.\n\n"
        "The README now carries an explicit registry of record with a row per experiment "
        "family and its status, and it states that listing the results directory is the "
        "authoritative check and will contradict the table if it ever drifts again. Eight "
        "families have been run: the two headline real-data comparisons, the weekly baseline, "
        "the context sweep, the gradient-boosted ladder, the two training-budget sweeps, the "
        "simulator reconciliation study and the crowded scenario. Three remain genuinely "
        "unexecuted: the covariate ablation, the sensor-degradation study and the "
        "decision-utility study. The manuscript's limitations section lists exactly those "
        "three and states that it makes no claim depending on them, so the two documents now "
        "agree in both directions. We also corrected two stale cross-references in the README "
        "that attributed the reconciliation evidence to the wrong experiment identifiers, and "
        "added a note to its results section flagging that the context sweep and the "
        "gradient-boosted ladder supersede the numbers shown there.",
    ),
    (
        "Explain the aggregation across series, how the Newey-West lag relates to the origin "
        "stride, and whether the results are robust to a block-based interval",
        "Accepted, and Section 7.10 with the new Table 13 answers all three. One of the "
        "answers weakens a claim in the submitted paper, and we report it.\n\n"
        "On aggregation, the loss differential is averaged across series within an origin "
        "first, giving one scalar per origin, and the test is then applied to that univariate "
        "sequence. This is stated where the inference is described. The consequence is that "
        "cross-series dependence, including dependence between series at the same station, is "
        "absorbed into the single per-origin statistic rather than being treated as extra "
        "independent observations, which is the conservative choice and is the reason the "
        "effective sample size is the number of origins rather than the number of "
        "series-origin pairs.\n\n"
        "On the Newey-West lag, the submitted paper used the conventional horizon minus one "
        "in forecast steps, and the reviewer is right that the correct scale is the overlap "
        "between origins, which depends on the stride. We computed both. On HZMetro the "
        "origin stride is 2 steps and the horizon is 8, so the design-implied overlap is 3 "
        "origins while the published lag was 7; on ROBOD the stride is 12 and the horizon is "
        "12, so the design-implied overlap is 0 while the published lag was 11. In both cases "
        "the lag we published is the larger and therefore the more conservative of the two, "
        "which we verified rather than assumed, so no reported rejection was made easier by "
        "the choice.\n\n"
        "On robustness, we added moving-block bootstrap confidence intervals with blocks of "
        "ten consecutive origins, which respect the overlap directly instead of modelling it. "
        "On ROBOD the interval excludes zero and the finding stands. On HZMetro the interval "
        "for the foundation model's advantage over the seasonal baseline at the longest "
        "horizon contains zero, so that advantage is not statistically established under "
        "dependence even though the pointwise test rejects. We have put this in the abstract "
        "and the conclusion rather than only in a table, because it is a limitation of our own "
        "result that a reader should not have to find.",
    ),
    (
        "The data-scarcity explanation for LightGBM's poor performance has not been isolated "
        "experimentally",
        "Accepted, and when we isolated it the explanation failed. We have withdrawn it rather "
        "than demoting it to a possibility.\n\n"
        "We ran the training-data sensitivity experiment the reviewer suggests, sweeping the "
        "local model's training budget on real ROBOD data from 3 to 60 days. Twenty times more "
        "data buys at most 4.3 percent. Re-specifying the same model, by adding seasonal lags, "
        "tuning it and fitting it per series, bought 14.8 percent on the same data. "
        "Specification therefore dominates data volume by more than threefold, and on HZMetro "
        "the re-specified model overtakes the foundation models outright (R2.17). Section 7.12 "
        "reports the sweep and states that the submitted explanation is refuted: the problem "
        "was our configuration of the competitor, not the amount of history available to it. "
        "We also report that the simulator's training-budget curve contradicts both real "
        "datasets, and read that as a limitation of the generator rather than as evidence "
        "about the models.",
    ),
    (
        "The opening claim contradicts Table 2, where last-value persistence wins at five "
        "minutes on ROBOD",
        "Accepted; the opening sentence was an overstatement contradicted by the table on the "
        "same page. It is now qualified in place rather than later. The results section opens "
        "by stating that the foundation models lead from 15 minutes onwards on both sites, "
        "and that at the shortest ROBOD horizon last-value persistence is better, with the "
        "mechanism given immediately: at five minutes the best forecast of a room's occupancy "
        "is its current occupancy, because the room's state has barely had time to change, and "
        "no amount of pre-training improves on knowing the present when the present is almost "
        "the answer. Hypothesis 1 was always stated over 15 to 60 minutes, so the five-minute "
        "result does not bear on it, and the text now says that explicitly instead of leaving "
        "the reader to reconcile the claim with the table.",
    ),
    (
        "The context comparison is limited to two lengths; does the improvement continue, "
        "saturate or reverse at three or seven daily periods?",
        "Accepted, and the extended sweep answers the question in a way that qualifies our "
        "own reporting convention. We swept one, two, three, seven and fourteen daily seasons "
        "on both real datasets, comparing on the subset of origins that every context length "
        "can score, so that the comparison is not confounded by longer contexts dropping "
        "origins for want of history.\n\n"
        "The improvement continues well past two seasons and saturates near seven. On HZMetro "
        "at 120 minutes the best foundation model goes 1.037, 0.677, 0.557, 0.424, 0.414 MASE "
        "across the five context lengths; the step from one to two seasons is the largest, but "
        "the step from two to seven is still 37 percent and the step from seven to fourteen is "
        "within 2.5 percent, which is where the curve flattens. On ROBOD at 60 minutes the "
        "pattern is the same shape and about two fifths as large. Section 7.3 and Table 5 are "
        "rewritten around the five-point sweep, and the abstract, introduction and conclusion "
        "now say that two seasons is a reporting convention rather than a saturation point. "
        "The methodological warning is unchanged and strengthened: a zero-shot comparison "
        "against a periodic baseline that does not report and vary its context length is "
        "reporting a property of its configuration.",
    ),
    (
        "The panel titles read \u201cFigure 0. ROBOD\u201d, and the legend and axis labels "
        "should be enlarged",
        "Both fixed. The panel titles were generated inside the image by the plotting script, "
        "which is why they were not LaTeX subcaptions and why the numbering was meaningless; "
        "they now read \u201c(a)\u201d and \u201c(b)\u201d, matching the LaTeX subcaptions of "
        "the enclosing figure, and we confirmed the rendering by extracting the text back out "
        "of the compiled PDF rather than by eye. Every figure was regenerated with larger "
        "type: base font and axis labels at 10 point, tick labels at 9, legends at 9, against "
        "the smaller sizes used in the submission. We also took the opportunity to fix two "
        "legibility faults the reviewer did not have to point out, namely a legend that "
        "occupied roughly 40 percent of one figure's height, now moved below the axes and "
        "reduced in content, and a duplicated y-axis label that printed over the "
        "neighbouring panel's tick numbers.",
    ),
    (
        "The better-calibration claim is not supported by Table 5; distinguish calibration "
        "from sharpness and compare widths at a common empirical coverage",
        "The reviewer is right, the claim was refuted by the numbers we ourselves printed, and "
        "we withdraw it. On ROBOD, Toto-2.0 covers 90.1 percent and seasonal naive covers 83.9 "
        "percent against a nominal 80 percent, so the deviations from nominal are 10.1 and 3.9 "
        "points and the baseline is the better calibrated of the two. The honest description of "
        "Toto-2.0 is that it is miscalibrated in the safe direction, and that is what the paper "
        "now says.\n\n"
        "We then did the matched-coverage comparison the reviewer asks for. Section 7.5 and "
        "Table 8 are new. Each method's interval is rescaled by the factor that brings its "
        "empirical coverage to a common 80 percent, estimated from the standardised absolute "
        "errors, and the widths are compared after rescaling. At matched coverage Toto-2.0 "
        "needs an interval of 0.77 persons and seasonal naive needs 3.75, so the foundation "
        "model is 4.9 times sharper at the same coverage. This is a cleaner and stronger result "
        "than the one we claimed, and it is the claim the abstract, introduction and conclusion "
        "now make: the advantage is sharpness at matched coverage, not calibration. Hypothesis "
        "3 is reported as partly refuted, with the sharpness half holding and the calibration "
        "half failing. We also disclose a limitation the rescaling exposed: for last-value "
        "persistence the intervals are degenerate, having zero width at most points, so no "
        "finite rescaling reaches nominal coverage; those cells report as not available with "
        "the reason given, rather than reporting a width of zero.",
    ),
    (
        "Separate the strong feasibility finding from the limited accuracy evidence, and add "
        "confidence intervals for the mean paired change",
        "Accepted, and the separation is now structural rather than a matter of wording.\n\n"
        "The paper now states three claims at three different strengths. Exact coherence and "
        "exact feasibility are guarantees, not estimates: they follow from the projection and "
        "are confirmed to solver tolerance at every origin. The internal design choices within "
        "the layer are established with high confidence, since the ablations of Table 10 are "
        "significant at p below 1e-7 with confidence intervals excluding zero. The improvement "
        "over the raw forecast is a fraction of a percent and is not statistically established: "
        "0.98 percent for TimesFM-3 and 0.10 percent for LightGBM, neither significant, and we "
        "now report that in its own block of the table with the reviewer's p = 0.18 included. "
        "We also explain why the ablations are overwhelmingly significant while this comparison "
        "is not, which is that the raw forecast is already nearly coherent on most origins, "
        "with a mean residual of 1.05 persons concentrated in a minority of occasions, so the "
        "projection has almost nothing to do on a typical origin and the paired difference is "
        "near zero.\n\n"
        "Confidence intervals are added throughout. Every paired change in the reconciliation "
        "tables now carries a 95 percent moving-block bootstrap interval with blocks of ten "
        "consecutive origins, alongside the mean, the median and the fraction of origins made "
        "worse, which is the set of statistics the asymmetry discussed in the next point makes "
        "necessary.",
    ),
    (
        "Seventy percent of origins become slightly worse; examine peak occupancy error, "
        "capacity exceedance or another risk-sensitive metric",
        "Accepted, and this produced what we consider the most useful result of the revision. "
        "The reviewer's hypothesis, that an occasional large improvement is valuable if it "
        "arrives near high occupancy, is correct and is now measured.\n\n"
        "We added three risk-sensitive metrics: the absolute error in the predicted peak "
        "occupancy over the horizon, the mean absolute error restricted to high-occupancy "
        "steps, and a Brier score for predicted capacity exceedance. On the baseline "
        "configuration the proposed projection reduces peak occupancy error from 1.298 to "
        "1.217 persons, a reduction of 6.3 percent against 0.56 percent for mean absolute "
        "error over the same origins, so the peak effect is eleven times the mean effect. On "
        "the crowded configuration it falls from 1.113 to 1.068, a reduction of 4.0 percent "
        "against 0.57 percent, a ratio of seven. Every reconciler improves the peak error and "
        "the proposed arm improves it most at both operating points.\n\n"
        "The asymmetry now has a clear reading, which the paper gives: the projection nudges "
        "the many quiet origins imperceptibly in the wrong direction and corrects the few "
        "crowded ones substantially, and the crowded ones are the origins a crowd-protection "
        "system exists to get right. We state this as the honest form of the claim for the "
        "layer. It is stronger operationally than the mean-error result the submitted paper led "
        "with, and weaker in that metric, and we prefer it on both counts.",
    ),
    (
        "The recommendation to apply uncertainty-weighted reconciliation is stronger than the "
        "evidence supports",
        "Accepted. The deployment recommendation has been rewritten and now distinguishes what "
        "we can guarantee from what we have only measured in simulation.\n\n"
        "We recommend the constrained projection on the strength of feasibility, because that "
        "is mathematics and holds for any admissible truth, at a cost of 6 to 7 milliseconds. "
        "We offer the uncertainty weighting as the better-performing variant in our simulations "
        "rather than as an established improvement, and we say in the same paragraph that the "
        "effect sizes are around one percent, that one comparison against the raw forecast is "
        "not significant, and that none of it has been reproduced on a real building. We also "
        "quantify what a practitioner who declined the recommendation would forgo, which is "
        "about one percent of accuracy and no guarantee, so that the reader can weigh it rather "
        "than take our word for it. The forecasting half of the recommendation has been "
        "rewritten for the same reason and now turns on a condition the practitioner can check, "
        "namely whether the site has the history and the engineering capacity to specify a "
        "local model properly, with the honest admission that we can offer no statistic that "
        "predicts which way that comparison will go.",
    ),
    (
        "Qualify two statements in the conclusion: the context dependence of the forecasting "
        "advantage, and that feasibility is guaranteed for the projected median rather than the "
        "whole distribution",
        "Both accepted, and the conclusion has been rewritten around the distinction between "
        "the two halves of the paper receiving answers of different strengths.\n\n"
        "On the first, the conclusion now states that the forecasting advantage is "
        "context-dependent and is not statistically established for all comparisons. It gives "
        "the three specific qualifications rather than gesturing at them: the comparison "
        "against a well-specified local model splits by site, ahead 26.8 percent on occupancy "
        "and behind 28.9 percent on flow; the comparison against a seasonal baseline is decided "
        "by context length and keeps improving to about seven seasons; and the moving-block "
        "bootstrap interval for the metro advantage at the longest horizon contains zero, so "
        "that advantage does not survive a dependence-aware interval even where the pointwise "
        "test rejects.\n\n"
        "On the second, the conclusion now says that the guarantee covers the projected median "
        "path and not the whole predictive distribution, gives the size of the residual "
        "incoherence at the extreme quantiles under the default shift strategy, and notes that "
        "the per-quantile strategy removes it at a cost in the quantile loss. It carries two "
        "further boundaries in the same paragraph: that every reconciliation measurement comes "
        "from the simulator, so the layer is demonstrated rather than validated and "
        "real-building validation is the principal item of future work; and that the "
        "operationally relevant effect is the peak error reduction rather than the mean one. "
        "The abstract was rewritten to match, so that no claim appears there at a strength the "
        "conclusion does not support.",
    ),
]


def add_body(document: Document, text: str) -> None:
    """One paragraph per blank-line-separated block, with modest spacing."""
    for block in text.split("\n\n"):
        paragraph = document.add_paragraph(block)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.space_after = Pt(6)


def add_quote(document: Document, text: str) -> None:
    """A reviewer comment, indented and italic so it cannot be mistaken for a reply."""
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Pt(24)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True


def add_label(document: Document, text: str) -> None:
    """A small coloured run introducing either the comment or the response."""
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = ACCENT


def build() -> None:
    comments = reviewer_two_comments()
    if len(comments) != len(REVIEWER_TWO_REPLIES):
        raise AnswerError(
            f"{len(comments)} comments but {len(REVIEWER_TWO_REPLIES)} replies"
        )

    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    document.add_heading("Response to the reviewers", level=0)
    document.add_paragraph(
        "Topology-Constrained Reconciliation for Zero-Shot Forecasting of Building "
        "Occupancy and Doorway Flow"
    )
    document.add_paragraph("A. A. Jafari, C. Ozcinar and G. Anbarjafari")

    add_body(
        document,
        "We thank both reviewers. Their objections changed several of this paper's "
        "conclusions rather than only its presentation, and we have said so in the "
        "acknowledgment. Three of the paper's claims are retracted outright, one "
        "mathematical error is corrected, one previously unstated constraint is written "
        "into the formulation, and nine new experiments were run. We have resisted the "
        "temptation to soften any of this: where a reviewer showed that we were wrong, the "
        "manuscript now says we were wrong, keeps the original claim visible, and attaches "
        "the refutation.\n\n"
        "Every point is answered below, individually and in the order it was raised. "
        "Reviewer 2's comments are quoted verbatim from the annotations embedded in the "
        "review PDF, extracted programmatically so that the quotations cannot drift. "
        "Reviewer 1's points arrived as prose rather than as annotations and are restated "
        "in our own words, which we flag so that the reviewer can check we have not "
        "reshaped them.\n\n"
        "Every addition to the manuscript is set in red. Every number quoted below comes "
        "from a logged run; the script that generates every table and figure fails rather "
        "than emitting a placeholder when a run is missing, so no figure in this response "
        "is an estimate.",
    )

    document.add_heading("Summary of what changed", level=1)
    add_body(
        document,
        "Retracted. (1) That the foundation models beat every local baseline on both real "
        "datasets: a properly specified local model wins on HZMetro by 28.9 percent. (2) "
        "That the foundation models are better calibrated: the seasonal baseline is nearer "
        "nominal coverage, and what survives is a sharpness claim at matched coverage, where "
        "the foundation model is 4.9 times narrower. (3) That the quantile shift keeps the "
        "whole predictive distribution coherent: only the median path is coherent.\n\n"
        "Corrected. The eigenvalue bound on the weight matrix, which had an inequality "
        "reversed. Corollary 5, which claimed an equality the theorem does not establish, now "
        "an inequality with the reviewer's counterexample included. The description of the "
        "admissible set as bounded, now compact and only because the doorway throughput bound "
        "has been written into the formulation. The affine versus linear distinction in the "
        "MinT correspondence. The conservation assumptions, now three conditions rather than "
        "one wrong one. The attribution of the MinT difference, now decomposed into a "
        "weighting effect and a box effect, of which the weighting effect is three times the "
        "larger. The solver configuration, which made the solution weakly dependent on the "
        "scale of the predictive spread. The word \u201cpre-registered\u201d. The HZMetro "
        "series counts. The reconciliation solve time. The repository's experiment registry.\n\n"
        "Added. Nine experiments: the five-point context sweep on both sites, the weekly "
        "seasonal baseline, the gradient-boosted ladder, the real-data training-budget sweep, "
        "the crowded simulator scenario, the equality-only arm with the proposed weights, the "
        "per-quantile reconciliation comparison, the interval-width sensitivity study and the "
        "dependence study with moving-block bootstrap intervals. Four new sections, six new "
        "tables, one new figure, and new Funding, Acknowledgment, Data and code availability "
        "and Competing interest statements. The title has changed.",
    )

    document.add_heading("Reviewer 1", level=1)
    add_body(
        document,
        "Reviewer 1's points were given as prose. We restate each one in our own words before "
        "replying, so that the reviewer can verify we have not narrowed the point in the "
        "restatement.",
    )
    for heading, restatement, reply in REVIEWER_ONE:
        document.add_heading(heading, level=2)
        add_label(document, "Comment, as we understand it")
        add_quote(document, restatement)
        add_label(document, "Response")
        add_body(document, reply)

    document.add_heading("Reviewer 2", level=1)
    add_body(
        document,
        "All thirty annotations embedded in the review PDF are reproduced verbatim below, in "
        "page order, each followed by our response. The page number is the page of the "
        "reviewed proof on which the annotation appears.",
    )
    for index, ((page, comment), (heading, reply)) in enumerate(
        zip(comments, REVIEWER_TWO_REPLIES, strict=True), start=1
    ):
        document.add_heading(f"R2.{index} (p. {page}) {heading}", level=2)
        add_label(document, "Comment, verbatim")
        add_quote(document, comment)
        add_label(document, "Response")
        add_body(document, reply)

    document.add_heading("What we were unable to do", level=1)
    add_body(
        document,
        "One request we could not satisfy, and we would rather state it here than let it be "
        "discovered. Both reviewers asked for the reconciliation layer to be tested on a real "
        "building measuring both occupancy and doorway flow. We do not have such data and we "
        "could not obtain it. No public dataset we could find reports both quantities for one "
        "building on a common clock, and neither of our real datasets can substitute, since "
        "ROBOD lacks doorway counts and HZMetro lacks station occupancy, so in each case one "
        "side of the continuity equation is missing and the constraint cannot be formed at "
        "all. The ATC indoor tracking dataset is the nearest candidate but provides no "
        "occupancy measurement independent of the tracks that give the flows, so using it "
        "would make the constraint hold by construction and test nothing.\n\n"
        "Rather than present a weak proxy as a validation, we have done three things. The "
        "title no longer claims the setting we cannot evidence. The manuscript states the "
        "boundary in the abstract, in a dedicated introduction paragraph, in the contribution "
        "list, in the limitations and in the conclusion, and says in the limitations exactly "
        "what a dataset would need to contain for the test to be possible. And we strengthened "
        "the simulated evidence where strengthening was available, by adding an operating point "
        "at which the constraints actually bind, by adding dependence-aware confidence "
        "intervals, and by adding the risk-sensitive metrics that turned out to carry the "
        "layer's real operational case.\n\n"
        "Three experiments from our original design also remain unrun: the covariate ablation, "
        "the sensor-degradation study and the decision-utility study. No claim in the revised "
        "manuscript depends on any of them, the limitations section lists all three, and the "
        "repository's registry now agrees.",
    )

    document.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
