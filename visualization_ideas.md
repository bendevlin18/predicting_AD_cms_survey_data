# Recall vs. False Positive Tradeoff — Visualization Ideas

The core clinical tension: **missing an AD case** (low recall) delays intervention, but **flagging a healthy person** (false positive) triggers unnecessary follow-up testing, patient anxiety, and wasted resources. These visualizations explore different angles of that tradeoff.

---

## 1. ROC Curves by Learning Rate

**What it shows:** One ROC curve per learning rate, all overlaid on the same (FPR, Recall) plane. Each curve traces how a given model moves through tradeoff space as the decision threshold changes. Points along each curve can be annotated with threshold values.

**Why it's clinically useful:** ROC curves are the standard language for communicating diagnostic tradeoffs in clinical and epidemiological literature. Clinicians and reviewers will immediately understand this. Overlaying multiple learning rates on one plot also answers: *does the learning rate meaningfully change the available tradeoffs, or are we mostly just sliding along a similar curve?*

**Implementation:** Matplotlib line plot. Each of the 15 learning rates becomes a line; color by learning rate. Optionally highlight the Pareto frontier (see idea 5) or annotate a few key threshold values.

---

## 2. Decision Curve Analysis (Net Benefit)

**What it shows:** For each decision threshold, compute the **net benefit**:

```
Net Benefit = (TP / N) - (FP / N) * (threshold / (1 - threshold))
```

Plot net benefit (y-axis) vs. threshold (x-axis), comparing the XGBoost model against two reference strategies: **"screen everyone"** (assume all positive) and **"screen no one"** (assume all negative).

**Why it's clinically useful:** Decision curve analysis is the gold standard in clinical prediction modeling papers (Vickers et al., 2006). It directly encodes the idea that a false positive at threshold=0.10 is less costly than one at threshold=0.50, because a clinician choosing a low threshold is implicitly saying they're willing to accept more false positives. The region where the model's curve sits *above* both reference lines is the range of thresholds where using the model adds clinical value.

**Implementation:** One curve per learning rate (or just the default lr=0.05). The "screen all" line is a known formula; "screen none" is always 0. This is a standard plot in clinical ML papers and would strengthen any write-up.

---

## 3. Number Needed to Screen (NNS) vs. Recall

**What it shows:** For each (learning rate, threshold) combination, compute:

```
NNS = Total Predicted Positive / True Positives = 1 / PPV
```

This answers: *"For every true AD case the model catches, how many total patients does it flag?"* Plot NNS (y-axis) vs. Recall (x-axis), colored by threshold.

**Why it's clinically useful:** NNS is immediately intuitive for clinicians — it translates the abstract FPR into a concrete workload question. A point at (recall=0.85, NNS=8) means *"we catch 85% of AD cases, but we flag 8 people for every real case."* Decision-makers can directly assess whether that screening burden is acceptable given their resources. The curve typically shows a sharp elbow where pushing for a bit more recall causes NNS to explode.

**Implementation:** Scatter plot, one point per grid combination. Color by threshold (or learning rate). Annotate the "elbow" region where marginal recall gains become very expensive in false positives.

---

## 4. Contour Plot of Recall - FPR over Parameter Space

**What it shows:** A smooth, interpolated surface of the (Recall - FPR) score over the learning rate (y-axis) x threshold (x-axis) grid. Contour lines connect regions of equal net performance, creating a topographic map of the parameter landscape.

**Why it's clinically useful:** The heatmap shows discrete cells; the contour plot shows the *gradient* — how quickly performance degrades as you move away from the sweet spot. This helps answer: *"How sensitive is our chosen operating point? If we nudge the threshold slightly, does performance fall off a cliff or degrade gracefully?"* Robustness matters in clinical deployment where patient populations shift over time.

**Implementation:** Matplotlib `contourf` with interpolation. Overlay a marker at the current operating point (lr=0.05, threshold=0.20). Optionally add contour line labels.

---

## 5. Pareto Frontier on the FPR-Recall Plane

**What it shows:** Starting from the existing FPR vs. Recall scatter plot, identify and highlight the **Pareto-optimal** points — combinations where you cannot improve recall without increasing FPR (or vice versa). Connect these into a frontier line. All other points fall below/right of this frontier.

**Why it's clinically useful:** The Pareto frontier is the "menu" of achievable tradeoffs. Every point on it represents a defensible clinical choice; every point below it is dominated by something better. This reframes the conversation from *"which single threshold is best?"* to *"here are the best available options — which tradeoff fits your clinical context?"* A primary care screening program might pick a high-recall point on the frontier; a confirmatory diagnostic setting might pick a low-FPR point.

**Implementation:** Build on the existing scatter plot. Compute the Pareto frontier (sort by FPR ascending, keep points where recall is non-decreasing). Draw the frontier as a bold line or shaded region. Annotate frontier points with their (lr, threshold) values.
