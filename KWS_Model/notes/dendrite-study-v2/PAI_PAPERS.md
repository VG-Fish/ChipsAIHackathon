# PerforatedAI / Perforated Backpropagation papers — reading record

**Updated:** 2026-09-17

## Scope and search boundary

This is a reading record for the research that evaluates **Perforated
Backpropagation (PB)** or the PerforatedAI system itself.  It is deliberately
separate from the broader dendritic-network literature.

The official [PerforatedAI papers index](https://github.com/PerforatedAI/PerforatedAI/tree/main/papers)
currently contains nine historical and related-work entries.  Only two of
those are PB papers: the method paper and the follow-up experiments paper.
The repository's main README cites the method paper.  A third PB paper,
[Perforated Neural Networks for Keyword Spotting](https://arxiv.org/abs/2605.15647),
was published after that index and is directly relevant to this KWS repository.

I read the three PB papers rather than treating related-work citations as
evidence for PB.  They are all arXiv preprints, not peer-reviewed venue
publications.  This document records the evidence boundary so a future agent
does not accidentally turn vendor claims into established facts.

## The complete PB-paper set located

| ID | Paper | Status | What it adds | Direct relevance here |
| --- | --- | --- | --- | --- |
| P1 | Rorry Brenner and Laurent Itti, [*Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks*](https://arxiv.org/abs/2501.18018), 2025 | Vendor-authored arXiv preprint | Method, phase alternation, first multi-domain experiments | Defines the optimisation mechanism exercised by our `perforatedbp` run. |
| P2 | Rorry Brenner et al., [*Exploring the Performance of Perforated Backpropagation through Further Experiments*](https://arxiv.org/abs/2506.00356), 2025 | Vendor-authored arXiv preprint | Hackathon experiments on BERT, ProteinBERT, and MobileNetV3 | Shows PB can be evaluated on smaller models, but all results are single-run. |
| P3 | Vishy Gopal et al., [*Perforated Neural Networks for Keyword Spotting*](https://arxiv.org/abs/2605.15647), 2026 | PerforatedAI-affiliated arXiv preprint | 800-trial edge/KWS sweep on Edge Impulse | Closest published task; importantly, architecture, split, and sweep differ from SparkNet study-v2. |

No fourth PB research paper was linked from the official index or found by
searching the PerforatedAI GitHub organisation and arXiv for the exact terms
``Perforated Backpropagation`` and ``PerforatedAI`` on the update date.  This
is a reproducible search result, not a claim that no future paper can exist.

## P1 — method paper

**Method.** PB first trains the base network.  It then enters a dendrite phase:
base neuron weights are frozen, candidate copies of selected modules are
trained to correlate with each neuron's remaining error, the candidate is
selected/frozen, and a subsequent neuron phase resumes ordinary learning.
The paper explicitly keeps the dendrite term out of the usual backpropagated
error path.  This is the reason that phase handling, optimizer reconstruction,
and validation-boundary calls in `src/kws/optimize/dendritic.py` are
load-bearing—not optional bookkeeping.

**Reported evidence.** The paper reports TrimNet/Tox21 results over 50 seeds,
HIST/CSI300 over 10 seeds, a small EMNIST experiment, and a
parameter-controlled mTAN/PhysioNet sweep.  The mTAN comparison is the only
experiment that directly tests a smaller PB model against a wider standard
model at a stated parameter budget.  The authors also report that larger mTAN
models overfit when a first dendrite is added, and that the reduced model's
training took substantially more epochs.

**How strongly to use it.** It supports the mechanics and motivates testing
very narrow SparkNet widths.  It does not establish that PB will help every
already-competent small KWS model: datasets, architectures, parameter budget,
and stopping policies differ.  The paper's paper-level figures are
vendor-authored preprint results, so our paired five-seed comparison remains
the deciding evidence for this project.

## P2 — further experiments

**Reported evidence.** This follow-up reports BERT-family work on SNLI and
IMDB, ProteinBERT/AMP-BERT, MobileNetV3 on CIFAR-10, and deployment cost
measurements.  Its best-known claims are up to 88.7% fewer parameters without
an IMDB accuracy loss and a narrower MobileNetV3 plus dendrites reaching a
similar/better reported result with 35% fewer parameters than the original.

**Critical limitation.** The paper states that its hackathon setting did not
produce error bars for repeated runs.  Every reported configuration is a
single run.  Its figures can suggest hypotheses for an arm or width sweep but
cannot resolve an effect at the 0.1--1 pp scale of study-v2.

## P3 — keyword spotting paper

**Design.** P3 applies PB to Edge Impulse's tutorial KWS pipeline and reports
800 hyperparameter trials.  It compares ordinary networks, gradient-descent
dendrites, and cascade-correlation (PB) dendrites across architecture, width,
regularisation, and dendrite settings.  Test scores are selected at the epoch
of best validation score.

**Reported headline.** The stated standard baseline has 3,859 parameters and
0.921 test accuracy.  The paper reports a PB model with roughly 1,556
parameters and 0.933 test accuracy, and a separate 11,421-parameter PB model
at 0.958.  Those are useful external benchmarks, but the 800 points are a
hyperparameter sweep rather than a fixed-recipe, paired, five-seed estimate;
the paper does not give an independent seed count or uncertainty interval for
the selected endpoints.

**What transfers to SparkNet.** The domain match makes its placement and
small-model motivations more relevant than P1/P2.  It does *not* validate the
current SparkNet recipe: SparkNet's data pipeline, parameter accounting,
identity-fine-tune step, placement arms, and selection protocol are different.
In particular, study-v2 must not claim P3's **test** figures while it is still
making all selection decisions on validation accuracy.

## Official-index related work — context, not PB evidence

The official index also links the following papers: Cascade-Correlation (1989),
Morphological Perceptrons with Dendritic Structure (2003), Efficient Training
for Dendrite Morphological Neural Networks (2014), Dendritic Neuron Model
(2018), a 2021 dendrite-ML review, Learning on Tree Architectures (2023), and
Dendrites Endow ANNs with Accurate, Robust and Parameter-Efficient Learning
(2025).  These establish the broader dendritic-ML lineage.  They do **not**
evaluate PerforatedAI/PB, and none should be cited as independent validation
of this integration.

For notes on those sources and exact methodological differences, see
[`PAI_KNOWLEDGE.md`](PAI_KNOWLEDGE.md#22-papers-the-project-cites-as-background--read-as-summaries-only).

## Implications for current decisions

1. Treat P1--P3 as motivation, not a reason to preselect an arm.  The
   study's paired scratch checkpoints and no-dendrite control are stronger
   evidence for SparkNet.
2. Preserve validation/test separation.  P3's held-out test reporting is not
   a license to inspect study-v2's test split before arm selection is frozen.
3. P1 and P3 both make small-model PB plausible, while P1 also documents
   overfitting in larger models.  That supports completing the already-running
   width/placement matrix, not changing it midstream.
4. Do not describe the primary PB papers as peer-reviewed or independent:
   all three are preprints and have PerforatedAI-affiliated authors.
