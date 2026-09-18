# PerforatedAI: Mechanism, Evidence, and Correct Integration

Authoritative digest for the SparkNet dendrite study.
Compiled 2026-09-17 from (a) perforatedai.com / github.com/PerforatedAI/PerforatedAI,
(b) the two arXiv papers the project cites for its own method, (c) the vendored
skill docs in `KWS_Model/PAI Skills/skills/`, and (d) the installed package source.

## Evidence labels used throughout

| Label | Meaning |
| --- | --- |
| **[SRC]** | Source-code fact. Read from actual library code; file + line cited. |
| **[VENDOR]** | Vendor claim from marketing site, README, API docs, or skill docs. Not independently verified. |
| **[PREPRINT]** | Result from an arXiv preprint. **Not peer reviewed.** |
| **[PEER]** | Result from a peer-reviewed venue. |
| **[INFER]** | My inference/derivation from the above. Flagged so it can be challenged. |

### Provenance and version note

- Installed package: `perforatedai 3.2.8` + `perforatedbp 3.2.7` in
  `KWS_Model/.venv/lib/python3.13/site-packages/`. **Both are Cython-compiled
  `.so` files.** `perforatedai` ships the Cython-generated `.c` files, from
  which the original Python was reconstructed (73-98% of lines recovered;
  missing lines are comments/docstrings, not statements).
- `perforatedbp` ships **only `.so`, no `.c`** — its source is not readable.
  Its configuration defaults were recovered by importing
  `perforatedbp.globals_pbp` and dumping module attributes. Its *algorithms*
  are stated below only where the papers or docs describe them, and are
  labelled accordingly — **I could not read perforatedbp's code.**
- Line numbers below refer to the GitHub repo at `main` (setup.py declares
  `version="3.2.6"`), cloned to a scratch dir. Its `perforatedai/*.py` matches
  the reconstructed 3.2.8 source except where noted. **Known 3.2.6 -> 3.2.8
  diff:** `first_fixed_switch_num` default changed `1` -> `-1`.
  Repo path prefix below: `PerforatedAI/perforatedai/`.

---

## 1. Mechanism

### 1.1 What a PAI dendrite actually is

**[SRC]** A dendrite is a *full structural copy of the wrapped module*, not a
small side-branch. `PAIDendriteModule.create_dendrite` is literally a deep copy:

> `PerforatedAI/perforatedai/modules_perforatedai.py:1209`
> ```python
> return UPA.deep_copy_pai(parent_module)
> ```

Wrapping replaces module `M` with a `PAINeuronModule` holding:
- `main_module` — the original `M` (the "neuron"),
- `dendrite_module` (a `PAIDendriteModule`) holding
  - `parent_module` — a permanent deep copy of `M` used *only* as a template
    for minting new dendrites (`modules_perforatedai.py:1115`). It is **not in
    the forward path** and is explicitly excluded from parameter counts
    (`utils_perforatedai.py`, `count_params`: `{p.data_ptr(): p for name, p in
    parameters if "parent_module" not in name}`).
  - `layers` — an `nn.ModuleList` of accepted dendrites, each a copy of `M`,
  - `dendrites_to_dendrites` — cascade weights between dendrites,
- `dendrites_to_top` — the dendrite→neuron output weights.

### 1.2 The forward equation

**[SRC]** `PAIDendriteModule.forward` (`modules_perforatedai.py:1451-1553`) and
`PAINeuronModule.forward` (`modules_perforatedai.py:770-878`) together compute,
for a wrapped module with `D` accepted dendrites, input `x`, and neuron output
`y = M(x)`:

```
z_d  = layers[d](x) + sum_{e<d}  W_dd[d][e,:] * a_e          # cascade input
a_d  = f(z_d)                                                 # f = pai_forward_function
y'   = y + sum_{d=0..D-1}  a_d * W_top[D-1][d,:]
```

The exact code for the cascade term and activation:

> `modules_perforatedai.py:1530-1541`
> ```python
> current_out = (
>     current_out
>     + self.dendrites_to_dendrites[out_index][in_index, :]
>     .view(view_tuple).to(current_out.device)
>     * outs[in_index]
> )
> outs[out_index] = GPA.pc.get_pai_forward_function()(current_out)
> ```

and the combination with the neuron:

> `modules_perforatedai.py:825-826`
> ```python
> to_top = self.dendrites_to_top[self.dendrite_modules_added - 1][i, :]
> ...
> out = out + (dendrite_outs[i].to(out.device) * to_top.to(out.device))
> ```

**So the combination is ADDITIVE, with a per-output-channel learned scalar
gain applied to a squashed dendrite activation.** It is not multiplicative
gating of the neuron output, and it is not concatenation. `W_top[D-1][d,:]` has
one scalar per output channel (per "neuron"), broadcast over every other
dimension — the `to_top.unsqueeze(dim)` loop at `modules_perforatedai.py:827-830`
inserts singleton dims everywhere except `this_node_index`.

**[SRC]** `f` defaults to `torch.sigmoid`:
> `globals_perforatedai.py:1037` — `self.pai_forward_function = torch.sigmoid`

Permitted shorthands are `sigmoid`, `relu`, `tanh`
(`globals_perforatedai.py`, `_TORCH_ACTIVATION_SHORTHAND`; also stated in
`configuration_descriptions.json`: *"Activation function used by dendrites
where applicable. Options are relu, tanh, and sigmoid."*).

**[SRC] A newly accepted dendrite starts at exactly zero influence.** When
switching back to neuron mode, the new row of `dendrites_to_top` is
zero-initialised:

> `modules_perforatedai.py:661-672`
> ```python
> values = torch.cat((
>     self.dendrites_to_top[self.dendrite_modules_added - 1],
>     nn.Parameter(torch.zeros((1, self.out_channels), ...)),
> ), 0)
> ```

**[INFER]** This is the key structural safety property: adding a dendrite is
output-preserving at the instant of addition. Any degradation you see right
after a switch comes from the *reload of a previous checkpoint* and from
optimizer/scheduler reset, **not** from the dendrite perturbing the function.

**[SRC] `out_channels` is discovered from the backward pass, not declared.**
`filter_backward` (`modules_perforatedai.py:177-268`) is registered as a hook on
the neuron output and on first call reads `grad_out.shape[this_node_index]` to
size all dendrite arrays. Consequence: **a module through which no gradient
ever flows can never be given dendrites**, and `set_mode("p")` fails on it with
an explicit error listing the causes (frozen weights, module defined but
unused, module with no trainable weights) — `modules_perforatedai.py:726-750`.

### 1.3 n mode vs p mode, and what is frozen

**[VENDOR/PREPRINT]** The cycle, as stated in arXiv:2506.00356 §3:

> "1) Train the original network until convergence; 2) Freeze the original
> network's weights and add new PB nodes, then train these with the objective of
> correcting any remaining errors still made by the frozen original network; 3)
> Freeze the PB weights, un-freeze the original network's weights, and iterate
> back to 1) until no further performance improvement is obtained."

**[SRC]** Mode is driven by the tracker:
- `set_dendrite_training()` (`tracker_perforatedai.py:2200`) calls
  `layer.set_mode("p")` on every neuron module, then `create_new_dendrite_module()`,
  which mints `global_candidates` fresh randomly-initialised candidate copies.
- `set_neuron_training()` (`tracker_perforatedai.py:2246`) calls `set_mode("n")`,
  which deep-copies `best_candidate_module` into `layers`, appends the zero row
  to `dendrites_to_top`, appends the `dendrites_to_dendrites` block, increments
  `num_dendrites`, then `del self.candidate_module, self.best_candidate_module`.

**[SRC]** Freezing is enforced through `requires_grad` plus optimizer
filtering. `setup_optimizer` filters every param group:
> `tracker_perforatedai.py:1799` — `filtered_group_params = [p for p in param_group["params"] if p.requires_grad]`

and in p mode with no explicit params it uses only
`UPA.get_pai_network_params(net)` (`tracker_perforatedai.py:1773`).

**[VENDOR]** The skill doc states plainly that in p mode the base weights are
out of the optimizer:
> `PerforatedAI/skills/perforatedai-complex-methods/SKILL.md`:
> "`scaler.scale(loss).backward()` runs — this registers inf checks only for
> params that receive gradients through this call, which are the
> **main_module params** (not in the optimizer in p mode)"

**[SRC + INFER] Important: the actual freezing and the correlation learning
rule live in `perforatedbp`, not in the open-source package.** In
`perforatedai` alone, `filter_backward` only sizes buffers; every candidate
code path is gated on `GPA.pc.get_perforated_backpropagation()`
(`modules_perforatedai.py:1546-1556`, `:833-836`). That flag is set
automatically at import time **iff `perforatedbp` is importable**:

> `globals_perforatedai.py:1327-1341`
> ```python
> try:
>     import perforatedbp.globals_pbp as perforatedbp_globals
>     print("Building dendrites with Perforated Backpropagation")
>     pc.set_perforated_backpropagation(True)
>     pc.set_no_extra_n_modes(False)
> except ImportError:
>     print("Building dendrites without Perforated Backpropagation")
> ```

`perforatedbp 3.2.7` **is installed in this project's venv**, so this project
runs the true PB algorithm (correlation-trained, gradient-perforated dendrites),
not the degraded gradient-descent fallback. If it were ever missing, dendrites
would silently become plain zero-gain GD copies and `p` phases would become
no-ops — a serious silent failure mode.

### 1.4 The "perforation" itself

**[PREPRINT]** arXiv:2501.18018 Eq. (2): the standard backprop error term is
split and the dendrite branch is multiplied by zero:

> "δj = g'(inj) [ Σ_i W_i,j δ_i + 0 · Σ_k W_k,j δ_k ]"

i.e. gradient flows through neuron→neuron connections only; the dendrites are
**in the forward graph but removed from the backward graph**. Dendrites instead
learn by a Cascade-Correlation-style rule (Eq. 3-4), correlating their own
activation with the backpropagated error of the single neuron they attach to:

> "δk = (g(ink) − ḡ(ink))(δi − δ̄i)"

**[VENDOR]** Mechanically this is implemented as a **second, separate
`backward()` fired from a backward hook**:
> `skills/perforatedai-complex-methods/SKILL.md`:
> "PAI uses a backward hook to run a **second, separate `loss.backward()`** for
> dendrite training in p mode. This hook fires after your main backward call."

### 1.5 Parameter-count cost of one dendrite — derived from source

**[SRC/INFER]** After `D` accepted dendrites on a wrapped module `M` with
`P_M` trainable parameters and `C` output channels at `this_node_index`:

| Component | Shape / count | Cumulative after D |
| --- | --- | --- |
| `layers[d]` (copy of `M`) | `P_M` each | `D · P_M` |
| `dendrites_to_top[k]`, k=0..D-1 | `(k+1, C)` | `C · D(D+1)/2` |
| `dendrites_to_dendrites[k]`, k=1..D-1 | `(k, C)` | `C · D(D−1)/2` |

Closed form: **added params = `D · P_M + D² · C`**, and for one dendrite
**`P_M + C`**.

Derivation of the two triangular series: `dendrites_to_top` is an
`nn.ParameterList` that is *appended to*, never replaced
(`modules_perforatedai.py:676-700`), while the forward only reads the last
entry (`:825`). `dendrites_to_dendrites` grows the same way
(`modules_perforatedai.py:1385-1398`) starting empty and getting its first real
block at the second dendrite.

**[INFER] The older `dendrites_to_top` rows are dead weight.** They stay
registered, stay `requires_grad=True`, and therefore stay in `count_params`
and in the optimizer, but only `dendrites_to_top[D-1]` is read in forward. For
`D<=3` and a `C` of 2-32 this is negligible; it is noted for completeness.

`parent_module` is a *second* full copy of `M` present from the moment of
wrapping, excluded from `count_params` but real in RAM and in `state_dict`.
During a `p` phase there are additionally `candidate_module` and
`best_candidate_module` (one copy each per `global_candidates`), i.e. the peak
resident copies of `M` are `1 (main) + 1 (parent) + D (layers) + 2 (candidates)`.

#### Worked numbers for the three requested layer types

Take SparkNet at block width `C`, `gate_channels=32`, 40 mel bins, 12 classes
(computed by instantiating the model, not estimated):

| Wrapped module | `P_M` | node-dim `C_out` | +1 dendrite | +3 dendrites |
| --- | --- | --- | --- | --- |
| `Conv2d(C,C,1,bias=False)` (pointwise), C=12 | 144 | 12 | **156** | 1,164 |
| `Conv2d(C,C,1,bias=False)` (pointwise), C=2 | 4 | 2 | **6** | 20 |
| `Conv2d(C,C,(1,29),groups=C,bias=False)` (depthwise), C=12 | 348 | 12 | **360** | 2,476 |
| `Conv2d(C,C,(1,29),groups=C,bias=False)` (depthwise), C=2 | 58 | 2 | **60** | 192 |
| `Conv2d(C,32,1,bias=True)` (`gate_conv`), C=2 | 96 | 32 | **128** | 576 |
| `Linear(32,12)` (`fc`) | 396 | 12 | **408** | 1,296 |

**[SRC] A dendrite on a depthwise conv is itself depthwise.** `create_dendrite`
deep-copies the module including `groups=C`, so dendrite channel *i* sees only
input channel *i*. **[INFER]** Such a dendrite can only add a per-channel
nonlinear temporal filter; it cannot introduce any cross-channel mixing. That
is a much weaker function class than a dendrite on a pointwise conv.

**[PREPRINT]** The paper's own rule of thumb agrees with the `D·P_M` term:
> arXiv:2501.18018, *Test Addressing the Increase in Parameters*: "Each time
> dendrites are added to these systems, when all modules are targeted, the total
> number of free parameters of the architecture increases by approximately the
> total number of parameters of the original model."

---

## 2. The papers

### 2.1 Papers I actually read in full

#### P1 — Perforated Backpropagation (the method paper)

**Citation.** R. Brenner and L. Itti, "Perforated Backpropagation: A
Neuroscience Inspired Extension to Artificial Neural Networks,"
arXiv:2501.18018v2 [cs.NE], 31 Aug 2026 (v1: 29 Jan 2025). 11 pages.
<https://arxiv.org/abs/2501.18018> / <https://arxiv.org/pdf/2501.18018>

**Status. [PREPRINT] — arXiv only, not peer reviewed.** Author 1 is
affiliated "Perforated AI Inc." as well as USC. Treat all numbers as
vendor-authored.

**Claims, with conditions:**

| Experiment | Baseline | Reported delta | Conditions |
| --- | --- | --- | --- |
| TrimNet on Tox21 (MoleculeNet), graph NN | avg test AUC 0.789 replicating original | avg test AUC 0.822, i.e. **13.6% of remaining error removed**; best run 0.885 | 50 seeds. Published TrimNet result 0.860 is *above* PAI's average and below PAI's max. 2/50 runs got **zero** useful dendrites. Dendrite count per run ranged 0-12, mode 3. |
| HIST on CSI300 stock forecasting | Precision@1 0.610; chance = 0.5 | best 0.648 = **+35% above chance**; **avg +8.6%** over 10 seeds | Improvement measured against 0.5, not against the baseline — inflates the headline percentage. |
| EMNIST-Balanced, PyTorch MNIST example CNN (conv 32/64 → FC 9216/47) | first-cycle test acc 86.55% | final cycle 87.13% = **4.3% error reduction** | 7 runs; new cycle after 25 epochs without improvement. |
| mTAN on PhysioNet-2012 (parameter-matched) | Net 1.0 width | **Net 0.125 + 3 dendrites scores 2.7% better than Net 1** at <1/10 the original size | The one genuinely parameter-controlled experiment in the paper. |

**Parameter-matching, honestly stated.** Only the mTAN/PhysioNet experiment is
parameter-matched. The TrimNet, HIST and EMNIST accuracy results are **not**
parameter-matched — dendrites roughly double the parameter count there. The
authors explicitly frame mTAN as the answer to that objection: *"A question
that this raises, is whether these networks are only improving because the
total number of parameters is increasing."*

**Limitations the paper itself states:**
- *"dendrites will always eventually begin to overfit the training data, so
  early stopping must be performed once this point is reached."*
- Widths Net 1 and Net 2 in the mTAN sweep got **no** dendrites at all:
  *"Dendrites were not included for Net 1 and Net 2 because these networks
  regularly overfit with the addition of the first dendrite."* — i.e. the
  method helps *under*-parameterised models and hurts over-parameterised ones.
- Training cost explodes: Net 0.125 ran 1,750 epochs vs 250 for the baseline
  (Table 2). Inference cost rises modestly.
- *"The experiments in this initial paper are confined to models that fit on a
  single GPU."*
- Timing in Table 2 came from an *older* implementation in which *"a smaller
  perforated model ended up slower than a larger traditional model"*; the fix
  is asserted, not shown.

**Limitations not stated but visible. [INFER]**
- Every headline is a max-over-seeds or an average against a self-run baseline,
  never against the published number for the architecture.
- No error bars on the compression claim.
- The HIST "+35%" normalisation against chance is not comparable to the other
  percentages.

#### P2 — Further Experiments (the hackathon paper)

**Citation.** R. Brenner, E. Davis, R. Chaudhari, R. Morse, J. Chen, X. Liu,
Z. You, L. Itti, "Exploring the Performance of Perforated Backpropagation
through Further Experiments," arXiv:2506.00356. 10 pages.
<https://arxiv.org/pdf/2506.00356>

**Status. [PREPRINT] — arXiv only, not peer reviewed.** The paper states its
own weakest point up front: *"Due to the nature of the hackathon, results will
not contain error bars representing repeated runs of the same experiments."*
**Every number in P2 is a single run.**

**Claims:**

| Experiment | Result | Conditions |
| --- | --- | --- |
| BERT family on SNLI (570K) | PB improved test accuracy "across a range of model sizes, from 3.9M BERT-tiny DSN to 124M RoBERTa-base" | No per-model numbers in text; figure only. |
| BERT on IMDB (25K train) | BERT-tiny DSN width reduced 87.5% (**88.7% fewer params**) matched full-width BERT-tiny accuracy | *"Due to the small dataset size, the BERT models tend to quickly overfit, and so the addition of PB did not benefit larger BERT variants."* Also: *"Parameter increases are minimal in this case because most of the parameters of the DSN model are in the embedding layer, which did not have Dendrites added."* |
| ProteinBERT / AMP-BERT (3,556 peptides) | 30→12 layers, width 1024→480, + dendrites ⇒ comparable accuracy at **21% of original params** | Single run. |
| MobileNetV3-Small on CIFAR-10 | 81.99% → 83.05% (**6% error reduction**) | Single run, dendrites added to the base model (params increase). |
| MobileNetV3-Small ×0.5 width + dendrites | 2.54M → 0.82M base, 80.8%; +dendrites ⇒ **82.25% at 1.66M (35% fewer params than original, +0.26pp accuracy)** | Single run. This *is* parameter-accounted. |
| Deployment (GCP) | PB+DSN-0.125 (496K) vs BERT-tiny (4.38M): **158× tokens/s on CPU**, **38× cheaper per B tokens on T4** | Throughput/cost, not accuracy. |

**Where the headline marketing numbers come from. [INFER]** perforatedai.com's
*"up to 90% compression"* maps to P2's 88.7%-fewer-params IMDB result, and
*"up to 16% increased accuracy"* to the SNLI figure. The site's *"up to 70%
accuracy improvements"* does **not** correspond to anything in either paper I
read; treat it as unsourced marketing.

### 2.2 Papers the project cites as background — read as summaries only

`PerforatedAI/papers/README.md` curates these. **I read the repo's summaries
and the papers' public abstracts/landing pages, not the full texts.** Two are
paywalled (IEEE, ScienceDirect) and I did not obtain them. Flagged explicitly
so nobody treats my one-line gloss as a reading of the paper.

| Paper | Venue / status | Relevance to PAI |
| --- | --- | --- |
| Fahlman & Lebiere, "The Cascade-Correlation Learning Architecture," NeurIPS 1989 | **[PEER]**, open | The direct ancestor. PAI's dendrite learning rule is this correlation rule, applied to a hidden neuron's backpropagated error instead of the network output error, and PAI's dendrite→dendrite cascade is this paper's cascading. Original was non-gradient and single-layer. |
| Ritter & Urcid, "Morphological Perceptrons with Dendritic Structure," IEEE FUZZ 2003 | **[PEER]**, **paywalled — not read** | Dendrites as hypercubes; no gradient descent; single layer. |
| Sossa & Guevara, "Efficient Training for Dendrite Morphological Neural Networks," Neurocomputing 2014 | **[PEER]**, **paywalled — not read** | Same lineage. |
| Todo et al., "Dendritic Neuron Model...," IEEE TNNLS 2018 | **[PEER]**, **paywalled — not read** | Non-backprop metaheuristic training. |
| Chavlis & Poirazi, "Drawing Inspiration from Biological Dendrites...," Curr Opin Neurobiol 2021 | **[PEER]**, **paywalled — not read** | Review. |
| Meir et al., "Learning on Tree Architectures Outperforms a Convolutional Feedforward Network," Sci Rep 13, 2023 | **[PEER]**, open | First dendrites-on-CNN with gradient descent; parameter-efficient on CIFAR-10 vs LeNet. **Single neuron layer only.** |
| Chavlis & Poirazi, "Dendrites Endow Artificial Neural Networks with Accurate, Robust and Parameter-Efficient Learning," Nature Communications 16, 2025 | **[PEER]**, open | One hidden dendritic layer + plain output layer; dendritic receptive fields; parameter efficiency vs MLP. **Not multi-layer.** |

**[VENDOR]** PAI's own differentiator claim, from that table, is the last
column: it is the only listed method that puts dendrites on **multiple neuron
layers** of an existing network while keeping the original architecture.

**[INFER] The honest state of the evidence.** The multiplicative/nonlinear
dendritic-integration neuroscience (Major/Larkum/Schiller 2013; Branco &
Häusser 2010) is well-established peer-reviewed work, and PAI cites it
correctly as *motivation*. The peer-reviewed ML results on dendrites
(Sci Rep 2023, Nat Commun 2025) support "dendritic structure is
parameter-efficient" — but for **single-hidden-layer, purpose-built**
architectures, not for bolting dendrites onto arbitrary deep nets. The
specific claim that matters to us — *Perforated Backpropagation improves an
existing PyTorch model* — rests **entirely on two non-peer-reviewed
vendor-authored preprints**, one of which has no error bars at all.

---

## 3. Canonical integration recipe

Source: `PerforatedAI/api/README.md` (main guide), `api/customization.md`
(options), and the vendored `PAI Skills/skills/perforatedai/SKILL.md`.
Steps marked **REQUIRED** are the ones the docs state as mandatory; **OPTIONAL**
are presented as alternatives or tuning.

### Step 0 — install **REQUIRED**
**[VENDOR]** `api/README.md`: `pip install perforatedai perforatedbp`.
Without `perforatedbp` you get the degraded no-correlation build (see §1.3).

### Step 1 — imports **REQUIRED**
```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

### Step 2 — configure **before** `perforate_model` **REQUIRED (output_dimensions), OPTIONAL (rest)**

```python
GPA.pc.set_output_dimensions([-1, 0, -1, -1])   # exactly one 0 = the neuron/channel axis
GPA.pc.set_module_names_to_perforate(["Conv2d", "Linear"])
GPA.pc.set_testing_dendrite_capacity(True)      # default True; smoke test
```
**[SRC]** `output_dimensions` **must contain exactly one `0`** or the library
`sys.exit(-1)`s: `modules_perforatedai.py:381-384`,
`"5 Need exactly one 0 in the input dimensions"`.
**[SRC]** `Linear` and `Conv1d` outputs are auto-truncated to `[-1,0]` /
`[-1,0,-1]` (`modules_perforatedai.py:405-425`); everything else needs
`module.set_this_output_dimensions([...])` **after** `perforate_model`.
**[VENDOR]** `api/customization.md` §4: *"This is based on the output of the
layer, not the input."*

**[SRC]** Module-ID filters must be dot-prefixed and bracket-free — validated,
raises `ValueError`: `globals_perforatedai.py`, `_validate_module_id`:
> `"Module ID '{module_id}' must start with '.' - model.module should be '.module'"`

### Step 3 — which modules to wrap / replace / leave **REQUIRED choice**

Three registries, checked in this order inside `convert_module`
(`utils_perforatedai.py`):
1. `modules_to_replace` / `replacement_modules` — swap the class *before*
   conversion (ResNet pattern). **OPTIONAL.**
2. `module_ids_to_track` / `module_names_to_track` / `modules_to_track` —
   wrap in `TrackedNeuronModule`: accounted for, **no dendrites**.
3. `module_ids_to_perforate` / `module_names_to_perforate` /
   `modules_to_perforate` — wrap in `PAINeuronModule`: gets dendrites.

**[SRC] Default perforate list** (`globals_perforatedai.py:824-830`):
```python
["PAISequential", "Conv1d", "Conv2d", "Conv3d", "Linear"]
```
**[VENDOR]** `api/README.md` / `customization.md` §2.1: everything except
nonlinearities should be *inside* a converted module, and normalization layers
must be grouped with the preceding layer via `GPA.pc.PAISequential([conv, bn])`.
**[SRC]** If a BatchNorm / InstanceNorm / LayerNorm is itself in the perforate
list the converter prints *"You have an unwrapped normalization layer, this is
not recommended"* and drops into `pdb.set_trace()` (`utils_perforatedai.py`,
`convert_module`).
**[SRC]** After conversion, `convert_network` enumerates every parameter and
`pdb.set_trace()`s if any is neither `wrapped` nor `tracked`
(`utils_perforatedai.py:797-843`). Silence it with
`GPA.pc.set_unwrapped_modules_confirmed(True)` — **[VENDOR]** not recommended.

### Step 4 — convert **REQUIRED**
```python
model = UPA.perforate_model(model, save_name="run", maximizing_score=True)
model = model.to(device)
```
**[VENDOR]** `api/README.md` §2.1: *"The call to initializePB should be done
directly after the model is initialized, before cuda and parallel calls."*
**[SRC]** `perforate_model` builds the tracker, calls `convert_network`, and
writes `{save_name}/{save_name}_config.json` unless
`testing_dendrite_capacity` is on (`utils_perforatedai.py:46-148`).

### Step 5 — optimizer and scheduler **REQUIRED (one of two forms)**

Preferred (**[VENDOR]** "we recommend"):
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimArgs = {'params': model.parameters(), 'lr': lr}
schedArgs = {'mode': 'max', 'patience': 5}   # must be < n_epochs_to_switch
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```
Fallback:
```python
GPA.pai_tracker.set_optimizer_instance(optimizer)
```
**[VENDOR]** `api/README.md` §3, verbatim, twice:
> "NOTE: If you use set_optimizer_instance, PAI will NOT handle the scheduler -
> you must manage scheduler.step() calls yourself."

and

> "If you use setup_optimizer with schedArgs passed in AND return a scheduler
> from the function, the scheduler will get stepped inside our code so get rid
> of your scheduler.step() calls."

**[SRC]** `find_best_lr` only functions on the `setup_optimizer` path —
`configuration_descriptions.json`: *"This can only be used when calling
setup_optimizer and not set_optimizer_instance."*
**[VENDOR]** `api/README.md` §3: *"weight decay can sometimes cause problems
with dendrite learning. If you currently have weight decay and are not happy
with the results, try without it."* **[SRC]** `setup_optimizer` emits a
warning whenever `weight_decay` is in `opt_args` and
`weight_decay_accepted` is False (`tracker_perforatedai.py:1758-1763`):
> `"For PAI training it is recommended to not use weight decay in your optimizer"`

### Step 6 — training loop **REQUIRED**
```python
epoch = -1
while True:
    epoch += 1
    train(...)
    val = validate(...)
    GPA.pai_tracker.add_extra_score(train_score, 'Train')          # OPTIONAL
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val, model)
    model.to(device)                                               # REQUIRED
    if training_complete:
        break
    elif restructured:
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```
**[VENDOR]** `api/README.md` §5: *"you should change your training loop to be a
while(True) loop or set epochs to be a very high number."*
**[VENDOR]** §4.1: *"if it does get restructured, reset the optimizer with the
same block of code you use initially"*, and *"the following line should be
replaced with whatever is being used to set up the gpu settings, including
DataParallel"* — the `.to(device)` is required because
`add_validation_score` may return a **different object** loaded from disk.
**[SRC]** `check_input_problems` hard-fails (pdb + `sys.exit(-1)`) if you pass
a `DataParallel` or anything with a `.module` attribute
(`tracker_perforatedai.py:126-165`).
**[SRC]** On a switch, `clear_optimizer_and_scheduler()` is called
(`tracker_perforatedai.py:3560`) — **the old optimizer is invalid after
`restructured=True`; reusing it is a silent no-op-ish bug.**

### Step 7 — checkpoints **REQUIRED to resume, OPTIONAL otherwise**
**[SRC/VENDOR]** files written per cycle (`api/output.md`): `latest`,
`best_model`, `beforeSwitch_x`, `switch_x`, `best_model_beforeSwitch_x`,
`final_clean_pai`.
```python
model = UPA.load_system(model, save_name, 'latest', True)   # resume mid-run
model = UPA.load_system(model, save_name, 'best_model', True)
```
**[VENDOR]** `api/customization.md` §7: *"This function should be called after
perforate_model and set_this_output_dimensions, but before setup_optimizer."*

### Step 8 — inference-time loading **OPTIONAL**
**[VENDOR]** `api/customization.md` §7, *Loading _pai Models*:
```python
from perforatedai import network_perforatedai as NPA
model = YourModelClass()
model = NPA.load_pai_model(model, 'PAI/best_model_pai.pt')   # dendrites frozen, no tracker
```
Requires `GPA.pc.set_pai_saves(True)` during training (default **False**,
`globals_perforatedai.py:691`). Alternatively, after `load_system`:
```python
from perforatedai import blockwise_perforatedai as BPA
from perforatedai import clean_perforatedai as CPA
model = CPA.refresh_net(BPA.blockwise_network(model))
```
**[VENDOR]** *"Note: all other GPA settings should still be set first"*, and
the `_pai` models *"can be run with open source code from this API without
requiring a license."*

---

## 4. Hyperparameters that matter

All defaults **[SRC]** from `PerforatedAI/perforatedai/globals_perforatedai.py`
(cross-checked against the reconstructed 3.2.8 source) and, for the `pbp_*`
rows, from `import perforatedbp.globals_pbp` in this project's venv.
"Guidance" is **[VENDOR]** from `configuration_descriptions.json`,
`api/customization.md`, or the skill docs.

### 4.1 Switch strategy

| Name | Default | Controls | Vendor guidance |
| --- | --- | --- | --- |
| `switch_mode` | `DOING_HISTORY` (=1) | Policy. `DOING_SWITCH_EVERY_TIME`=0, `DOING_HISTORY`=1, `DOING_FIXED_SWITCH`=2, `DOING_NO_SWITCH`=3 | "DOING_HISTORY switches when validation has not improved over a patience window... DOING_SWITCH_EVERY_TIME switches every epoch (implementation debugging), and DOING_NO_SWITCH never adds dendrites." **[SRC]** `DOING_NO_SWITCH` also never returns `training_complete` — infinite loop. |
| `n_epochs_to_switch` | `10` | History-mode patience, in `add_validation_score` calls | **[SRC]** comment: *"Epochs to try before deciding to load previous best and add dendrites. Be sure this is higher than scheduler patience."* |
| `history_lookback` | `1` | Averaging window for `running_accuracy` | **[SRC]** `update_running_accuracy`: `running = running*(1 - 1/lookback) + acc/lookback`. **At the default 1 this is not an average at all — `running_accuracy == acc`.** |
| `initial_history_after_switches` | `0` | Grace epochs after a switch before history checks and before a score can be a new best | "After dendrites are added there is often an initial drop in scores as the model adjusts; this setting ensures that history-based switching logic does not react prematurely." |
| `fixed_switch_num` | `250` | Fixed-mode interval | Only used when `switch_mode == DOING_FIXED_SWITCH`. |
| `first_fixed_switch_num` | `1` (3.2.6) / `-1` (**3.2.8, installed**) | Delay for the very first fixed switch | "ignored unless it is > fixed_switch_num; its primary use is to ensure the pre-dendrite model has sufficient training before the first structural change." |

**[SRC] Exact history trigger** (`tracker_perforatedai.py:2028-2044`):
```python
mode == "n"
and (num_epochs_run - epoch_last_improved >= n_epochs_to_switch)
and (this_count >= initial_history_after_switches + n_epochs_to_switch)
```
so the minimum n-phase length is `n_epochs_to_switch` epochs.

### 4.2 Acceptance thresholds

| Name | Default | Controls | Vendor guidance |
| --- | --- | --- | --- |
| `improvement_threshold` | `[0.001, 0.0001, 0.0]` | **Relative** margin for a new best | "For a new best score to be defined it must beat the previous best score by at least this much." |
| `improvement_threshold_raw` | `1e-5` | **Absolute** margin, ANDed with the above | "Absolute improvement floor to avoid treating tiny metric deltas as meaningful." |
| `maximizing_score` | `True` | Sign of comparison | Set `False` for a loss. Also settable via `perforate_model(maximizing_score=...)`. |
| `reset_best_score_on_switch` | `True` | Zeroes `current_best_validation_score` on return to n mode | **[SRC]** `tracker_perforatedai.py:2276-2278`. |
| `max_dendrite_tries` | `2` | Re-rolls of a failed dendrite before giving up | "If this is >= 2 then when an added set of dendrites does not improve validation scores it will be deleted, and a new dendrite cycle will begin with a different random initialization." |
| `max_dendrites` | `100` | Hard cap | "Hard upper bound on dendrites added." |
| `retain_all_dendrites` | `False` | Keep dendrites that did not help | "Useful for debugging to view all calculated metrics over multiple cycles." **[SRC]** also forces `current_n_set_global_best = True` every cycle, so the no-improvement/early-stop path is never taken (`tracker_perforatedai.py:2190-2193`). |
| `global_candidates` | `1` | Candidate dendrites raced per cycle | **[SRC]** >1 is **not implemented**: `modules_perforatedai.py:1423-1438` prints *"This was a flag that will be needed if using multiple candidates. It's not set up yet but nice work finding it."* and drops into `pdb`. **Leave at 1.** |

**[SRC] The threshold list is a per-dendrite-count schedule, not a list-valued
setting.** `add_pai_config_var_functions`'s value-getter
(`globals_perforatedai.py`) does:
```python
if type(getattr(self, private_name)) is list:
    return getattr(self, private_name)[
        min(len(...) - 1, pai_tracker.member_vars["num_dendrites_added"])
    ]
```
So `[0.001, 0.0001, 0.0]` means *0.1% required before the first dendrite,
0.01% after one dendrite, 0% after two or more*. **[INFER]** Any scalar config
registered without `list_type=True` can be given a schedule this way
(`n_epochs_to_switch`, `max_dendrite_tries`, ...).

**[SRC] The comparison** (`tracker_perforatedai.py:218-247`,
`score_beats_current_best`) — note it is multiplicative **and** additive:
```python
maximizing and (new * (1.0 - thr) > old) and (new - thr_raw > old)
```
**[INFER]** The multiplicative part is scale-invariant, but
`improvement_threshold_raw = 1e-5` is **not**: passing accuracy as `0.95`
versus `95.0` changes what that floor means by 100×.

### 4.3 Dendrite construction

| Name | Default | Controls | Guidance |
| --- | --- | --- | --- |
| `pai_forward_function` | `torch.sigmoid` | Dendrite activation `f` | Options `relu`, `tanh`, `sigmoid`. |
| `candidate_weight_initialization_multiplier` | `0.01` | `param = randn(shape) * mult * scale` (`modules_perforatedai.py:1029-1068`) | "Can be adjusted if dendrites have too high or too small of an impact when added." Analyze skill: lower it (0.01 not 0.1) if scores spike after a switch. |
| `candidate_weight_init_by_main` | `False` | If True, `scale = mean(abs(main weights))` | "Scales initialization weights by magnitude of parent weights." |
| `d_type` | `torch.float` | dtype of dendrite / to-top weights | — |
| `learn_dendrites_live` | `False` | **[SRC]** "Not used in open source implementation, leave as default." | — |

### 4.4 Learning rate across switches

| Name | Default | Controls | Guidance |
| --- | --- | --- | --- |
| `find_best_lr` | `True` | Sweeps previously-seen LRs when dendrites are added | "Sometimes it's best to go back to initial LR, but often its best to start at a lower LR." **Requires `setup_optimizer`**; with step-based schedulers it *"will add large timing addition to the run as it iterates over every candidate learning rate."* |
| `dont_give_up_unless_learning_rate_lowered` | `True` | Blocks *all* switch triggers until the LR has been stepped at least once | "Should not be set to True without a scheduler." **[SRC]** the guard also requires `scheduler is not None`, so with `set_optimizer_instance` (scheduler `None`) it is inert (`tracker_perforatedai.py:1995-2010`). |
| `param_vals_setting` | `PARAM_VALS_BY_UPDATE_EPOCH` (=1) | Whether scheduler "epoch" counts total epochs or epochs-since-switch | `PARAM_VALS_BY_TOTAL_EPOCH`=0, `..._BY_UPDATE_EPOCH`=1 (reset each switch), `..._BY_NEURON_EPOCH_START`=2 (not in OSS). |

**[SRC]** The LR search procedure is documented in the docstring of
`process_scheduler_update` (`tracker_perforatedai.py:486-500`):
> "1. Start at default rate 2. Learn at that rate until scheduler increments
> twice 3. Save that version, start dendrites at LR current increment - 1
> 4. Repeat 2 and 3 until version has worse final score at set LR 5. Load
> previous model with best accuracy at that LR as initial rate"

### 4.5 Weight decay

**[SRC]** There is no dendrite-specific weight-decay knob. The only related
settings are the warning gate `weight_decay_accepted` (default `False`) and
the fact that `setup_optimizer` preserves per-group `weight_decay` when
filtering params in p mode (`tracker_perforatedai.py:1796-1806`).
**[VENDOR]** The guidance is binary: *don't use weight decay* with PAI.

### 4.6 Perforated-Backprop-only settings (from `perforatedbp.globals_pbp`, installed 3.2.7)

| Name | Default | Notes |
| --- | --- | --- |
| `p_epochs_to_switch` | `2` | Patience, in epochs, for ending a dendrite phase. |
| `pai_improvement_threshold` | `0.1` | **[VENDOR]** `customization.md` §9.1: *"if at least one Dendrite in the entire network has improved its correlation score by at least 10% AND at least 1e-5 during that epoch, then the patience counter resets"*. Note the doc says 1e-5 while the installed default is 1e-4. |
| `pai_improvement_threshold_raw` | `1e-4` | See above. |
| `cap_at_n` | `False` | Cap p-phase length at the length of the first n phase. *"Recommended usage is to set this to True during experimentation."* |
| `initial_correlation_batches` | `100` | Batches used to prime correlation statistics. **[SRC]** forced to `1` when `testing_dendrite_capacity` is on. |
| `no_extra_n_modes` | `False` under PB (`True` without it) | **[SRC]** flipped automatically at import (`globals_perforatedai.py:1336`). |
| `normalized_covariance` | `True` | Correlation normalisation. |
| `correlations_by_mean` | `True` | — |
| `candidate_grad_clipping` | `0.0` | Off. |
| `dendrite_learn_mode` / `dendrite_update_mode` | `True` / `True` | — |
| `pai_email` / `pai_token` | `""` | Licensing. `PAIEMAIL`/`PAITOKEN`/`PAIPASSWORD` env vars documented in `customization.md` §9. |

### 4.7 Run-hygiene settings worth knowing

| Name | Default | Note |
| --- | --- | --- |
| `testing_dendrite_capacity` | **`True`** | **[SRC]** `tracker_perforatedai.py:2450-2459`: overrides `switch_mode -> DOING_SWITCH_EVERY_TIME`, `retain_all_dendrites -> True`, `max_dendrite_tries -> 1000`, `max_dendrites -> 1000`, `initial_correlation_batches -> 1`, then stops after 3 dendrites. **Leaving this at the default silently invalidates a "real" experiment.** |
| `test_saves` | `True` | Writes `latest` every epoch. |
| `pai_saves` | `False` | Needed for `_pai` inference checkpoints. |
| `using_safe_tensors` | `True` | Flip to `False` on shared-tensor save errors. |
| `strict_loading` | `True` | — |
| `save_old_graph_scores` | `True` | Keeps pre-early-stop history in the graphs. |
| `checked_skipped_modules` | `False` | *"Reccomended to not use this setting."* |
| `unwrapped_modules_confirmed` | `False` | Silences the unwrapped-params pdb. |
| `module_names_to_not_save` | `[".base_model"]` | Duplicate-pointer workaround. |

---

## 5. Which layers to add dendrites to

### 5.1 Vendor guidance

**[SRC]** Default targets: `["PAISequential", "Conv1d", "Conv2d", "Conv3d", "Linear"]`.

**[VENDOR]** `api/customization.md` §2.1 — group, don't shred:
> "you want to make sure everything other than nonlinearities are contained
> within PAI modules so that each dendrite block performs the same processing
> as the associated neuron blocks... Performance is often better when these
> modules are grouped as a single PAI module as opposed to PAI-ifying each
> module within them."

**[VENDOR]** §2.1, on depth of placement:
> "Wrapping everything usually generates the best results, but often the deeper
> layers of the network and encoding modules do not provide significant
> benefits."

**[VENDOR]** Analyze skill, `perforatedai-analyze/SKILL.md`, on efficiency:
> "**Later layers are more efficient:** They have already-processed features,
> so dendrites there are more impactful per parameter"
> ... "**Perforate only the last 1-3 layers** instead of the whole network"

and on data-driven pruning of placements:
> "Higher scores (> 0.02) = dendrites aligned well with learning signal = good
> dendrite placement. Lower scores (< 0.01) = dendrites poorly aligned =
> wasting parameters"

**[VENDOR]** `api/output.md` sets the hard floor:
> "For PB training the PB scores of every module should not be significantly
> lower than 0.001, if they're lower than this it means correlation isn't being
> learned."

### 5.2 Per-layer-type

- **BatchNorm / LayerNorm / InstanceNorm — never wrap alone.** **[VENDOR]**
  §2.1: *"all normalization layers should be contained in blocks. This always
  improves performance so it is checked for in the initialization function."*
  Fix with `GPA.pc.PAISequential([conv, bn])`. **[SRC]** putting a norm class
  in the perforate list triggers the warning + `pdb.set_trace()` in
  `convert_module`.
- **Classifier / FC layers — yes, and PAI measures the effect separately.**
  **[PREPRINT]** P1 Figure 6 ablates "PB Only Head" (dendrites on the output
  layer only) against "PB Only Backbone". Both are reported; the full
  configuration is the strongest. **[VENDOR]** the transformer preset in the
  skill actually *excludes* the final projection:
  `GPA.pc.set_module_ids_to_track([".output_projection"])  # Skip final layer`.
- **Pointwise (1×1) convs — supported, standard Conv2d path.** No special
  guidance in the docs. **[INFER]** they are the natural target when the thing
  removed by narrowing is cross-channel mixing.
- **Depthwise convs — no guidance anywhere.** **[SRC/INFER]** They *are*
  wrapped by the default `Conv2d` name filter, and the dendrite inherits
  `groups`, so a depthwise dendrite is a per-channel nonlinear temporal filter
  with no channel mixing. The vendor never discusses this case; **[PREPRINT]**
  P2's MobileNetV3 result is the closest evidence that depthwise-separable
  stacks benefit overall, but it does not isolate depthwise vs pointwise.
- **Embeddings — excluded in practice.** **[PREPRINT]** P2 §3.1:
  *"most of the parameters of the DSN model are in the embedding layer, which
  did not have Dendrites added."*
- **Very narrow layers (2-16 channels) — no guidance exists.** See §8.

### 5.3 Modules that must NOT be wrapped

**[VENDOR]** `api/customization.md` §8, *Model doesn't Seem to Learn at All*:
> "Make sure you didn't wrap something that requires specific output values for
> future math down the line. Adding the Dendrite output to these values will
> mess up that math. For example, If a module ends with a Softmax layer going
> into NLL loss, you need to make sure the Softmax layer is not being wrapped
> because the output of Softmax is supposed to be probabilities."

Also:
- **Modules whose forward returns a tuple / takes multiple tensors** — need a
  processor class first (`customization.md` §2.2); otherwise
  `PAINeuronModule.forward` prints *"The output of the above module ... is a
  tuple when it must be a single tensor"* and `pdb`s
  (`modules_perforatedai.py:846-858`).
- **Modules called more than once in one forward.** **[VENDOR]** §8:
  *"If you have a layer that gets called more than once in the forward that has
  been seen to cause problems. See if you can make a second copy."*
  **[SRC]** related: `weight_tying_experimental` exists precisely for this and
  is labelled experimental.
- **Frozen / unused modules** — `set_mode("p")` fails, module is silently
  removed from `neuron_module_vector` (`tracker_perforatedai.py:2216-2228`).
- **An already-perforated model.** **[VENDOR]** `debugging.md`:
  *"This means you are trying to perforate a model that has already been
  perforated. This should never be done."*

---

## 6. Known pitfalls and anti-patterns

### 6.1 Silent no-ops (worst class — the run "works" and means nothing)

1. **`testing_dendrite_capacity` left at its `True` default.** **[SRC]** It
   rewrites five settings and terminates after 3 dendrites, returning
   `(net, False, True)` from `add_validation_score`. Any accuracy number from
   such a run is meaningless as an experiment.
2. **`perforatedbp` not importable.** **[SRC]** Prints
   *"Building dendrites without Perforated Backpropagation"* at import and
   every candidate code path is skipped. Dendrites become zero-gain random
   copies trained by plain GD. **Check the import banner in the log.**
3. **`switch_mode = DOING_NO_SWITCH`.** **[SRC]** never switches *and* never
   sets `training_complete` — a `while True` loop runs forever.
4. **Optimizer not rebuilt on `restructured=True`.** **[SRC]**
   `clear_optimizer_and_scheduler()` is called inside `add_validation_score`;
   a stale optimizer holds parameters of a model object that was replaced.
5. **`retain_all_dendrites=True`.** **[SRC]** forces
   `current_n_set_global_best = True` each cycle, so the "did this dendrite
   help?" branch never fires and `max_dendrite_tries` never engages.
6. **Modules never wrapped.** **[VENDOR]** `api/output.md`: *"When you first
   initialize the network there should be a warning for modules that exist in
   the model that were not wrapped."* Suppressing it with
   `unwrapped_modules_confirmed(True)` hides real mistakes.

### 6.2 Things that break checkpointing

- **Shared/duplicate tensors + safetensors.** **[VENDOR]** `debugging.md`:
  *"Some tensors share memory..."* → set `using_safe_tensors(False)` or add the
  offending path to `module_names_to_not_save`.
- **`KeyError: 'moduleName.mainModule.numCycles'`** **[VENDOR]** — either
  `perforate_model` was called on an already-PAI model, or the
  perforate/track lists differ between training and inference scripts.
- **`load_system` called in the wrong place.** **[VENDOR]** it must be after
  `perforate_model` and `set_this_output_dimensions` but **before**
  `setup_optimizer`.
- **[SRC]** `PAIDendriteModule.__getstate__` strips PB-bound methods before
  pickling (`modules_perforatedai.py:1159-1190`) — hand-rolled `torch.save`
  of a PAI model outside `UPA.save_system` is not supported.

### 6.3 AMP / GradScaler

**[VENDOR]** `skills/perforatedai-complex-methods/SKILL.md`, verbatim symptom
and fix:
> `AssertionError: No inf checks were recorded for this optimizer.`
```python
optimizer.zero_grad(set_to_none=True)
if scaler is not None and GPA.pai_tracker.member_vars['mode'] != 'p':
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
else:
    loss.backward(); optimizer.step()
```
Also: *"Rebuild the scaler too if using AMP — the old scaler has stale state
tied to the previous optimizer."*
**[VENDOR]** `debugging.md`, *Loss scaling* — timm's `ApexScaler`/`NativeScaler`
cause `RuntimeError "has changed the type of value"` and are listed under
**"Errors that are currently not fixable"**: *"you just have to turn off loss
scaling."*

### 6.4 Autograd / forward hazards

- **`Trying to backward through the graph a second time`** **[VENDOR]** — a
  tensor appears twice in the graph. Expected to bite when a module is reused.
- **In-place ops.** **[VENDOR]** *"This can happen anytime the forward is using
  += type functions. be sure to use `var = var + var2`"*, and
  `nn.Dropout(..., inplace=False)`. **[INFER]** `nn.ReLU(inplace=True)`
  immediately after a wrapped module is exactly this hazard.
- **EMA.** **[VENDOR]** *"EMA keeps a shadow copy... This has to be created
  after perforate_model... it must be reinitialized after each time restructured
  returns true."*
- **Centered RMSprop** — listed as *currently not fixable* (NaN).
- **Output-dimension mismatches** — `sys.exit(0)` from `filter_backward`
  unless `debugging_output_dimensions` is set to 1.

### 6.5 Dataloader / seed / evaluation interactions

- **[VENDOR]** `api/output.md`: during a dendrite phase *"the
  validation/training scores should be flatlined... There may be minor
  fluctuation if you are randomizing the order of your inputs but if there are
  any significant changes there is a problem."* **[INFER]** With shuffled
  loaders plus BatchNorm in `train()` mode, running stats keep moving during p
  phases even with weights frozen, which will produce non-flat validation.
  Freeze BN (eval mode) during p phases if you want the flatline diagnostic to
  be interpretable.
- **[VENDOR]** `customization.md` §8 *Good Science*: *"The validation scores are
  used to determine when to add dendrites, so one could argue they are even
  part of the training dataset... be sure to always have a final test dataset."*
- **[VENDOR]** `customization.md` §7/loading: after a switch the model object
  is reloaded from disk — **anything holding a stale reference** (EMA, a
  `self.model` attribute, a profiler, a pruning mask) is now wrong.

### 6.6 Small-model caveats the docs actually state

- **[PREPRINT]** P1: networks that already overfit get *worse* with the first
  dendrite (Net 1, Net 2 in the mTAN sweep).
- **[VENDOR]** `customization.md` §8 *Overfitting*: the prescribed fix is to
  shrink the base model further or add dropout until train < val, then perforate.
- **[VENDOR]** `customization.md` §9: *"Some larger models will continue showing
  score increases from random noise (even with learning rate 0) if these numbers
  are set too low."* **[INFER]** the converse is the risk here: with tiny `C`,
  the correlation estimate per output unit is computed over very few units, so
  noise dominates in the other direction.

### 6.7 How to tell from the CSVs that a run is degenerate

**[VENDOR]** `perforatedai-analyze/SKILL.md` + `api/output.md`, plus **[INFER]**
markers:

| Symptom | Meaning |
| --- | --- |
| `*noImprove_lr*` files present and/or `switch_epochs.csv` empty | **No dendrite was ever integrated.** The analyze skill calls this "a critical issue". |
| `param_counts.csv` flat across cycles | Same — nothing was structurally added. **[SRC]** a count is appended on every `change_learning_modes`. |
| `Best PBScores.csv` values `< 0.001` | **[VENDOR]** *"correlation isn't being learned"* — wrong wrapping or wrong processors. |
| PB scores `< 0.01`-`0.02` for a module | **[VENDOR]** that placement is "wasting parameters"; move it to `module_ids_to_track`. |
| Validation **not** flat during p-mode epochs | Modules not properly wrapped (or BN still updating). |
| Validation does **not** improve after each blue line | **[VENDOR]** *"After every blue vertical line, your scores should be getting better."* |
| `best_arch_scores.csv` shows dendrite-1 row ≤ dendrite-0 row | Dendrites did not help at all; check `maximizing_score` and the raw score scale first. |
| Run ended after exactly 3 dendrites with the "You may now set that to False" line | `testing_dendrite_capacity` was on. |
| Scores spike/crash right after a switch | **[VENDOR]** lower `candidate_weight_initialization_multiplier`; also check the optimizer/scheduler rebuild. |

**A worked degenerate example already in this repo.**
`outputs/compression-run/pai/candidates/w10_classifier/`:
```
w10_classifierswitch_epochs.csv   -> Switch Number,Switch Epoch
                                     0,33
w10_classifierparam_counts.csv    -> Switch Number,Param Count
                                     0,1246
                                     1,1246
w10_classifier_best_arch_scores.csv -> Param Counts,Max Valid Scores,Train
                                     1246,0.6426...,0.3314...
```
One switch fired, the parameter count never moved, and `best_arch_scores.csv`
has exactly one row. **[INFER]** That is the signature of a run that entered a
dendrite phase and ended before any dendrite was ever accepted into the
network — the accuracy number in it is a *baseline*, not a dendrite result.
(Its `Best PBScores.csv` does show `.fc` correlation ≈0.019, i.e. the
correlation learning itself was working; the run simply never integrated.)

---

## 7. How to read PAI's output artifacts

**[VENDOR]** `api/output.md`, in full paraphrase, plus **[SRC]** file names
from `tracker_perforatedai.py` / `utils_perforatedai.py`.

### Graphs (4 panels, regenerated every epoch, `{save_name}/{save_name}.png`)
1. **Scores** — validation + running validation + every `add_extra_score`
   series. **Red vertical line = switch into dendrite (p) training; blue
   vertical line = switch back to neuron (n) training.** With GD-only dendrites
   (no `perforatedbp`) only blue lines appear.
2. **Times** — per-epoch training time as dendrites accumulate.
3. **Learning rate** — includes the effect of the LR sweep.
4. **Dendrite correlation scores per module** — *"This graph is what can be used
   to determine if you should be wrapping your network differently."* Blank
   without `perforatedbp`.

### CSVs

**[SRC]** Exact emitted names, from `tracker_perforatedai.py:2589-3151`, and
confirmed against files on disk in `outputs/compression-run/pai/candidates/`.
Note `Best PBScores.csv` contains a **space**, and only `_best_arch_scores.csv`
has a leading underscore. `{extra}` is `""` for the live files and
`_beforeSwitch_N` / `before_final` / a timestamped `..._noImprove_lr_N` for the
snapshot copies.

| File | Header seen on disk | Contents |
| --- | --- | --- |
| `{name}{extra}_best_arch_scores.csv` | `Param Counts,Max Valid Scores,Train` | **The main result.** One row per architecture (dendrite count): parameter count, best validation score of that cycle, and each extra/test score at the epoch of that best validation score. |
| `{name}{extra}Best PBScores.csv` | `Epochs,Best ever for all nodes Layer .X,Best current for all nodes Layer .X` | Per-module dendrite correlation scores per epoch. Only written under `perforatedbp`. |
| `{name}{extra}Scores.csv` | `Epochs,Validation Scores,Validation Running Scores,<extras>` | Raw validation score, the `history_lookback` EMA, and every `add_extra_score` series. |
| `{name}{extra}Times.csv` | — | Per-epoch train/val timing, split by n and p phases. |
| `{name}{extra}learning_rate.csv` | — | LR per epoch. |
| `{name}{extra}param_counts.csv` | `Switch Number,Param Count` | One row per mode switch. `count_params` excludes `parent_module`. |
| `{name}{extra}switch_epochs.csv` | `Switch Number,Switch Epoch` | Epoch of each n↔p switch. |
| `{name}/array_dims.csv` | — | Written by `save_tracker_settings()` for the multi-GPU two-step setup. |

### Checkpoints
| Name | Meaning |
| --- | --- |
| `latest` | Most recent. **[VENDOR]** *"This is what you should use if anything crashes and you want to pick up where it left off."* |
| `best_model` | Best validation score so far, globally. |
| `beforeSwitch_x` / `switch_x` | Network immediately before / after switch `x`. |
| `best_model_beforeSwitch_x` | Best validation model of cycle `x`. |
| `final_clean_pai` | **[SRC]** written by `pai_save_system` when training completes — the deployable model. |
| `name_x_startSteps_y` | `x` = dendrites created, `y` = LR steps taken before this cycle. |
| `*_pai` | Scaffolding removed; runnable with the open-source package only. |

**Judging success:** compare rows of `best_arch_scores.csv`. A successful run
shows the dendrite-`k` row beating the dendrite-0 row on a **held-out test**
metric, and you should divide the gain by the parameter delta in the same file
before calling it an improvement — **[VENDOR]** the analyze skill's own
formula: `(accuracy_gain / parameter_increase) × 100`.

---

## 8. Direct applicability to SparkNet (C16 → C2, ~1.2K-3.6K params)

SparkNet at the study's widths (40 mel bins, 12 classes, `gate_channels=32`),
measured by instantiating the model:

| Width | Total params | `.fc` | `.gate_conv` | `.blocks.3.pointwise` | `.blocks.3.depthwise` |
| --- | --- | --- | --- | --- | --- |
| C12 | 3,584 | 396 | 416 | 144 | 348 |
| C8 | 2,508 | 396 | 288 | 64 | 232 |
| C4 | 1,624 | 396 | 160 | 16 | 116 |
| C2 | 1,254 | 396 | 96 | 4 | 58 |

### 8.1 What holds

- **The additive, zero-initialised combination is scale-free.** **[SRC]**
  Nothing in `PAINeuronModule.forward` depends on width. A dendrite added at C2
  is still exactly output-preserving at addition time. The mechanism will not
  break.
- **The parameter formula `D·P_M + D²·C` holds exactly** and is unusually
  favourable at narrow widths for the pointwise arm: +6 params at C2 for one
  dendrite on a 4-param conv. **[INFER]** this is the best parameter-efficiency
  story available in the whole study.
- **The vendor's own prescription is "shrink first, then perforate"** —
  **[VENDOR]** `perforatedai-analyze/SKILL.md`: *"❌ WRONG: Take a large model →
  perforate it → hope it gets smaller... ✅ CORRECT: Take a large model → reduce
  width/depth → perforate the smaller model"*. The SparkNet study is exactly
  that design, so it is aligned with the intended use case.
- **The "under-parameterised models benefit, over-parameterised ones overfit"
  finding** **[PREPRINT]** (P1 mTAN: dendrites helped Net 0.125/0.25/0.5, hurt
  Net 1/Net 2) predicts the gain should be *largest at the narrow end* — the
  study's stated hypothesis.
- **`.fc` and `.gate_conv` are "later layers"**, which is where **[VENDOR]** the
  analyze skill says dendrites are most parameter-efficient.
- **BatchNorm handling.** SparkNet's `TCSBlock` already keeps `pointwise` and
  `bn` as separate attributes; wrapping `.blocks.k.pointwise` alone leaves `bn`
  unwrapped-but-trackable, which is **[VENDOR]** explicitly the non-preferred
  grouping (*"all normalization layers should be contained in blocks"*). The
  vendor-preferred form would be a `PAISequential([pointwise, bn])`.
  **[INFER]** worth an arm, but it changes the module identity and so is not a
  drop-in for the current study.

### 8.2 What is likely to break or mislead at 1K-5K parameters

1. **`.fc` is not a placement, it is a 32% parameter increase.** **[SRC/INFER]**
   One `.fc` dendrite costs `396 + 12 = 408` params on a 1,254-param C2 model.
   Any accuracy gain is confounded with a one-third capacity increase. The
   `pointwise` arm at C2 costs 6 params — a 48× difference in budget. **Arms
   cannot be ranked against each other at C2**; each is only interpretable
   against the matched control. (The repo's own `sparknet_c2_paper.yaml`
   comment already says this; the source confirms the exact numbers.)
2. **`out_channels = 2` makes correlation estimation degenerate.** **[SRC]**
   `dendrites_to_top` has one scalar per output channel, and the candidate is
   selected by per-unit correlation with that unit's backpropagated error. At
   C2 there are **two** units. **[INFER]** the candidate-selection signal is
   effectively a 2-sample statistic; `max_dendrite_tries` re-rolls will look
   like pure seed noise. Expect PB correlation scores near the
   **[VENDOR]** 0.001 "this is noise" floor on the `pointwise` and `depthwise`
   arms at C2-C4, and check `{name}Best PBScores.csv` before believing any delta.
3. **A depthwise dendrite cannot mix channels.** **[SRC]** `groups` is
   deep-copied. **[INFER]** At C2, `blocks.3.depthwise` is 2 independent
   1×29 filters; its dendrite is 2 more, squashed through `tanh` and scaled by
   2 scalars. This is the least expressive placement in the study and the one
   with the weakest prior from the literature (no PAI guidance at all).
4. **`improvement_threshold_raw = 1e-5` and the threshold *schedule*.** **[SRC]**
   the schedule indexes by `num_dendrites_added`. With `max_dendrites: 1` (as in
   the current arm configs) only the **first** entry of
   `[0.005, 0.002, 0.001]` is ever used — the other two are dead. **[INFER]**
   At 0.005 relative on a ~0.93 accuracy, a switch requires ≈+0.47pp, which at
   12-class KWS with a few-thousand-sample val set is within seed noise. Expect
   many runs to trip `process_no_improvement` and end early — and note that
   `max_dendrite_tries: 3` means each such run burns 3 full p+n cycles first.
5. **`history_lookback: 8` is an EMA, not a mean.** **[SRC]**
   `running = running*(7/8) + acc/8`. It lags by ~8 epochs, and the switch
   decision reads the EMA, not the raw score. With
   `dendritic_schedule_epochs: 30`, the EMA is still warming up for the first
   quarter of a phase.
6. **Weight decay is on.** The arm config sets `weight_decay: 0.0001`.
   **[SRC/VENDOR]** `setup_optimizer` warns against exactly this, and
   `api/README.md` says to try without it. **[INFER]** On a 1.2K-param model,
   1e-4 decay pulling on a 4-param dendrite copy plus 2 to-top scalars that
   start at **zero** is a real bias toward never letting the dendrite grow.
   This is my strongest single suspicion for "dendrites do nothing at narrow
   widths".
7. **The zero-init to-top gain needs LR and time to escape zero.** **[SRC]**
   new row is `torch.zeros`. **[INFER]** with a cosine schedule that has
   already decayed, and `post_integration_lr_multiplier: 0.25`, the integrated
   dendrite may never acquire meaningful gain within the n phase — producing a
   run that is *architecturally* dendritic and *functionally* identical to the
   control.
8. **BatchNorm running stats during p phases.** The arm config sets
   `freeze_base_batchnorm_in_dendrite_phase: true`, which is the right call —
   **[VENDOR]** `api/output.md` requires validation to be flat during p phases
   for the diagnostic to mean anything, and updating BN stats would break that.
9. **Stale references after restructuring.** **[SRC]**
   `add_validation_score` can return a *different* model object loaded from
   disk. Anything in the KWS pipeline that cached the model (profilers, MAC
   counters, KD teacher wiring, EMA) must be re-bound on `restructured=True`.
10. **`in-place ReLU`.** `TCSBlock` uses `nn.ReLU(inplace=True)` directly after
    `bn(pointwise(depthwise(x)))`. **[VENDOR]** `debugging.md` lists in-place
    ops as a known autograd hazard with PAI's second backward pass. **[INFER]**
    Not currently crashing (so presumably fine here, since the ReLU consumes
    the wrapped module's output rather than being consumed by it), but it is
    the first thing to change if a second-backward error ever appears.
11. **`residual` branches.** Blocks 1-3 compute `y = bn(pw(dw(x))) + res_bn(res_conv(x))`
    outside any single module. **[VENDOR]** §2.1.2: *"If there is functioning
    done outside of modules just make a new module that performs those
    calculations within a forward function."* **[INFER]** the current
    module-ID placements sidestep this by wrapping leaves, at the cost of the
    vendor-preferred block-level grouping.
12. **Statistical power.** **[PREPRINT]** P1 needed **50 seeds** on TrimNet
    because *"of the high fluctuation of scores depending on the random seed"*,
    and 2/50 runs got zero dendrites. **[INFER]** 5 seeds per (arm, width) cell
    is likely underpowered for the effect sizes that a ~1.2K-param model can
    show; treat single-cell rankings as hypotheses, not results, and pool
    across widths where the design allows.

### 8.3 Recommended checks before trusting any SparkNet dendrite number

1. Confirm `Building dendrites with Perforated Backpropagation` appears in the
   run log (else the whole study measured GD copies).
2. Confirm `testing_dendrite_capacity: false` took effect (the arm configs set
   it, but verify the emitted `{save_name}_config.json`).
3. Read `{name}Best PBScores.csv` for every arm/width. Anything at or below 0.001 is
   **[VENDOR]** noise, and that cell should be reported as "no correlation
   learned", not as "dendrites did not help".
4. Read `{name}param_counts.csv` and `{name}switch_epochs.csv` to separate "dendrite was
   integrated and did not help" from "no dendrite was ever integrated".
5. Report every placement's accuracy delta **next to** its parameter delta from
   `best_arch_scores.csv`, and compare each arm only against the matched
   `control`, never against another arm at the same width.
6. Run one ablation with `weight_decay: 0.0` on the narrowest widths. It is the
   cheapest test of the highest-probability confound, and it is the vendor's own
   first recommendation when dendrites underperform.
