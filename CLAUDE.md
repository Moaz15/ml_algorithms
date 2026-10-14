# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal study repository with two purposes: learning machine learning algorithms **from first principles**, and serving as the user's **go-to reference for revising ML concepts and preparing for Senior MLE interviews at FAANG companies**. It is not a library or an application — it is a collection of per-topic folders, each pairing plain-text theory notes with from-scratch NumPy implementations. When adding or modifying code, favor readable, pedagogical implementations over performance or production robustness. When adding or modifying notes, favor content that builds durable intuition (the "why", not just the "what") and that doubles as interview prep — trade-offs, failure modes, and comparisons to related algorithms are as valuable as the derivation itself.

## Layout convention

Each algorithm/topic lives in its own top-level folder (e.g. `Linear_Regression/`, `Logistic_Regression/`, `Decision_Trees/`, `SVM/`, `Clustering/`, `Ensemble_Learning/`, `Dimensionality_Reduction/`). Within a topic, the recurring pattern is:

- `theory/` (or `.txt` files) — conceptual notes: mathematical derivations, assumptions, evaluation metrics, `roadmap.txt` (learning plan / TODO), `concept.txt`. Most of the repo is these notes.
- `interview_qa.txt` — interview-style Q&A for the topic, living alongside `concept.txt` in the same `theory/` folder (e.g. `Linear_Regression/SLR/interview_qa.txt`, `Decision_Trees/theory/interview_qa.txt`). See "Interview-prep content guidelines" below.
- `scratch_code/` or `code/` — the from-scratch implementation in `.py`.

Root-level `.txt` files (`bias_variance.txt`, `regularization_concept.txt`, `optimization_gradient_descent.txt`, etc.) are cross-cutting concept notes not tied to a single algorithm. These may also get a matching companion, e.g. `bias_variance_interview_qa.txt`, following the same convention since they live outside a topic folder.

When a topic has multiple variants (e.g. `Linear_Regression/` covers SLR and MLR), the split happens *within* `theory/` and `code/`, not at the top level — `theory/SLR/`, `theory/MLR/`, `code/SLR/`, `code/MLR/` — so `theory/` and `code/` stay the two entry points for every topic.

## Interview-prep content guidelines

When asked to add or extend an `interview_qa.txt` file:

- Format as `Q:` / `A:` pairs, one concept per pair.
- Keep answers interview-length: 3-8 sentences, crisp and structured, not a re-derivation of the full theory. Reference `concept.txt` / derivation files for the underlying math rather than duplicating it.
- Favor the kind of questions a FAANG panel asks a Senior MLE candidate: intuition checks ("why does X work"), failure modes ("what breaks if Y"), bias-variance/complexity trade-offs, and comparisons between related algorithms (e.g. "how does bagging differ from boosting").
- Where it aids revision, end an answer with a natural follow-up question an interviewer might ask next, to encourage drilling deeper.

## Theory-note writing style

Notes are plain text, ASCII only, wrapped at ~100 columns, no emojis. Beyond that, follow the
shape used in `regularization_concept.txt` sections 2.6 and 2.7 - it is the house style:

- **Lead with the answer.** The definition or conclusion is the first sentence, never the last. If
  a heading asks two things ("what it is, and why it works"), answer them in that order under CAPS
  labels: `WHAT IT IS`, `WHY IT WORKS`, `THE PROBLEM`.
- **Pointer bullets over paragraphs.** Anything enumerable - options being compared, when to reach
  for something, what it costs - becomes a named header line followed by `-->` bullets:

      MORE DATA - the only one that cuts variance without adding any bias
      --> use it whenever it is available and affordable
      --> it cannot rescue a model that is too simple - that is bias, not variance

  Give every item in a comparison the same bullet slots, so the reader can scan one slot down the
  list. Reserve running prose for a narrative that genuinely carries a hard idea.
- **Aligned two-column lists** for mechanism or hyperparameter enumerations: the setting on the
  left, a one-line consequence on the right, padded to a common column.
- **One metaphor per concept, reused.** The L1/L2 geometry is a stone, ripples and a fence, and
  every later sentence in that subsection reuses those three nouns. Do not stack metaphors.
- **Concrete numbers inline** - `(1.0, 0.0)`, `learning_rate = 0.05`, `round 300 of 2000` - wherever
  a number can replace an abstraction.
- **Close every subsection** with a `> Real-world:` block and a `> Likely follow-up:` block (the
  question an interviewer asks next, plus the answer).
- **Converting prose to this style must not grow the section.** Net-neutral or shorter, unless the
  rewrite is adding a concept that was missing.

## Implementation style

Scratch implementations follow scikit-learn's shape: a class with `__init__` hyperparameters, `fit_*` training methods (often multiple, e.g. `fit_ols` vs `fit_gradient_descent`, `fit_gd`), and `predict`. Fitted parameters are stored as instance attributes (`self.theta`, `self.bias`, `self.centroids`, `self.inertia_`). Numerical-stability guards are used deliberately (`np.clip` on sigmoid inputs, epsilon in log loss, `np.isclose` on OLS denominators) — preserve these when editing. Inputs are coerced with `np.asarray(..., dtype=float)`.

## Environment & commands

- Dependencies: `pip install -r requirements.txt` (numpy, scikit-learn, matplotlib, fastapi, uvicorn).
- `ML_Algo/` is a committed local virtualenv and is gitignored for its venv contents — do not edit files under it; it is not project source.
- There is no build system, test suite, or linter configured. Implementations are exercised by running their scripts directly, e.g. `python Clustering/KMeans/code/kmeans.py`.

### Serving example (Logistic Regression)

`Logistic_Regression/scratch_code/main.py` is a FastAPI app that loads trained artifacts (`theta.npy`, `bias.npy`, `scaler.pkl`) from its own directory and exposes `POST /predict`. Run it from that directory so relative artifact loading in `predict_driver.py` resolves:

```
cd Logistic_Regression/scratch_code
uvicorn main:app --reload
```

Note `main.py` resolves artifact paths via `BASE_DIR` (absolute), but `predict_driver.py` loads them relative to the CWD — run from the `scratch_code/` folder to avoid path errors.
