# Session context

Handoff notes so a fresh session can pick up without re-deriving anything. Covers **how** notes are
written here and **where** the work has got to.

---

## 1. Purpose

Personal study repo, two jobs: learn ML from first principles, and serve as the revision + interview
reference for **Senior MLE interviews at FAANG**. Notes should build durable intuition and be
speakable out loud in an interview. Not research-heavy — trade-offs, failure modes and comparisons
beat derivations.

---

## 2. File layout convention (established for Decision_Trees, apply to new topics)

Inside a topic's `theory/` folder:

| File | Role |
|---|---|
| `roadmap.txt` | **Section 1** — the subtopic checklist. The index for the whole topic. |
| one file per checklist group | **Sections 2, 3, 4, …** — the expansion of that group |

- Group `1.1` → section `2`, group `1.2` → section `3`, and so on. File is named after the group
  (`foundations.txt`, `splitting_criteria.txt`, `building_the_tree.txt`, …).
- Checklist bullets carry a tag linking to where they are written up:
  - `- [4.1] Finding the best split` — written, lives in section 4.1 of that group's file
  - `- [   ] Some subtopic` — still to write
- The group header points at its file: `1.3 Building the Tree  ->  building_the_tree.txt`

---

## 3. Writing style (this is the part to preserve)

### Per-file skeleton

```
N. SECTION TITLE IN CAPS
(expands group 1.x of roadmap.txt)

N.1 Subsection title
--------------------          <- dashes match title length exactly

<body>

> Real-world: <one concrete scenario, hanging indent 2>

> Likely follow-up: "<the question an interviewer asks next>"
  <a 2-4 line answer, indented 2>
```

Two blank lines between subsections. One blank line everywhere else.

### Body style

- **Labelled definition blocks** wherever something is defined. Label column aligned within the
  block, `  :  ` separator, **blank line between every entry** (the spacing matters — packed rows
  were explicitly rejected):

  ```
      Entropy    :  how uncertain you still are about the label of a random row in this node,
                    measured in bits.

      Formula    :  H  =  - SUM over classes i of  p_i * log2(p_i)
  ```

- **CAPS sub-headings** inside a subsection when it has real parts (`THE PROCEDURE`,
  `WHERE THE FORMULA COMES FROM`, `DEPTH IS THE DIAL`).
- **Numbered procedures** for algorithms, not prose describing them.
- **Worked examples with real arithmetic**, column-aligned (`=` and `+` at fixed columns).
- **Small tables** for comparisons.
- **ASCII diagrams** when the idea is geometric or structural.

### Hard rules

- Wrap at **100 columns**. No line longer.
- **No trailing whitespace.** No emojis. Use `-` not em-dashes.
- Voice: plain words, spoken-interview register, first person where natural. Build the concept from
  scratch — say *why a thing is needed* before defining it.
- **Cross-reference by section number** (`see 2.6`, `the shortcut in 4.3`) so the files interlock.
- **Scope discipline**: inside `Decision_Trees/`, single tree only. Bagging/RF/boosting belong in
  `Ensemble_Learning/`. A one-line pointer out is fine; a lesson is not.

### Numbers and diagrams are generated, never typed

Every table of numbers was produced by a throwaway Python build script that defines the actual
functions (entropy, Gini, sklearn fits) and formats the output. Diagrams with alignment were built
by writing characters at **explicit 1-indexed columns**, then verified by printing the column
positions — not eyeballed. Those build scripts lived in the session scratchpad and are **gone**;
this document is the spec, so reproduce the approach rather than hunting for them.

---

## 4. Progress

### `Decision_Trees/` — 4 of 8 groups written

| Group | File | Status |
|---|---|---|
| 1.1 Foundations | `foundations.txt` | done (7 subsections) |
| 1.2 Splitting Criteria | `splitting_criteria.txt` | done (6) |
| 1.3 Building the Tree | `building_the_tree.txt` | done (6) |
| 1.4 Regularization and Pruning | `regularization_and_pruning.txt` | done (5) |
| 1.5 Failure Modes | — | **next**, 6 bullets |
| 1.6 Feature Importance and Interpretability | — | 5 bullets |
| 1.7 Variants and Comparisons | — | 5 bullets |
| 1.8 Interview Gotchas | — | 6 bullets |

Deleted along the way (all recoverable from git): `concept.txt`, `entropy.txt`,
`gini_impurity.txt`. Their content was folded into the new files.

**Open item:** `Decision_Trees/theory/regularization.txt` (old format, 9.7 KB) is now fully
superseded by `regularization_and_pruning.txt` — everything worth keeping was folded in. It should
be `git rm`'d; was left in place pending confirmation.

`Decision_Trees/theory/feature_importance.txt` (8.3 KB, old format) still holds material to fold
into group 1.6 when that gets written.

### `Decision_Trees/code/practice.ipynb`

12 coding interview questions, each a markdown prompt + worked answer, verified to run in order at
build time. Q1-Q6 are the from-scratch track (impurity → information gain → best-split search →
full recursive classifier → sklearn check → regression tree). Q7-Q12 are practical (overfitting
curve, cost-complexity pruning, MDI-vs-permutation bias demo, rule extraction from `tree_`,
instability demo, extrapolation failure). Each markdown cell ends with **"What is being checked"**.

The user has since added their own practice cells (36 cells now vs the 26 built). **Do not rebuild
this notebook from a script** — it would wipe their work. Edit cells in place, or append.

### `regularization_concept.txt` (repo root)

Restructured the same way: checklist first, then expansions. Sections 2 (Foundations, 8 subsections)
and 3 (The Penalties, 5 subsections) are written. Groups 1.3-1.10 still tagged `[   ]`. The user
edited this file directly afterwards — removed the title/how-to block and marked
`1.8 Deep Learning : Not required in this repo`. Respect those edits.

---

## 5. Agreed next-topic plan (discussed, not started)

After Decision Trees, the tree family in this order — the chain is the point:

```
single tree      low bias, HIGH variance
  -> BAGGING     average B trees; but Var(avg) = rho*sigma^2 + (1-rho)*sigma^2/B
  -> RANDOM FOREST   drive rho down by subsampling features; max_features IS the rho knob
  -> BOOSTING    averaging never fixes bias, so attack bias instead
  -> XGBOOST / LIGHTGBM / CATBOOST   regularized objective + engineering
```

**Start with Boosting, not Bagging.** `Ensemble_Learning/Boosting/theory/concept.txt` is **empty** —
no AdaBoost, no gradient boosting, no XGBoost/LightGBM/CatBoost anywhere in the repo. The bagging
side already has ~665 lines to restructure later.

Isolation Forest sits **off** that chain deliberately — unsupervised, random splits, prediction is
path length not a leaf value. Its peers are LOF / One-Class SVM, not Random Forest. It is currently
filed at `Ensemble_Learning/Bagging/theory/isolation_forest.txt` (316 lines), which is misleading
since it is not bagging; consider moving it or adding a note.

---

## 6. Working preferences observed

- Wants the format critiqued and fixed when it reads badly — flagged "somewhat confusing" and
  "not looking readable" and expected a restructure, not a defence.
- Wants gaps and errors surfaced plainly. Several build passes silently dropped content
  (a whole checklist group once); verifying output and reporting the miss was the right move.
- Prefers destructive changes confirmed first, but says "remove X" directly when they mean it.
- Asks for discussion-only answers sometimes ("now discussing only") — no file edits then.
