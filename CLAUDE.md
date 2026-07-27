# Project context

Research project: how much does brain-to-text decoding accuracy depend on the quality of its
forced-alignment labels? Built on the Willett et al. (2021) intracortical handwriting dataset.

**Read `docs/RQ1_alignment_error_types.md` first.** It is the paper and the source of truth for
every finding. `docs/LAB_NOTEBOOK.md` records why each decision was made, including the ones
that turned out wrong.

## Status

RQ1 is complete: 198 committed runs, three findings.

1. Alignment error *type* beats error *rate*. Boundary timing errors are nearly free (up to 17%
   of frame labels mistimed costs nothing measurable); character identity errors cost
   immediately. Corruption is 2.4x to 10.8x steeper than jitter.
2. Sequence truncation can invert an architecture conclusion. At 1500 bins the decoders looked
   identical (1.64 pp spread, below the 2.32 pp noise floor); at 3000 bins they separate by
   17.55 pp.
3. The pre-registered hypothesis (H1c: label quality dominates architecture) was **refuted and
   inverted**, 0.54x. Reported unmodified.

RQ2 (cross-session calibration) and RQ3 (alignment-free CTC) were deliberately descoped and
live as future work in the paper, section 7. Do not restart them without being asked.

## Hard conventions

These are load-bearing. Breaking them undermines the project's central claim, which is that its
results are checkable.

- **No em dashes anywhere.** Enforced by `tests/test_no_em_dashes.py`. Use commas, colons,
  semicolons, parentheses or hyphens.
- **Every number in the docs must trace to a committed artifact.** Never hand-type a statistic.
  `analysis/analyze_exp1.py` re-derives them all; `tests/test_results_integrity.py` fails if the
  writeup drifts from the data.
- **Never edit a pre-registered hypothesis to match a result.** `docs/RESEARCH_PLAN.md` sections
  1 to 7 are frozen. Anything decided after seeing data goes in section 8, labelled exploratory.
- **Never run a sweep at `--max-len 1500`.** It compresses CER toward a ceiling and inverts the
  architecture conclusion. Use 3000. This is the whole point of finding 2.
- **`docs/RESULTS.md` is a superseded preliminary study** kept with a caveat banner. Do not cite
  its numbers or delete it.

## Commands

```bash
python verify_data.py --data-dir <path>          # check the dataset before spending GPU time
python experiments/exp1_alignment_sensitivity.py --sweep --max-len 3000 \
    --out-dir results/exp1_sweep_3000 --data-dir <path>
python analysis/analyze_exp1.py                  # all statistics + results/exp1_summary.json
python analysis/make_figures.py --experiment exp1
pytest tests/ -q                                 # 37 tests
```

Dataset is not in the repo. It lives at `C:\Users\aaron\handwritingBCIData\handwritingBCIData`
(deliberately outside OneDrive, which would otherwise sync 5.7 GB). Pass `--data-dir`.

Use the `handwriting` conda env: it has torch with CUDA. The `tutorial` env does not.

## Downstream

Findings are mirrored on the personal site at
`C:\Users\aaron\OneDrive\Desktop\Coding\Personal_portfolio\aaron-portfolio`, in
`src/data/projects.js` under slug `imagined-handwriting-decoding`. **If results change here,
that entry must be updated too**, or the site advertises numbers this repo has superseded.
