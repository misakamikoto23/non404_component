# Contest 6 local scorer

`component-lab score` re-implements the public scoring rules of Contest Problem 6
(spec 0803Rev1, section 5 and appendix 4). It is **not** the official evaluator;
every open choice is a flag and is listed under "Assumptions" in each report.

```
score = 100 × (0.30·ComponentF1 + 0.25·PinF1 + 0.35·NetHypergraphF1 + 0.10·NetLineF1)
```

Missing, unreadable or schema-violating predictions score 0 and still count in the mean.
The **headline** number is `type_policy=strict`, `anon_policy=strict` and no oracle. Reports mark
every other row "(diagnostic)" and print all policies and tolerances at the top.

## Data and split

```bash
component-lab extract-gt --archive "../赛题六公开数据集.zip" --output data/official_contest6
python -I scripts/fingerprint_sources.py . outputs/eval_split_audit_20261017/fingerprints.json
python -I scripts/build_group_split.py . outputs/eval_split_audit_20261017 configs/contest6_group_split_v1.json
```

- `data/official_contest6/train200/*.json` and `gt10/*.json` hold the official GT, in bottom-left
  pixel coordinates.
- **`configs/contest6_group_split_v1.json`** is the split to use: test 40 (sealed), dev 20 (model
  selection), train 140. It assigns whole duplicate groups. The groups are built from exact pixels,
  calibrated perceptual hashes, official label similarity and declared lab variants, over the
  official, PCBnet Dataset300, HF open_schematics and all component_lab images.
- It is a split for a **clean retrain**. A retrain must drop every image listed in
  `outputs/eval_split_audit_20261017/forbidden_for_clean_retrain.json`. That includes 288 images
  from existing detector training folders, plus ResNet crops.
- Existing checkpoints, round 004 included, are contaminated on it. Per-case flags show how; only
  18 test cases were never in any detector train or val folder.
- Excluding everything round 004 trained on leaves only 37 official cases, too few for test 40 plus
  dev 20. Full audit: `outputs/eval_split_audit_20261017/REPORT.md`.
- GT10 groups are always train. GT10 duplicates train200, so use it as a format smoke test only.
- `configs/contest6_dev_diagnostic_contaminated.json` is the superseded 30-case split (14 detector
  val and 16 detector test cases). Use it only to compare against earlier diagnostic runs.

## Scoring

```bash
component-lab score --pred <pred_dir> --gt data/official_contest6/train200 --split configs/contest6_group_split_v1.json --partition dev --ablate --out outputs/eval_<run>
```

Score the `test` partition only once per frozen system.

- Prediction files may be nested in any way; the case id comes from the file name
  (`0001-KiCad.json`, `0001.json`, `case_001/case_001.json` all map to case 1).
  Predictions must be in the GT's **bottom-left** frame; `component_lab.eval.coords.flip_case`
  converts from top-left image pixels. A prediction left in top-left coordinates still scores 35,
  because NetHypergraph has no coordinates.
- `--type-policy strict|mapped|both` (default `strict`). `mapped` is a diagnostic that compares types
  after `src/component_lab/eval/type_vocab.json`:
  - box/boxic → ic, r → resistor, c → capacitor, l → inductor;
  - npn/pnp forms collapse to mosfet/bjt, because GT10 re-annotates bare `mosfet`/`bjt` with
    polarity.
  - Types missing from the table (in the official data, only `m3螺丝`) are listed under "Types not
    covered" in each report.
  - The text rule keys on the canonical GT type under both policies: Name for ic (including
    `box`), value for resistor/capacitor, nothing otherwise.
- `--anon-policy strict|observable_diagnostic|both` (default both). About 44% of GT component keys
  are internal `∅<n>` IDs. These include real parts (battery, buzzer, motor...) as well as gnd/v/net
  labels; every report prints the ∅ counts by type.
  - `strict` scores them like any key.
  - `observable_diagnostic` drops ∅ components, their pins and ∅ net members from **both** sides. It
    is not the contest metric.
  - A pipeline that leaves ∅ objects out of its output is a prediction-side choice, scored under
    `strict`.
  - ∅ IDs are not even stable between the organizer's own two annotation versions (GT10 vs train200).
- `--bbox-metric max_corner|center`, `--bbox-tol 20`: how bbox error is measured (assumption).
- `--point-tol 5`: pin and segment-endpoint tolerance (Euclidean, inclusive: 5.0 passes, 5.01 fails).
- `--net-tiebreak segments|none` (default `segments`, an **assumption** not in the spec). With
  `segments`, ties between nets with equal pin overlap go to the pair with more matching segments,
  and nets with no pins can pair with each other by segments. `none` uses pin overlap alone. The two
  modes differ only on the 4 official cases that contain a net with no pins.
- `--oracle components,pins,nets` is an **output-substitution oracle**: after the pipeline has run,
  that JSON section is replaced by GT, and later stages are not rerun. It is not a pipeline oracle.
  `--ablate` runs each single-section substitution.
- Each summary row has a **95% CI** from a case-level percentile bootstrap (10,000 resamples). It
  covers case sampling only, not scorer assumptions. With 40 cases, expect intervals several points
  wide.
- Outputs: `per_case.csv` with scores, tp/fp/fn, and diagnostics (key hits; wrong type, bbox or text;
  wrong pinname or point; matched nets; unmapped types). Also `summary.md` and `summary.json`.

## Rules as implemented

- Component: exact key match. A match is correct when the type is equal (raw or canonical, per
  `--type-policy`), the bbox is within tolerance, and the text matches strictly. GT `None` vs pred
  `None` counts as equal (assumption).
- Pin: key `(component, pin_<n>)`; the pinname is strict and case-sensitive, and the point must be
  within 5 px.
- NetHypergraph: each net is a set of `component.pin` members. Keys may contain dots, so the longest
  known component key wins. Nets are paired by the Hungarian maximum of pin-set intersection, with
  ties resolved per `--net-tiebreak`. tp/fp/fn are counted per pin over matched pairs; members of
  unmatched nets are fp/fn.
- NetLine: compared only within matched net pairs. 2-point segments are paired by
  maximum-cardinality matching; both endpoints must be within 5 px, in either orientation. Edges of
  unmatched nets are fp/fn.
- A metric whose GT and prediction are both empty has F1 = 1.

Tests: `python -m unittest tests.test_eval_score tests.test_eval_rules tests.test_eval_data`.
