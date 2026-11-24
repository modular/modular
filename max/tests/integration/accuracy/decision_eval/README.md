# Decision model evals

Scores a MAX server's `/v1/systemone` (and so `/v1/decisions`) answers against
JevBench and the Decision Index, checks MAX against the decider reference
implementation, and adds the calibration metrics the benchmarks leave out.

All metrics are the same everywhere: accuracy is the top option against the
gold option, Brier is the sum over options of squared error (averaged over
cases), ECE uses 10 equal-width bins on the top probability, and an order flip
is a reordered twin of a case picking a different option.

## 1. Serve a model

```sh
./bazelw run //max/python/max/_entrypoints:pipelines -- serve \
  --model-path Mapika/decider-2b --max-length 8192 --max-batch-size 16 \
  --port 8123
```

`--max-length` and `--max-batch-size` here are conservative values that fit
most GPUs; raise them on a larger one.

MAX picks the decider prompt format from the checkpoint's
`decider_config.json`, so a chat model such as `Qwen/Qwen3.5-2B` is served the
same way.

## 2. JevBench

```sh
huggingface-cli download Leanmcp/jevbench --repo-type dataset --local-dir jev

./bazelw run //max/tests/integration/accuracy/decision_eval:run_jevbench -- \
  --cases-dir jev/cases --base-url http://localhost:8123 \
  --model Mapika/decider-2b --output-dir out/max_2b
```

It sends every text case with at most 10 options (the decider limit) as one
System One request, then writes `predictions.jsonl` and `scores.json`. Skipped:
the image slices, SST-5 (its state isn't redistributed), and BANKING77 (77
options). Both scripts exit non-zero if any case fails or goes unanswered, so
a score never silently covers an easier subset; `--allow-partial` opts in to a
diagnostic run, and `scores.json` records `complete`. Score a published system
on the same cases for comparison:

```sh
./bazelw run //max/tests/integration/accuracy/decision_eval:score_published -- \
  --cases-dir jev/cases \
  --published jev/predictions/djev-0.1/predictions.jsonl --output-dir out/djev
```

## 3. Parity with the reference implementation

The reference is the `decider-ai` package (Apache-2.0). It needs `torch` and a
recent `transformers`, so run it from its own environment, not Bazel:

```sh
pip install torch transformers huggingface_hub click numpy
pip download decider-ai==1.4.0 --no-deps -d wheel
unzip -q wheel/decider_ai-1.4.0-*.whl -d decider_ai

python reference_decider.py --model-path Mapika/decider-2b \
  --reference-dir decider_ai --cases-dir jev/cases \
  --output-dir out/ref_2b --limit-per-slice 200

./bazelw run //max/tests/integration/accuracy/decision_eval:parity -- \
  out/max_2b/predictions.jsonl out/ref_2b/predictions.jsonl \
  --min-agreement 0.98
```

The reference runs in fp32, one row at a time, with the model's own
temperatures. MAX runs bf16 with batching, so expect probabilities within a few
hundredths and a top option that differs only where two options nearly tie.

## 4. Decision Index

The kit is
[apolinario/decision-index](https://github.com/apolinario/decision-index) (MIT).
Its `http` engine speaks `/v1/systemone`:

```sh
git clone https://github.com/apolinario/decision-index && cd decision-index
pip install -e ".[rebuild]" && export HF_HUB_DISABLE_XET=1
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
  --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
  --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
python -m decision_index suite sample --n 3000 --out sample.jsonl.gz
python -m decision_index run --engine http \
  --option base_url=http://127.0.0.1:8123 --option model=Mapika/decider-2b \
  --rows sample.jsonl.gz --out runs/decider-2b
python -m decision_index score --results runs/decider-2b/results.jsonl
```

The full suite needs access to gated sources (HLE) and takes hours, so a
rebuild without that access covers only part of the suite. The kit has no
calibration metric, so add Brier and ECE from the recorded probabilities:

```sh
./bazelw run //max/tests/integration/accuracy/decision_eval:decision_index_calibration -- \
  --rows sample.jsonl.gz --results runs/decider-2b/results.jsonl
```

A request the server refuses as over capacity (for example more than 10
options with a decider model) is recorded by the kit as `unsupported`, and
counts as wrong in the kit's index.
