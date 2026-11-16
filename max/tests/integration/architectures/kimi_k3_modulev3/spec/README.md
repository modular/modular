# K3 (ModuleV3) speculative-decoding oracle tests

Run remotely: `./bazelw test --config=remote-mi355
//max/tests/integration/architectures/kimi_k3_modulev3/spec:<target>
--test_output=errors --curses=no --nocache_test_results`

## `test_kda_spec_rollback.py` — every KDA state leaf rolls back

Each arm verifies `K + 1` rows per request (`K` in 1 and 7; one batch
mixes widths), rolls back over the accepted count, and requires the
pools and the verify's readout on the accepted rows to be bit-identical
to a plain `kda_decode` / conv forward over those rows alone, with the
reference decoding one token per request per call and `pin_fma=True`.
The layer-level arm and the production-path arm live in the parent
directory's `test_spec_kda_rollback.py`, one file for both TP degrees;
this one keeps the op-level oracle, the ring-leaf declaration contract
and the sabotage controls. Cannot catch wrong gate tensors or TP
sharding (one device).

## `spec_parity.py` + `test_spec_parity.py` — parity and determinism

Serves one checkpoint twice — DSpark (width 7) and plain — and reports per
prompt: identical, the first divergence, the ids around it, and the plain top-2
logit gap there (a few bfloat16 ulps = near-tie). The pipeline's draft counts
must show drafts proposed, a synthetic run drafts accepted, so a dead or
never-folding draft path cannot hide behind parity. The graded step-bonus
oracle holds every committed bonus token to the plain argmax at its position.
One shared plain pipeline grades every run. The smoke test runs this pair at
TP1.
