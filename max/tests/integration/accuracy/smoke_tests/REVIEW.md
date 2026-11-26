# Smoke test review

## Read first

- `.agents/skills/model-bringup/SKILL.md` and its
  `references/idiomatic-bringup.md`. Flag every violation of their rules,
  citing the rule.

## Mach smoke test

In `smoke_test_github_matrix.py`, flag any HW tag that exists only to keep
Mach off HW it cannot serve yet (such as a `_MACH_*` tag). Mach runs every
entry on the same HW as MAX, and a red Mach job is the intended signal to
implement the model or drop it from the matrix. The one exception is the
`mach-ci` matrix filter in `serveSmokeTestBase.yaml`: without a
`models_override`, it skips a variant with no `mach_variants.py` entry and a
row with no Mach registry entry for its GPU count, and lists them in the run
summary.
Flag any other list in `smoke_tests/internal` that makes the Mach smoke test
refuse or skip a key before trying to serve it.

Flag any change that makes the Mach smoke test read a MAX recipe or
translate recipe settings into Mach flags. A Mach variant is an entry in
`smoke_tests/internal/mach_variants.py`, with flags copied from
`mach/docs/models`.
