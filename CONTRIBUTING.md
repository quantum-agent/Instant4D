# Contributing

This fork exists to carry reusable fixes that make Instant4D easier to adopt safely in real projects.

## What belongs in this fork

Good candidates:
- upstream-ish bug fixes
- path / config / CLI cleanup
- portability improvements
- documentation fixes
- small, fast regression tests for pure or near-pure logic
- lightweight CI that can run on GitHub-hosted CPU runners

Usually better kept outside the fork (for example in a parent project such as `video-to-4d-gs`):
- project-specific orchestration
- private datasets and fixtures
- benchmark suites
- GPU-heavy smoke tests
- full reconstruction/training verification
- workflow glue that is specific to one deployment or one downstream pipeline

## Testing philosophy for this fork

Keep fork-side verification cheap and reliable.

That means we prefer:
- unit tests for pure helpers and data-shaping logic
- regression tests for recently fixed CLI/config behavior
- shell sanity checks like `--help`

And we generally avoid putting these into the fork CI:
- tests that require CUDA
- tests that need large checkpoints or datasets
- end-to-end quality evaluation
- long-running preprocessing or training jobs

The goal is not to prove full reconstruction quality in this fork.
The goal is to catch accidental regressions in the reusable fixes we carry here.

## Local verification

Typical lightweight checks:

```bash
pytest -q
bash script/reconstruct.sh --help
bash script/smoke_test.sh --help
```

If you add a new reusable cleanup or pure helper, please add or update a focused regression test alongside it.
