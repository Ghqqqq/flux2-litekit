# Contributing

Thanks for contributing to Flux2 LiteKit.

## Scope

- Keep the public surface domain-neutral.
- Preserve the official `Flux2KleinPipeline` semantics for `i2i`.
- Prefer small, reviewable pull requests over broad refactors.

## Local Checks

Run these checks before opening a pull request:

```bash
python -m py_compile flux2_litekit/*.py
bash -n scripts/*.sh
```

If your change affects training or inference behavior, include:

- the exact command you ran
- the config file used
- a short note on what was verified

## Pull Requests

- Describe the user-facing change first.
- Call out config, checkpoint, or compatibility impacts explicitly.
- Avoid bundling unrelated cleanup into the same PR.
