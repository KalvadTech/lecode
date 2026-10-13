# Security

## Supported versions

Only the latest GitHub release receives security fixes. lecode is not on
PyPI; update with `uv tool install --force git+https://github.com/KalvadTech/lecode`.

## Reporting a vulnerability

Report privately through a
[GitHub Security Advisory](https://github.com/KalvadTech/lecode/security/advisories/new).
Please do not open a public issue with exploit details. If private reporting
is unavailable, open a minimal public issue saying you have a security
report and a maintainer will reach out.

## Scope notes

lecode is an agent that runs tools on your machine. Some behaviors are
**by design**, not vulnerabilities:

- `yolo` permission mode (the default) executes what the model emits.
  Use `--safe` / `readonly` mode and per-tool rules for untrusted input.
- Hooks can narrow but never widen a permission decision — a hook that
  tries to widen is ignored, by design.

Genuine vulnerabilities include: bypassing a `deny` rule or `readonly`
mode, exfiltrating secrets outside the declared tool surface, and config or
session files written with unsafe permissions.
