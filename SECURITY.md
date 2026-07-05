# Security Policy

## Supported versions

This is a research codebase accompanying an MSc thesis. Only the latest
`main` branch receives security consideration.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email the maintainer at the address listed in the GitHub profile
linked from this repository. Include:

- A description of the issue and its potential impact.
- Steps to reproduce (minimal example if possible).
- Your assessment of severity.

You will receive an acknowledgment within 5 business days. Please do not
disclose the issue publicly until a fix or mitigation has been released.

## Scope

This project trains and evaluates reinforcement-learning agents against
simulated IoT attacks. It does **not** process live network traffic, handle
real credentials, or deploy to production systems. Security considerations
are therefore primarily about:

- Supply-chain risks in pinned dependencies (see `requirements.txt`).
- Arbitrary code execution via untrusted model checkpoints or data files —
  **never load a `.zip` / `.joblib` / `.npy` artifact from an untrusted
  source**; `joblib` and `torch.load` can execute arbitrary code.

## Out of scope

- The thesis LaTeX under `tex/` is documentation only.
- `notebooks/` is exploratory and not on the reproduction path.
