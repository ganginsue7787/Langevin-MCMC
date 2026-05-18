# Contributing

Thank you for your interest. This repository accompanies a submitted paper.

## Issue Reports

If you find a bug in the code or a discrepancy between the code and the paper:

1. Open an **Issue** with the label `bug` or `paper-discrepancy`.
2. Include the exact command or code that reproduces the problem.
3. Include the expected vs. actual output.

## Questions

For questions about the mathematics or theory, open a **Discussion** rather than an Issue.

## Pull Requests

PRs are welcome for:
- Bug fixes in `code/core/`
- Additional test cases for the experiments
- Documentation improvements

Please run the verification before submitting:

```bash
python code/verify_paper.py
```

All three experiments (EXP-1, EXP-2, EXP-3) must pass.

## Code Style

- PEP 8 for Python
- Every function must have a docstring with `Paper:` reference to the relevant theorem/section
- No new dependencies beyond `requirements.txt` without discussion
