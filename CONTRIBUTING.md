# Contributing

By submitting a contribution, you agree to the terms in `CLA.md`.

Run `python -m pytest` and `ruff check --isolated --select E9,F src tests`
before you open a pull request; continuous integration runs both. Keep the
numerics in PyTorch tensor operations, so that the model runs on CPU and GPU
and stays differentiable.
