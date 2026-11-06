"""The vertically implicit solver of the acoustic step: values and derivatives."""
import torch

from mpas_pytorch.autodiff.tridiagonal_solve import tridiagonal_solve, tridiagonal_solve_scan


def system(ncells=3, nlevels=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 0.3 * torch.rand(ncells, nlevels, generator=g, dtype=torch.float64)
    alpha = 0.8 + 0.4 * torch.rand(ncells, nlevels, generator=g, dtype=torch.float64)
    gamma = 0.3 * torch.rand(ncells, nlevels, generator=g, dtype=torch.float64)
    rhs = torch.randn(ncells, nlevels + 1, generator=g, dtype=torch.float64)
    return a, alpha, gamma, rhs


def reference(a, alpha, gamma, rhs):
    """Forward elimination then back substitution, one column at a time in Python floats."""
    out = []
    for i in range(rhs.shape[0]):
        u = rhs[i].tolist()
        for k in range(1, a.shape[1]):
            u[k] = (u[k] - a[i, k].item() * u[k - 1]) * alpha[i, k].item()
        for k in range(a.shape[1] - 1, -1, -1):
            u[k] = u[k] - gamma[i, k].item() * u[k + 1]
        out.append(u)
    return torch.tensor(out, dtype=torch.float64)


def test_solution_matches_the_column_recurrence():
    a, alpha, gamma, rhs = system()
    assert torch.equal(tridiagonal_solve(a, alpha, gamma, rhs), reference(a, alpha, gamma, rhs))


def test_scan_form_matches_the_sequential_form():
    a, alpha, gamma, rhs = system(seed=1)
    torch.testing.assert_close(tridiagonal_solve_scan(a, alpha, gamma, rhs),
                               tridiagonal_solve(a, alpha, gamma, rhs), rtol=1e-13, atol=1e-13)


def test_reverse_and_forward_derivatives_match_finite_differences():
    inputs = tuple(x.requires_grad_() for x in system(ncells=2, nlevels=4, seed=2))
    assert torch.autograd.gradcheck(tridiagonal_solve, inputs, check_forward_ad=True)


def test_adjoint_identity():
    a, alpha, gamma, rhs = system(seed=3)
    g = torch.Generator().manual_seed(4)
    v = tuple(torch.randn(x.shape, generator=g, dtype=torch.float64) for x in (a, alpha, gamma, rhs))
    w = torch.randn(rhs.shape, generator=g, dtype=torch.float64)
    _, jv = torch.func.jvp(tridiagonal_solve, (a, alpha, gamma, rhs), v)
    _, pullback = torch.func.vjp(tridiagonal_solve, a, alpha, gamma, rhs)
    jtw = pullback(w)
    lhs = torch.dot(jv.reshape(-1), w.reshape(-1))
    rhs_ = sum(torch.dot(x.reshape(-1), y.reshape(-1)) for x, y in zip(v, jtw))
    torch.testing.assert_close(lhs, rhs_, rtol=1e-13, atol=0.0)
