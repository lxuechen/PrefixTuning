import torch

from . import utils


def test_power_iter():
    """Test power iteration with batched and non-batched version."""
    # TODO: Test diagonalizable non-PSD matrices.
    torch.set_default_dtype(torch.float64)

    # Batched version.
    batch_size, d = 16, 10
    A = torch.randn(batch_size, d, d)
    At = A.permute(0, 2, 1)
    A = At.bmm(A)
    v0 = torch.randn(batch_size, d, 1)

    func = lambda vec: torch.matmul(A, vec)
    eval_pi, evec_pi = utils.power_iter(func=func, v0=v0, num_iters=1000, eigenvectors=True)

    eval_tr, evec_tr = torch.symeig(A, eigenvectors=True)
    eval_tr = eval_tr[:, -1]
    torch.testing.assert_allclose(eval_pi, eval_tr)

    # Non-batched version.
    A = torch.randn(d, d)
    A = A.t().mm(A)
    v0 = torch.randn(d, 1)

    func = lambda vec: torch.matmul(A, vec)
    eval_pi, evec_pi = utils.power_iter(func=func, v0=v0, num_iters=1000, eigenvectors=True)

    eval_tr, evec_tr = torch.symeig(A, eigenvectors=True)
    eval_tr = eval_tr[-1]
    torch.testing.assert_allclose(eval_tr, eval_pi)


def test_top_singular():
    torch.set_default_dtype(torch.float64)

    d = 10
    mat = torch.randn(d, d)

    singularval, lsv, rsv = utils.top_singular(
        mat, left_singularvectors=True, right_singularvectors=True, num_iters=200)

    u, s, v = torch.svd(mat)
    lsv_true = u[:, 0:1]
    rsv_true = v[:, 0:1]
    singularval_true = s[0]
    torch.testing.assert_allclose(singularval_true, singularval)
    assert torch.allclose(lsv_true, lsv) or torch.allclose(-lsv_true, lsv)
    assert torch.allclose(rsv_true, rsv) or torch.allclose(-rsv_true, rsv)
