import torch


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def weight_decomposition(W, R=None, rank=1):
    outdim, indim = W.shape[0], W.shape[1]

    R = torch.normal(0, 1, size=(indim, rank)).cuda()
    for _ in range(1):
        L = torch.matmul(W, R)  # outdim x rank
        orthogonalize(L)
        R = torch.matmul(W.T, L)  # indim x rank
        orthogonalize(R)
    R = R.T
    approx_error = W - torch.matmul(L, R)
    return L, R, approx_error
