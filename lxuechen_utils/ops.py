import torch


def kronecker(A, B):
  return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))



def test_kronecker():
  m, n, p, q  = 3, 4, 1, 2
  A = torch.randn(m, n)
  B = torch.tensor([[1, 2]]).float()
  print(f'A: {A}, B: {B}')
  print(f'Kronecker: {kronecker(A, B)}')


if __name__ == '__main__':
  test_kronecker()
