import numpy as np
import torch
from matplotlib import pyplot as plt


class NURBS:
    def __init__(self, p, degree=3, k=None, w=None, u=None):
        self.N_in = None
        self.p = p
        self.n_paths, self.n_points, self.n_dim = self.p.shape
        self.degree = degree

        if k is None:
            deg2 = int(np.ceil((self.degree+1)/2))
            self.k = [0]*deg2 + list(range(self.n_points)) + [self.n_points-1] * (self.degree+1-deg2)
        else:
            self.k = k

        if w is None:
            self.w = torch.ones((self.n_paths, self.n_points))
        else:
            w[w < 0] = 0.001
            self.w = w

        if u is None:
            self.u = np.linspace(0.01, self.n_points - 1 - 0.01, 15)
        else:
            self.u = u

        assert len(self.k) == self.degree + self.n_points + 1
        assert self.p.shape[:-1] == self.w.shape

    def __repr__(self):
        return f"NURBS (degree={self.degree}, #paths={self.n_paths}, #points={self.n_points}, dim={self.n_dim})"

    @staticmethod
    def divide(n, d):
        return np.divide(n, d, out=np.zeros_like(n), where=d != 0)

    def f_in(self, i, n):
        f_in = self.divide(self.u - self.k[i],  self.k[i+n] - self.k[i])
        return f_in

    def n_in(self, i, n):
        if n == 0:
            k0 = self.k[i]
            k1 = self.k[i+1]
            n_in = np.logical_and(k0 <= self.u, self.u < k1).astype(int)

        else:
            n_in = (self.f_in(i=i, n=n) * self.n_in(i=i, n=n-1) +
                    (1-self.f_in(i=i+1, n=n)) * self.n_in(i=i+1, n=n-1))
        return n_in

    def compute_n_in(self):
        self.N_in = torch.zeros((self.n_points, self.u.shape[0]))
        for i in range(self.n_points):
            self.N_in[i] = torch.tensor(self.n_in(i=i, n=self.degree))

    def r_in(self):
        # rational basis function
        if self.N_in is None:
            self.compute_n_in()
        return self.w[..., None] * self.N_in / (self.w @ self.N_in)[..., None, :]

    def evaluate(self):
        paths = torch.matmul(torch.moveaxis(self.r_in(), 1, 2), self.p)
        return paths

    def evaluate_jac(self):
        return torch.moveaxis(self.r_in(), 1, 2)

