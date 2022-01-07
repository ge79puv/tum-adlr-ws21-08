import numpy as np
import torch
from matplotlib import pyplot as plt


class NURBS:
    def __init__(self, p, degree=3, k=None, w=None):
        self.p = p  # np.array(p)
        self.n_points, self.n_dim = self.p.shape
        self.degree = degree

        self.k = None
        self.set_knotvector(k=k)

        if w is None:
            self.w = torch.ones(len(self.p))  # np.ones(len(self.p))
        else:
            w_min = w.min()
            if w_min < 0:
                w = w + torch.abs(w_min)
                w_min = w.min()
            w_max = w.max()
            dst = w_max - w_min
            w = (w-w_min)/dst
            print(w)
            self.w = w

        assert len(self.k) == self.degree + self.n_points + 1  # noqa
        assert len(self.p) == len(self.w)

    def __repr__(self):
        return f"NURBS (degree={self.degree}, #points={self.degree})"

    def set_knotvector(self, k):
        if k is None:
            deg2 = int(np.ceil((self.degree+1)/2))
            k = ([0]*deg2 +
                 list(range(self.n_points)) +
                 [self.n_points-1] * (self.degree+1-deg2))

        k = np.atleast_1d(k)
        assert len(k) == self.degree + self.n_points + 1, f"len(k)[{len(k)}] == self.degree[{self.degree}] + self.n_points[{self.n_points}] + 1"
        self.k = k
        self.normalize_knotvector_01()

    def normalize_knotvector_01(self):
        self.k = self.k / self.k.max()

    @staticmethod
    def divide(n, d):
        # print("d", d)
        if isinstance(d, float):
            if np.all(d != 0):
                div = torch.div(n, d)
            elif np.any(d == 0):
                div = torch.zeros_like(n)
        else:
            if torch.all(d != 0.0) and isinstance(d, torch.FloatTensor):
                div = torch.div(n, d)
            elif torch.any(d == 0.0) and isinstance(d, torch.FloatTensor):
                div = torch.div(n, d)
                div = torch.where(torch.isnan(div), torch.full_like(div, 0), div)

        return div  # np.divide(n, d, out=np.zeros_like(n), where=d != 0)

    def f_in(self, u, i, n):

        f_in = self.divide(u - self.k[i], self.k[i+n] - self.k[i])
        # f_in = torch.from_numpy(f_in)
        return f_in

    def n_in(self, u, i, n):
        if n == 0:
            k0 = self.k[i]
            k1 = self.k[i+1]
            n_in = np.logical_and(k0 <= u, u < k1).int()

        else:
            n_in = (self.f_in(u=u, i=i, n=n) * self.n_in(u=u, i=i, n=n-1) +
                    (1-self.f_in(u=u, i=i+1, n=n)) * self.n_in(u=u, i=i+1, n=n-1))
        # print("n_in", self.f_in(u=u, i=i, n=n))
        return n_in

    def evaluate(self, u):
        # u = np.atleast_1d(u)
        x = 0.0
        for i in range(self.n_points):
            x = x + self.r_in(u=u, i=i)[:, np.newaxis] * self.p[i:i+1]

        # otherwise they are zero, weighting is little bit of somewhere, handle edgecases
        x[u == 0] = self.p[0]
        x[u == 1] = self.p[-1]
        # print(x.dtype)
        return x

    def evaluate_jac(self, u):
        # u = np.atleast_1d(u)
        jac = torch.zeros(u.shape + (self.n_points,))
        for i in range(self.n_points):
            jac[..., i] = self.r_in(u=u, i=i)

        return jac

    def r_in(self, u, i):
        # rational basis function
        r_in = self.n_in(u=u, i=i, n=self.degree) * self.w[i]

        temp = 0.0
        for j in range(self.n_points):
            # print("w", self.n_in(u=u, i=j, n=self.degree))
            temp += self.n_in(u=u, i=j, n=self.degree) * self.w[j]

        # print("r_in", r_in)
        # print("temp", temp)

        r_in = self.divide(r_in, temp)
        return r_in


def try_basis_function():

    n_points = 10
    p = np.random.random((n_points, 3))
    nurbs = NURBS(p=p, degree=0)

    u = np.linspace(0, 1, 100)
    for degree in range(5):
        nurbs.degree = degree
        nurbs.set_knotvector(k=np.linspace(0, 1, n_points+degree+1))
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        for i in range(10):
            n_in = nurbs.n_in(u=u, i=i, n=degree)
            ax.plot(u, n_in+i*0.001,  label=f"{degree} - {i}")

        ax.legend()


def try_random():
    n = 3
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(*x.T, color='black', marker='o', markersize=3)
    ax.plot(*p.T, color='blue', marker='o')


def try_random_jac():
    n = 5
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    def length_jac(x):
        steps = x[..., 1:, :] - x[..., :-1, :]
        do_dx = steps[..., :-1, :] - steps[..., +1:, :]
        return do_dx

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hx = ax.plot(*x.T, color='black', marker='o', markersize=3)[0]
    hp = ax.plot(*p.T, color='blue', marker='o')[0]

    dx_dp = nurbs.evaluate_jac(u)[..., 1:-1, :]  # does not depend on p
    for i in range(100):
        do_dx = length_jac(x)
        do_dp = (do_dx[:, np.newaxis, :] * dx_dp[:, :, np.newaxis]).sum(axis=0)
        nurbs.p[1:-1] -= do_dp[1:-1]
        x = nurbs.evaluate(u)

        hx.set_data(*x.T)
        hp.set_data(*nurbs.p.T)
        plt.pause(0.1)
        # input('press key for next gradient step')


def try_unit_circle():
    k = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=float)
    p = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])

    sq22 = np.sqrt(2)/2
    w = np.array([1, sq22, 1, sq22, 1, sq22, 1, sq22, 1])
    nurbs = NURBS(p=p, k=k, w=w, degree=2)
    u = np.linspace(0, 1, 90)
    x = nurbs.evaluate(u=u)
    j = nurbs.evaluate_jac(u=u)

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.plot(*x.T, color='black', marker='o')
    ax.plot(*p.T, color='blue', marker='o', markersize=3)


#from wzk.mpl import new_fig, DraggableCircleList
# def try_gui():
#     n = 3
#     p = np.random.random((n, 2))
#     u = np.linspace(0., 1, 20)
#
#     fig, ax = new_fig(aspect=1)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     dcl = DraggableCircleList(ax=ax, xy=p, radius=0.02, color='r')
#     nurbs = NURBS(p=p, degree=3)
#     x = nurbs.evaluate(u)
#     h = ax.plot(*x.T, color='k', marker='o', lw=3)[0]
#
#     def update(*args):
#         p = dcl.get_xy()
#         nurbs.p = p
#         x = nurbs.evaluate(u)
#         h.set_xdata(x[:, 0])
#         h.set_ydata(x[:, 1])
#
#     dcl.set_callback_drag(update)


if __name__ == '__main__':
    pass
    try_random_jac()
    # try_basis_function()
    # try_unit_circle()
    # try_random()



