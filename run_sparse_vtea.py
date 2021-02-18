import numpy as np
import ipopt
from datetime import datetime

MIN = -2.0e19
MAX = 2.0e19


class Parameter:
    def __init__(self, D, P, jj: np.array, ll: np.array,
                 tt0: np.array, uu: np.array, ww: np.array, mm: np.array,
                 CC: np.ndarray, KK: np.ndarray, RR: np.ndarray, SS: np.ndarray):
        self.D = D
        self.P = P
        self.ll = ll
        self.tt0 = tt0
        self.uu = uu
        self.ww = ww
        self.mm = mm
        self.CC = CC
        self.KK = KK
        self.RR = RR
        self.SS = SS
        self.s = SS.shape[0]
        self.vj = jj


class Variable:
    def __init__(self, v, t, e, a):
        self.x0 = np.concatenate((v, t, e, a))
        self.len_x = self.x0.shape[0]

        self.m, self.l, self.n, self.k = v.shape[0], t.shape[0], e.shape[0], a.shape[0]
        self.start_idx = [self.m, self.m+self.l, self.m+self.l+self.n]


class MeMoSparse:
    def __init__(self, f: np.array, var: Variable, para: Parameter):
        """
        :param f: array with the same length as the variable 'v'
        """
        self.f = np.concatenate([f, np.zeros(var.l + var.n + var.k)])
        self.var = var
        self.para = para

        self.lj = self.para.vj.shape[0]
        self.c_start_idx = [self.lj, self.lj+self.para.s, self.lj+self.para.s+1, self.lj+self.para.s+1+self.var.l]

    def objective(self, x):
        return np.dot(self.f, x)

    def gradient(self, x):
        """
        The gradient of the objective function
        """
        return self.f

    def constraints(self, x):
        """
        The 'm' + 's' + 1 + 2*'l' linear and 'l' nonlinear constraints
        """
        v = x[:self.var.m]
        vj = v[self.para.vj]
        t = x[self.var.start_idx[0]: self.var.start_idx[1]]
        e = x[self.var.start_idx[1]: self.var.start_idx[2]]
        a = x[self.var.start_idx[2]:]

        p = np.matmul(self.para.CC, e)

        return np.concatenate([
            vj - np.matmul(self.para.KK, e),
            np.matmul(self.para.SS, v),
            [np.dot(self.para.mm, p)],
            np.divide(p, self.para.ww) - t,
            np.multiply(self.para.tt0, np.power(2, np.matmul(self.para.RR, a))) - t,
        ])

    def jacobianstructure(self):
        """
                :[0](m) :[1](l) :[2](n) :[:](k)
        :[0](j) *       -       *       -
        :[1](s) *       -       -       -
        :[2](1) -       -       *       -
        :[3](l) -       dia     *       -
        :[:](l) -       dia     -       *
        :return:
        """
        j, m, s, l, n, k = self.lj, self.var.m, self.para.s, self.var.l, self.var.n, self.var.k
        r_sidx, c_sidx = self.c_start_idx, self.var.start_idx
        n_r, n_c = r_sidx[-1] + l, c_sidx[-1] + k

        ss = np.zeros((n_r, n_c))
        ss[:r_sidx[1], :c_sidx[0]] = np.ones((j+s, m))                     # r=:, c=0

        ss[r_sidx[2]:r_sidx[3], c_sidx[0]:c_sidx[1]] = np.eye(l)           # r=3, c=1
        ss[r_sidx[3]:, c_sidx[0]:c_sidx[1]] = np.eye(l)                    # r=4, c=1

        ss[:r_sidx[0], c_sidx[1]:c_sidx[2]] = np.ones((j, n))              # r=0, c=2
        ss[r_sidx[1]:r_sidx[3], c_sidx[1]:c_sidx[2]] = np.ones((1+l, n))   # r=1:,c=2

        ss[r_sidx[3]:, c_sidx[2]:] = np.ones((l, k))                       # r=:, c=3

        return np.nonzero(ss)

    def jacobian(self, x):
        """
        The Jacobian of the constraints with shape (|constraints|, |x|)
        """
        j, m, s, l, n, k = self.lj, self.var.m, self.para.s, self.var.l, self.var.n, self.var.k

        j1 = np.concatenate([
            np.ones((j, m)),
            np.negative(self.para.KK)
        ], axis=1).flatten()

        j2 = self.para.SS.flatten()

        j3 = np.dot(self.para.mm, self.para.CC).flatten()

        j4 = np.concatenate([
            np.negative(self.para.ww.reshape((-1, 1))),
            self.para.CC
        ], axis=1).flatten()

        a = x[self.var.start_idx[2]:]
        tmp = np.multiply(np.log(2)*np.power(2, np.matmul(self.para.RR, a)), self.para.tt0)
        j5 = np.concatenate([
            np.negative(np.ones((l, 1))),
            np.multiply(tmp.reshape((-1, 1)), self.para.RR)       # da
        ], axis=1).flatten()

        return np.concatenate([j1, j2, j3, j4, j5])

    def hessianstructure(self):
        """
        the structure of the hessian is of a lower triangular matrix.
        shape of (|x|, |x|)
        """
        k = self.var.k
        row, col = np.nonzero(np.tril(np.ones((k, k))))

        return row + self.c_start_idx[-1], col + self.var.start_idx[-1]

    def hessian(self, x, lagrange, obj_factor):
        """
        the hessian of the lagrangian
        :param lagrange: 1d-array with length of |j6|.shape[0]
        """
        a = x[self.var.start_idx[2]:]
        H = np.power(np.log(2), 2) * np.multiply(self.para.tt0, np.matmul(self.para.RR, a))
        H = np.multiply(lagrange[-self.var.l:], H)
        H = np.multiply(H.reshape((-1, 1)), self.para.RR)
        H = np.matmul(H.T, self.para.RR)

        return np.tril(H).flatten()

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trial):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")
        # print("Current Time =", current_time)


def optimizer(ff: np.array, var: Variable, para: Parameter):
    lb = np.concatenate(
        (para.ll, [para.D]*var.l, [0]*var.n, [MIN]*var.k))
    ub = np.concatenate([para.uu, [MAX]*(var.l + var.n + var.k)])

    j = para.vj.shape[0]
    cl = np.concatenate(
        [[MIN] * j, np.zeros(para.s + 1 + 2 * var.l)])
    cu = np.concatenate(
        [np.zeros(j + para.s), [para.P], np.zeros(2 * var.l)])

    # define problem
    nlp = ipopt.problem(
        n=var.len_x,
        m=len(cl),
        problem_obj=MeMoSparse(ff, var, para),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    # Set solver options
    # nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    # scale problem
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=[1] * var.x0.shape[0]
    )
    nlp.addOption('nlp_scaling_method', 'user-scaling')

    # solve problem
    x, info = nlp.solve(var.x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))
    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
    print("Objective=%s\n" % repr(info['obj_val']))

    return x, info


def test():
    m, l, n, k = 3, 5, 7, 11
    v = np.ones(m)
    t = np.ones(l)
    e = np.ones(n)
    a = np.ones(k)

    ll = np.ones(m) * (-10)
    jj = np.array([0, 1])
    uu = np.ones(m) * 10
    KK = np.abs(np.random.randn(jj.shape[0], n))
    s = 13
    SS = np.abs(np.random.randn(s, m))
    P = 100
    CC = np.abs(np.random.randn(l, n))
    ww = np.abs(np.random.randn(l))
    D = 0.01
    RR = np.abs(np.random.randn(l, k))
    tt0 = np.abs(np.random.randn(l))
    ff = np.ones(m)
    mm = np.ones(l)
    variable = Variable(v, t, e, a)
    parameter = Parameter(D, P, jj, ll, tt0, uu, ww, mm, CC, KK, RR, SS)
    x, info = optimizer(ff, variable, parameter)

# ######################################
# run with toy data
# test()

# ######################################
# run with toy mini data
import os

basepath = 'regulateme/data_mini'

SS = np.load(os.path.join(basepath, 'SS.npy'))
RR = np.load(os.path.join(basepath, 'RR.npy'))
CC = np.load(os.path.join(basepath, 'CC.npy'))
KK = np.load(os.path.join(basepath, 'KK.npy'))
ll = np.load(os.path.join(basepath, 'll.npy'))
uu = np.load(os.path.join(basepath, 'uu.npy'))
ff = np.load(os.path.join(basepath, 'ff.npy'))
ww = np.load(os.path.join(basepath, 'ww.npy'))
tt0 = np.load(os.path.join(basepath, 'tt0.npy'))
mm = np.load(os.path.join(basepath, 'mm.npy'))

D = float(np.load(os.path.join(basepath, 'D.npy')))
P = float(np.load(os.path.join(basepath, 'P.npy')))

jj = np.array([0, 1, 3, 4])   # the index of v for the constraint of "vj-ke<=0"
KK = KK[:jj.shape[0]]

v = np.zeros(len(ll))
p = D*np.ones(len(tt0))
e = D*np.ones(CC.shape[1])
t = D*np.ones(len(tt0))
a = np.zeros(RR.shape[1])

variable = Variable(v, t, e, a)
parameter = Parameter(D, P, jj, ll, tt0, uu, ww, mm, CC, KK, RR, SS)

# print("======== only nonlinear and linear constraints ==========")
# _, info = optimizer(ff, variable, parameter)
# print(info)

print("======== only with linear constraints ==========")
_, info = optimizer(ff, variable, parameter)
print(f"info: ")



