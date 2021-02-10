import numpy as np
import ipopt

MIN = -2.0e19
MAX = 2.0e19


class Parameter:
    def __init__(self, D, P, jj, ll, tt0, uu, ww, mm, CC, KK, RR, SS):
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
        v = x[:self.var.m][self.para.vj]
        t = x[self.var.start_idx[0]: self.var.start_idx[1]]
        e = x[self.var.start_idx[1]: self.var.start_idx[2]]
        a = x[self.var.start_idx[2]:]

        p = np.matmul(self.para.CC, e)

        return np.concatenate([
            v - np.matmul(self.para.KK, e),
            np.matmul(self.para.SS, v),
            [np.dot(self.para.mm, p)],
            np.divide(p, self.para.ww) - t,
            np.multiply(self.para.tt0, np.power(2, np.matmul(self.para.RR, a))) - t,
        ])

    def jacobianstructure(self):
        j, m, s, l, n, k = self.lj, self.var.m, self.para.s, self.var.l, self.var.n,self.var.k
        r_sidx, c_sidx = self.c_start_idx, self.var.start_idx
        n_r, n_c = r_sidx[-1] + l, c_sidx + k
        s = np.zeros((n_r, n_c))
        s[:r_sidx[1], :c_sidx[0]] = np.ones((j+s, m))

        # // TODO
        s[r_sidx[2]: r_sidx[3], c_sidx[0]: c_sidx[1]] = np.ones((l, l))



    def jacobian(self, x):
        """
        The Jacobian of the constraints with shape (|constraints|, |x|)
        """

        j1 = np.concatenate([
            np.ones((self.var.m, self.var.m)),        # dv
            np.zeros((self.var.m, self.var.l)),       # dt
            np.negative(self.para.KK),                # de
            np.zeros((self.var.m, self.var.k))        # da
        ], axis=1)

        j2 = np.concatenate([
            self.para.SS,                                                  # dv
            np.zeros((self.para.s, self.var.l + self.var.n + self.var.k))  # dt, de, da
        ], axis=1)

        j3 = np.concatenate([
            np.zeros((1, self.var.m + self.var.l)),                             # dv, dt
            np.dot(self.para.mm, self.para.CC),                              # de
            np.zeros((1, self.var.k))    # da
        ], axis=1)

        j4 = np.concatenate([
            np.zeros((self.var.l, self.var.m)),                   # dv
            np.negative(np.ones((self.var.l, self.var.l))),       # dp
            np.zeros((self.var.l, self.var.l)),                   # dt
            self.para.CC,                                         # de
            np.zeros((self.var.l, self.var.k))                    # da
        ], axis=1)

        j5 = np.concatenate([
            np.zeros((self.var.l, self.var.m)),                   # dv
            np.multiply((-1/self.para.ww).reshape((-1, 1)), np.ones((self.var.l, self.var.l))),      # dp
            np.ones((self.var.l, self.var.l)),                    # dt
            np.zeros((self.var.l, self.var.n + self.var.k))       # de, da
        ], axis=1)

        tmp = np.multiply(-1*np.log(2)*np.power(2, np.matmul(self.para.RR, x[self.var.start_idx[3]:])), self.para.tt0)
        j6 = np.concatenate([
            np.zeros((self.var.l, self.var.m + self.var.l)),      # dv, dp
            np.ones((self.var.l, self.var.l)),                    # dt
            np.zeros((self.var.l, self.var.n)),                   # de
            np.multiply(tmp.reshape((-1, 1)), self.para.RR)       # da
        ], axis=1)

        return np.concatenate([j1, j2, j3, j4, j5, j6], axis=0).flatten()

    def hessianstructure(self):
        """
        the structure of the hessian is of a lower triangular matrix.
        """
        return np.nonzero(np.tril(np.ones((self.var.len_x, self.var.len_x))))

    def hessian(self, x, lagrange, obj_factor):
        """
        the hessian of the lagrangian
        :param lagrange: 1d-array with length of |j6|.shape[0]
        """
        dim = self.var.len_x - self.var.k
        H = -1 * np.power(np.log(2), 2) * np.multiply(self.para.tt0, np.matmul(self.para.RR, x[self.var.start_idx[3]:]))
        H = np.multiply(lagrange[-self.var.l:], H)
        H = np.multiply(H.reshape((-1, 1)), self.para.RR)
        H = np.matmul(H.T, self.para.RR)
        H = np.concatenate([np.zeros((self.var.k, dim)), H], axis=1)
        H = np.concatenate([np.zeros((dim, self.var.len_x)), H], axis=0)

        row, col = self.hessianstructure()

        return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trial):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def optimizer(ff: np.array, var: Variable, para: Parameter):
    lb = np.concatenate(
        (para.ll, [MIN] * var.l, [para.D] * var.l, [MIN] * (var.n + var.k)))
    ub = np.concatenate([para.uu, [MAX] * (2 * var.l + var.n + var.k)])

    cl = np.concatenate(
        [[MIN] * var.m, np.zeros(para.s + 1 + 3 * var.l)])
    cu = np.concatenate(
        [np.zeros(var.m + para.s), [para.P], np.zeros(3 * var.l)])

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
    p = np.ones(l)
    t = np.ones(l)
    e = np.ones(n)
    a = np.ones(k)
    ll = np.ones(m) * (-10)
    uu = np.ones(m) * 10
    KK = np.abs(np.random.randn(m, n))
    s = 13
    SS = np.abs(np.random.randn(s, m))
    P = 100
    CC = np.abs(np.random.randn(l, n))
    ww = np.abs(np.random.randn(l))
    D = 0.01
    RR = np.abs(np.random.randn(l, k))
    tt0 = np.abs(np.random.randn(l))
    ff = np.ones(m)
    variable = Variable(v, p, t, e, a)
    parameter = Parameter(D, P, ll, tt0, uu, ww, CC, KK, RR, SS)
    x, info = optimizer(ff, variable, parameter)

test()

# D, P, ll, tt0, uu, ww, CC, KK, RR, SS = None, None, None, None, None, None, None, None, None, None
# v, p, t, e, a = None, None, None, None, None
# ff = None
#
# variable = Variable(v, p, t, e, a)
# parameter = Parameter(D, P, ll, tt0, uu, ww, CC, KK, RR, SS)
# x, info = optimizer(ff, variable, parameter)
