# # import numpy as np
# # import ipopt
# #
# # m, l, n, k = 3, 5, 7, 11
# # v, p, t = np.random.randn(m), np.random.randn(l), np.abs(np.random.randn(l))
# # e, a = np.random.randn(n), np.random.randn(k)
# #
# #
# # ll = np.ones(m) * (-10)
# # uu = np.ones(m) * 10
# # KK = np.abs(np.random.randn(m, n))
# #
# # s = 13
# # SS = np.abs(np.random.randn(s, m))
# # P = 100
# # CC = np.abs(np.random.randn(l, n))
# # ww = np.abs(np.random.randn(l))
# # D = 0.01
# # RR = np.abs(np.random.randn(l, k))
# # tt0 = np.abs(np.random.randn(l))
# #
# # ff = np.ones(m)
# #
# #
# # start_idx = [m, m + l, m + 2 * l, m + 2 * l + n]
# #
# #
# # class MeMo:
# #     def __init__(self, f, m, l, n, k):
# #         """
# #         :param f: array with the same length as the input 'x'
# #         """
# #         self.f = np.concatenate([f, np.zeros(2 * l + n + k)])
# #         self.m,  self.l, self.n, self.k = m, l, n, k
# #         self.len_x = m + 2*l + n + k
# #
# #     def objective(self, x):
# #         return np.dot(self.f, x)
# #
# #     def gradient(self, x):
# #         """
# #         The gradient of the objective function
# #         """
# #         return self.f
# #
# #     def constraints(self, x):
# #         """
# #         The 'm' + 's' + 1 + 2*'l' linear and 'l' nonlinear constraints
# #         """
# #         return np.concatenate([
# #             np.add(x[:m], np.matmul(KK, x[start_idx[2]: start_idx[3]])),
# #             np.matmul(SS, x[:m]),
# #             [np.sum(x[m: start_idx[1]])],
# #             np.matmul(CC, x[start_idx[2]: start_idx[3]]) - x[m: start_idx[1]],
# #             # np.divide(x[m: start_idx[1]], x[start_idx[1]: start_idx[2]]),
# #             x[start_idx[1]: start_idx[2]] - np.divide(x[m: start_idx[1]], ww),
# #             # np.log2(np.divide(x[start_idx[1]: start_idx[2]], tt0)) - np.matmul(RR, a),
# #             x[start_idx[1]: start_idx[2]] - np.multiply(tt0, np.power(2, np.matmul(RR, a)))
# #         ])
# #
# #     def jacobian(self, x):
# #         """
# #         The Jacobian of the constraints with shape (|constraints|, |x|)
# #         """
# #         j1 = np.concatenate([
# #             np.ones((m, m)),        # dv
# #             np.zeros((m, 2 * l)),     # dp, dt
# #             np.negative(KK),        # de
# #             np.zeros((m, k))        # da
# #         ], axis=1)
# #
# #         j2 = np.concatenate([
# #             SS,                     # dv
# #             np.zeros((s, 2 * l + n + k))  # dp, dt, de, da
# #         ], axis=1)
# #
# #         j3 = np.concatenate([
# #             np.zeros((1, m)),       # dv
# #             np.ones((1, l)),        # dp
# #             np.zeros((1, l + n + k))    # dt, de, da
# #         ], axis=1)
# #
# #         j4 = np.concatenate([
# #             np.zeros((l, m)),                   # dv
# #             np.negative(np.ones((l, l))),       # dp
# #             np.zeros((l, l)),                   # dt
# #             CC,                                 # de
# #             np.zeros((l, k))                    # da
# #         ], axis=1)
# #
# #         j5 = np.concatenate([
# #             np.zeros((l, m)),                   # dv
# #             np.multiply((-1/ww).reshape((-1, 1)), np.ones((l, l))),      # dp
# #             np.ones((l, l)),                    # dt
# #             np.zeros((l, n + k))                  # de, da
# #         ], axis=1)
# #
# #         tmp = np.multiply(-1*np.log(2)*np.power(2, np.matmul(RR, x[start_idx[3]:])), tt0)
# #         j6 = np.concatenate([
# #             np.zeros((l, m + l)),                 # dv, dp
# #             np.ones((l, l)),                    # dt
# #             np.zeros((l, n)),                   # de
# #             np.multiply(tmp.reshape((-1, 1)), RR)       # da
# #         ], axis=1)
# #
# #         return np.concatenate([j1, j2, j3, j4, j5, j6], axis=0).flatten()
# #
# #     def hessianstructure(self):
# #         """
# #         the structure of the hessian is of a lower triangular matrix.
# #         """
# #         return np.nonzero(np.tril(np.ones((self.len_x, self.len_x))))
# #
# #     def hessian(self, x, lagrange, obj_factor):
# #         """
# #         the hessian of the lagrangian
# #         :param lagrange: 1d-array with length of |j6|.shape[0]
# #         """
# #         dim = self.len_x - k
# #         H = -1 * np.power(np.log(2), 2) * np.multiply(tt0, np.matmul(RR, x[start_idx[3]:]))
# #         H = np.multiply(lagrange[-l:], H)
# #         H = np.multiply(H.reshape((-1, 1)), RR)
# #         H = np.matmul(H.T, RR)
# #         H = np.concatenate([np.zeros((k, dim)), H], axis=1)
# #         H = np.concatenate([np.zeros((dim, self.len_x)), H], axis=0)
# #
# #         row, col = self.hessianstructure()
# #
# #         return H[row, col]
# #
# #     def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
# #                      d_norm, regularization_size, alpha_du, alpha_pr, ls_trial):
# #         print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
# #
# #
# #
# #
# #
# #
# #
# # MIN = -2.0e19
# # MAX = 2.0e19
# #
# # # x0 = np.concatenate([v, p, t, e, a])
# # x0 = np.ones(31)
# #
# # lb = np.concatenate([ll, [MIN] * l, [D] * l, [MIN] * (n + k)])
# # ub = np.concatenate([uu, [MAX] * (2 * l + n + k)])
# #
# # cl = np.concatenate([[MIN] * m, np.zeros(s + 1 + 3 * l)])
# # cu = np.concatenate([np.zeros(m + s), [P], np.zeros(3 * l)])
#
# nlp = ipopt.problem(
#     n=x0.shape[0],
#     m=len(cl),
#     problem_obj=MeMo(ff, m, l, n, k),
#     lb=lb,
#     ub=ub,
#     cl=cl,
#     cu=cu
# )
#
# #
# # Set solver options
# #
# # nlp.addOption('derivative_test', 'second-order')
# nlp.addOption('mu_strategy', 'adaptive')
# nlp.addOption('tol', 1e-7)
#
# #
# # Scale the problem (Just for demonstration purposes)
# #
# nlp.setProblemScaling(
#     obj_scaling=2,
#     x_scaling=[1]*x0.shape[0]
# )
# nlp.addOption('nlp_scaling_method', 'user-scaling')
#
# #
# # Solve the problem
# #
# x, info = nlp.solve(x0)
#
# print("Solution of the primal variables: x=%s\n" % repr(x))
#
# print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
#
# print("Objective=%s\n" % repr(info['obj_val']))
