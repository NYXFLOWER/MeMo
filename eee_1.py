from __future__ import print_function, unicode_literals

import numpy as np
import ipopt


class hs071:

    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobianstructure(self):
        tmp = np.array([[1, 2, 3, 4], [1, 2, 3, 0]])
        return tmp.nonzero()

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        tmp = np.vstack([np.prod(x) / x, 2*x])

        return tmp.flatten()[:-1]

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        row, col = np.nonzero(np.tril(np.ones((4, 4))))
        return row[:-1], col[:-1]

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
        tmp = np.eye(4)
        tmp[3, 3] = 0
        H += lagrange[1]*2*tmp

        row, col = self.hessianstructure()

        return H[row, col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def main():
    #
    # Define the problem
    #
    x0 = [1.0, 5.0, 5.0, 1.0]

    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

    nlp = ipopt.problem(
        n=len(x0),
        m=len(cl),
        problem_obj=hs071(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
        )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=[1, 1, 1, 1]
        )
    nlp.addOption('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))


if __name__ == '__main__':
    main()