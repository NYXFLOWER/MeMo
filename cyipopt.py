# encoding: utf-8
# module cyipopt
# from /Users/nyxfer/anaconda3/envs/mm/lib/python3.8/site-packages/cyipopt.cpython-38-darwin.so
# by generator 1.147
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2020 cyipopt developers

Author: Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>
(original Author: Amit Aides <amitibo@tx.technion.ac.il>)
URL: https://github.com/matthias-k/cyipopt
License: EPL 1.0
"""

# imports
import builtins as __builtins__  # <module 'builtins' (built-in)>
import \
    numpy as np  # /Users/nyxfer/anaconda3/envs/mm/lib/python3.8/site-packages/numpy/__init__.py
import \
    logging as logging  # /Users/nyxfer/anaconda3/envs/mm/lib/python3.8/logging/__init__.py
import sys as sys  # <module 'sys' (built-in)>
import \
    six as six  # /Applications/PyCharm.app/Contents/plugins/python/helpers/six.py
import numpy as __numpy

# Variables with simple values

CREATE_PROBLEM_MSG = '\n----------------------------------------------------\nCreating Ipopt problem with the following parameters\nn = %s\nm = %s\njacobian elements num = %s\nhessian elements num = %s\n'

INF = 10000000000000000000


# functions

def setLoggingLevel(*args, **kwargs):  # real signature unknown
    pass


# classes

class DTYPEd(__numpy.floating, float):
    """
    Double-precision floating-point number type, compatible with Python `float`
        and C ``double``.
        Character code: ``'d'``.
        Canonical name: ``np.double``.
        Alias: ``np.float_``.
        Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.
    """

    def as_integer_ratio(self):  # real signature unknown; restored from __doc__
        """
        double.as_integer_ratio() -> (int, int)

                Return a pair of integers, whose ratio is exactly equal to the original
                floating point number, and with a positive denominator.
                Raise OverflowError on infinities and a ValueError on NaNs.

                >>> np.double(10.0).as_integer_ratio()
                (10, 1)
                >>> np.double(0.0).as_integer_ratio()
                (0, 1)
                >>> np.double(-.25).as_integer_ratio()
                (-1, 4)
        """
        pass

    def __abs__(self, *args, **kwargs):  # real signature unknown
        """ abs(self) """
        pass

    def __add__(self, *args, **kwargs):  # real signature unknown
        """ Return self+value. """
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        pass

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(self, value). """
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        pass

    def __float__(self, *args, **kwargs):  # real signature unknown
        """ float(self) """
        pass

    def __floordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return self//value. """
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        pass

    def __hash__(self, *args, **kwargs):  # real signature unknown
        """ Return hash(self). """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        pass

    def __mul__(self, *args, **kwargs):  # real signature unknown
        """ Return self*value. """
        pass

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        pass

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        pass

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    def __sub__(self, *args, **kwargs):  # real signature unknown
        """ Return self-value. """
        pass

    def __truediv__(self, *args, **kwargs):  # real signature unknown
        """ Return self/value. """
        pass


class DTYPEi(__numpy.signedinteger):
    """
    Signed integer type, compatible with C ``int``.
        Character code: ``'i'``.
        Canonical name: ``np.intc``.
        Alias *on this platform*: ``np.int32``: 32-bit signed integer (-2147483648 to 2147483647).
    """

    def __abs__(self, *args, **kwargs):  # real signature unknown
        """ abs(self) """
        pass

    def __add__(self, *args, **kwargs):  # real signature unknown
        """ Return self+value. """
        pass

    def __and__(self, *args, **kwargs):  # real signature unknown
        """ Return self&value. """
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        pass

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(self, value). """
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        pass

    def __float__(self, *args, **kwargs):  # real signature unknown
        """ float(self) """
        pass

    def __floordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return self//value. """
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        pass

    def __hash__(self, *args, **kwargs):  # real signature unknown
        """ Return hash(self). """
        pass

    def __index__(self, *args, **kwargs):  # real signature unknown
        """ Return self converted to an integer, if self is suitable for use as an index into a list. """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        pass

    def __invert__(self, *args, **kwargs):  # real signature unknown
        """ ~self """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self<<value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        pass

    def __mul__(self, *args, **kwargs):  # real signature unknown
        """ Return self*value. """
        pass

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __or__(self, *args, **kwargs):  # real signature unknown
        """ Return self|value. """
        pass

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        pass

    def __rand__(self, *args, **kwargs):  # real signature unknown
        """ Return value&self. """
        pass

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        pass

    def __rlshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value<<self. """
        pass

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __ror__(self, *args, **kwargs):  # real signature unknown
        """ Return value|self. """
        pass

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rrshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value>>self. """
        pass

    def __rshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self>>value. """
        pass

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        pass

    def __rxor__(self, *args, **kwargs):  # real signature unknown
        """ Return value^self. """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    def __sub__(self, *args, **kwargs):  # real signature unknown
        """ Return self-value. """
        pass

    def __truediv__(self, *args, **kwargs):  # real signature unknown
        """ Return self/value. """
        pass

    def __xor__(self, *args, **kwargs):  # real signature unknown
        """ Return self^value. """
        pass


class problem(object):
    """
    Wrapper class for solving optimization problems using the C interface of
        the Ipopt package.

        It can be used to solve general nonlinear programming problems of the form:

        .. math::

               \min_ {x \in R^n} f(x)

        subject to

        .. math::

               g_L \leq g(x) \leq g_U

               x_L \leq  x  \leq x_U

        Where :math:`x` are the optimization variables (possibly with upper an
        lower bounds), :math:`f(x)` is the objective function and :math:`g(x)` are
        the general nonlinear constraints. The constraints, :math:`g(x)`, have
        lower and upper bounds. Note that equality constraints can be specified by
        setting :math:`g^i_L = g^i_U`.

        Parameters
        ----------
        n : integer
            Number of primal variables.
        m : integer
            Number of constraints.
        problem_obj: object, optional (default=None)
            An object holding the problem's callbacks. If None, cyipopt will use
            self, this is useful when subclassing problem. The object is required
            to have the following attributes (some are optional):

                - 'objective' : function pointer
                    Callback function for evaluating objective function. The
                    callback functions accepts one parameter: x (value of the
                    optimization variables at which the objective is to be
                    evaluated). The function should return the objective function
                    value at the point x.
                - 'constraints' : function pointer
                    Callback function for evaluating constraint functions. The
                    callback functions accepts one parameter: x (value of the
                    optimization variables at which the constraints are to be
                    evaluated). The function should return the constraints values
                    at the point x.
                - 'gradient' : function pointer
                    Callback function for evaluating gradient of objective
                    function. The callback functions accepts one parameter: x
                    (value of the optimization variables at which the gradient is
                    to be evaluated). The function should return the gradient of
                    the objective function at the point x.
                - 'jacobian' : function pointer
                    Callback function for evaluating Jacobian of constraint
                    functions. The callback functions accepts one parameter: x
                    (value of the optimization variables at which the jacobian is
                    to be evaluated). The function should return the values of the
                    jacobian as calculated using x. The values should be returned
                    as a 1-dim numpy array (using the same order as you used when
                    specifying the sparsity structure)
                - 'jacobianstructure' : function pointer, optional (default=None)
                    Callback function that accepts no parameters and returns the
                    sparsity structure of the Jacobian (the row and column indices
                    only). If None, the Jacobian is assumed to be dense.
                - 'hessian' : function pointer, optional (default=None)
                    Callback function for evaluating Hessian of the Lagrangian
                    function. The callback functions accepts three parameters x
                    (value of the optimization variables at which the hessian is to
                    be evaluated), lambda (values for the constraint multipliers at
                    which the hessian is to be evaluated) objective_factor the
                    factor in front of the objective term in the Hessian. The
                    function should return the values of the Hessian as calculated
                    using x, lambda and objective_factor. The values should be
                    returned as a 1-dim numpy array (using the same order as you
                    used when specifying the sparsity structure). If None, the
                    Hessian is calculated numerically.
                - 'hessianstructure' : function pointer, optional (default=None)
                    Callback function that accepts no parameters and returns the
                    sparsity structure of the Hessian of the lagrangian (the row
                    and column indices only). If None, the Hessian is assumed to be
                    dense.
                - 'intermediate' : function pointer, optional (default=None)
                    Optional. Callback function that is called once per iteration
                    (during the convergence check), and can be used to obtain
                    information about the optimization status while Ipopt solves
                    the problem. If this callback returns False, Ipopt will
                    terminate with the User_Requested_Stop status. The information
                    below corresponeds to the argument list passed to this
                    callback:

                        'alg_mod':
                            Algorithm phase: 0 is for regular, 1 is restoration.
                        'iter_count':
                            The current iteration count.
                        'obj_value':
                            The unscaled objective value at the current point
                        'inf_pr':
                            The scaled primal infeasibility at the current point.
                        'inf_du':
                            The scaled dual infeasibility at the current point.
                        'mu':
                            The value of the barrier parameter.
                        'd_norm':
                            The infinity norm (max) of the primal step.
                        'regularization_size':
                            The value of the regularization term for the Hessian
                            of the Lagrangian in the augmented system.
                        'alpha_du':
                            The stepsize for the dual variables.
                        'alpha_pr':
                            The stepsize for the primal variables.
                        'ls_trials':
                            The number of backtracking line search steps.

                    more information can be found in the following link:
                    http://www.coin-or.org/Ipopt/documentation/node56.html#sec:output

        lb : array-like, shape(n, )
            Lower bounds on variables, where n is the dimension of x.
            To assume no lower bounds pass values lower then 10^-19.
        ub : array-like, shape(n, )
            Upper bounds on variables, where n is the dimension of x..
            To assume no upper bounds pass values higher then 10^-19.
        cl : array-like, shape(m, )
            Lower bounds on constraints, where m is the number of constraints.
            Equality constraints can be specified by setting cl[i] = cu[i].
        cu : array-like, shape(m, )
            Upper bounds on constraints, where m is the number of constraints.
            Equality constraints can be specified by setting cl[i] = cu[i].
    """

    def addOption(self, *args, **kwargs):  # real signature unknown
        """
        Add a keyword/value option pair to the problem. See the Ipopt
                documentaion for details on available options.

                Parameters
                ----------
                keyword : string
                    Option name.

                val : string, int, or float
                    Value of the option. The type of val should match the option
                    definition as described in the Ipopt documentation.

                Returns
                -------
                    None
        """
        pass

    def close(self, *args, **kwargs):  # real signature unknown
        """
        Deallcate memory resources used by the Ipopt package. Called implicitly
                by the 'problem' class destructor.

                Parameters
                ----------
                    None

                Returns
                -------
                    None
        """
        pass

    def setProblemScaling(self, *args, **kwargs):  # real signature unknown
        """
        Optional function for setting scaling parameters for the problem.
                To use the scaling parameters set the option 'nlp_scaling_method' to
                'user-scaling'.

                Parameters
                ----------
                obj_scaling : float
                    Determines, how Ipopt should internally scale the objective
                    function.  For example, if this number is chosen to be 10, then
                    Ipopt solves internally an optimization problem that has 10 times
                    the value of the original objective. In particular, if this value
                    is negative, then Ipopt will maximize the objective function
                    instead of minimizing it.
                x_scaling : array-like, shape(n, )
                    The scaling factors for the variables. If None, no scaling is done.
                g_scaling : array-like, shape(m, )
                    The scaling factors for the constrains. If None, no scaling is done.

                Returns
                -------
                    None
        """
        pass

    def solve(self, *args, **kwargs):  # real signature unknown
        """
        Returns the optimal solution and an info dictionary. Solves the posed
                optimization problem starting at point x.

                Parameters
                ----------
                x : array-like, shape(n, )
                    Initial guess.

                Returns
                -------
                x : array, shape(n, )
                    Optimal solution.
                info: dictionary
                    'x': ndarray, shape(n, )
                        optimal solution
                    'g': ndarray, shape(m, )
                        constraints at the optimal solution
                    'obj_val': float
                        objective value at optimal solution
                    'mult_g': ndarray, shape(m, )
                        final values of the constraint multipliers
                    'mult_x_L': ndarray, shape(n, )
                        bound multipliers at the solution
                    'mult_x_U': ndarray, shape(n, )
                        bound multipliers at the solution
                    'status': integer
                        gives the status of the algorithm
                    'status_msg': string
                        gives the status of the algorithm as a message
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    __constraints = property(lambda self: object(), lambda self, v: None,
                             lambda self: None)  # default

    __exception = property(lambda self: object(), lambda self, v: None,
                           lambda self: None)  # default

    __gradient = property(lambda self: object(), lambda self, v: None,
                          lambda self: None)  # default

    __hessian = property(lambda self: object(), lambda self, v: None,
                         lambda self: None)  # default

    __hessianstructure = property(lambda self: object(), lambda self, v: None,
                                  lambda self: None)  # default

    __intermediate = property(lambda self: object(), lambda self, v: None,
                              lambda self: None)  # default

    __jacobian = property(lambda self: object(), lambda self, v: None,
                          lambda self: None)  # default

    __jacobianstructure = property(lambda self: object(), lambda self, v: None,
                                   lambda self: None)  # default

    __m = property(lambda self: object(), lambda self, v: None,
                   lambda self: None)  # default

    __n = property(lambda self: object(), lambda self, v: None,
                   lambda self: None)  # default

    __objective = property(lambda self: object(), lambda self, v: None,
                           lambda self: None)  # default


# variables with complex values

STATUS_MESSAGES = {
    -199: b'An unknown internal error occurred. Please contact the Ipopt authors through the mailing list.',
    -102: b'Not enough memory.',
    -101: b'Unknown Exception caught in Ipopt',
    -100: b'Some uncaught Ipopt exception encountered.',
    -13: b'Algorithm received an invalid number (such as NaN or Inf) from the NLP; see also option check_derivatives_for_naninf',
    -12: b'Invalid option encountered.',
    -11: b'Invalid problem definition.',
    -10: b'Problem has too few degrees of freedom.',
    -4: b'Maximum CPU time exceeded.',
    -3: b'An unrecoverable error occurred while Ipopt tried to compute the search direction.',
    -2: b"Restoration phase failed, algorithm doesn't know how to proceed.",
    -1: b'Maximum number of iterations exceeded (can be specified by an option).',
    0: b'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).',
    1: b'Algorithm stopped at a point that was converged, not to "desired" tolerances, but to "acceptable" tolerances (see the acceptable-... options).',
    2: b'Algorithm converged to a point of local infeasibility. Problem may be infeasible.',
    3: b'Algorithm proceeds with very little progress.',
    4: b'It seems that the iterates diverge.',
    5: b'The user call-back function intermediate_callback (see Section 3.3.4 in the documentation) returned false, i.e., the user code requested a premature termination of the optimization.',
    6: b'Feasible point for square problem found.',
}

__all__ = [
    'setLoggingLevel',
    'problem',
]

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7ffbd5ad0fd0>'

__spec__ = None  # (!) real value is "ModuleSpec(name='cyipopt', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7ffbd5ad0fd0>, origin='/Users/nyxfer/anaconda3/envs/mm/lib/python3.8/site-packages/cyipopt.cpython-38-darwin.so')"

__test__ = {}

