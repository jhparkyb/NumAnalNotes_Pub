import numpy as np
from scipy.linalg import toeplitz

def secant(f, ini1, ini2, tol=1e-8, max_iter=100):
    """
    Return an approximate root of a function using Secant's method.

    INPUT
        f: function whose zero is sought.
        ini1, ini2: initial guesses
        tol: tolerance for stopping criterion. If consecutive iterates differ by less than this, it is considered convergenct.
        max_iter: maximum number of iterations
    OUTPU
        approximated zero and the number of iterations. When the maximum number of iterations is reached, the last iterate with a warning message.
    """
    x1 = ini1
    x2 = ini2
    f1 = f(x1)
    for i in range(max_iter):
        f2 = f(x2)
        x_tmp = x2
        x2 = x2 - f(x2)*(x2 - x1)/(f2 - f1)
        
        # update history
        x1 = x_tmp
        f1 = f2
        if np.abs(x2 - x1) < tol: 
            break
    """
    if i == max_iter - 1:
        print("   Warning (Secent method): maximum number of iteration reached.\n     --> The output may not be close enough to the zero.")
    """
    return x2, i + 1

def RK4(F, x0, t0, T, K):
    """
    Return numerical solution of ODE x'=F(x) and time grid.

    INPUT
        F: slope function
        x0: initial condition
        t0: initial time
        T: final time
        K: number of time steps
    OUTPUT
        sol, tgrid: numerical solution and time grid
    """
    dd = x0.size
    tgrid = np.linspace(t0, T, K+1)
    xx = np.zeros((K+1, dd))
    xx[0] = x0

    for k in range(K):
        # time step
        tau = tgrid[k+1] - tgrid[k]
        x = xx[k]
        t = tgrid[k]

        # Runge-Kutta 4th order
        k1 = F(t, x)
        k2 = F(t, x + tau/2*k1)
        k3 = F(t, x + tau/2*k2)
        k4 = F(t, x + tau*k3)
        xx[k+1] = x + tau/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return xx, tgrid

def tridiag(sub_diag, diag, super_diag, n):
    """
    Return tridiagonal matrix.

    INPUT
        sub_diag: sub-diagonal elements
        diag: diagonal elements
        super_diag: super-diagonal elements
        n: size of the matrix
    OUTPUT
        A: tridiagonal matrix
    """
    col = np.zeros(n)
    col[0] = diag
    col[1] = sub_diag
    if sub_diag != super_diag:
        row = np.zeros(n)
        row[0] = diag
        row[1] = super_diag
    else:
        row = None
        
    A = toeplitz(col) if row != None else toeplitz(col, row)

    return A

def CG(A, b, x0, tol=1e-9, max_iter=None):
    """
    Conjugate Gradient method for solving linear systems of equations. The matrix must be symmetric and positive definite.
    
    Parameters:
        A (ndarray): The coefficient matrix of the linear system.
        b (ndarray): The right-hand side vector of the linear system.
        x0 (ndarray): The initial guess for the solution.
        max_iter (int): The maximum number of iterations (default: 100).
        tol (float): The tolerance for convergence (default: 1e-9).
    
    Returns:
        x (ndarray): The approximate solution to the linear system.
        i (int): The number of iterations performed.
    """
    # default number of iterations is the dimension of the matrix
    if max_iter is None:
        n = A.shape[0]
        max_iter = n

    # initialize
    x = x0
    r = b - A @ x
    d = r

    r_nrm2 = np.dot(r, r)

    for i in range(1, max_iter + 1): 
        # intermediate computations part 1 
        Ad = A @ d

        # main conjugate gradient iteration part 1
        alpha = r_nrm2 / np.dot(d, Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad
        
        # intermediate computations part 2
        r_new_nrm2 = np.dot(r_new, r_new)
        
        # stopping criterion
        if np.sqrt(r_new_nrm2) < tol:
            break

        # main conjugate gradient iteration part 2
        beta = r_new_nrm2 / r_nrm2
        d = r_new + beta * d
        
        # updata
        r = r_new
        r_nrm2 = r_new_nrm2
    
    return x, i

def poly_eval(a, x, algorithm='Horner'):
    """
    Evaluates a polynomial at a given point x.

    Inputs:
        a: 1D array of polynomial coefficients (ascending order). 
        x: 1D array of points at which to evaluate the polynomial.
        algorithm: algorithm to use for polynomial evaluation. (default: 'Horner')
    Output:
        p: array of polynomial values at x.
    """
    
    if algorithm == 'Horner':
        p = a[-1]*np.ones_like(x) # broadcasting is in effect 
        for i in range(len(a)-2, -1, -1):
            p = x*p + a[i]
    
    return p

def poly_eval_simplest(c, x, order='increasing'):
    """
    Evaluate a polynomial at a given point.

    Parameters:
        c (ndarray): The coefficients of the polynomial.
        x (ndarray): The point at which to evaluate the polynomial.
        order (str): The order of the coefficients (default: 'increasing').
    
    Returns:
        p (ndarray): The value of the polynomial at the point x.

    Note:
        This function implements the simplest translation of mathematical work.
        That is, it does not utilize Horner's meothod or any other optimization.
    """
    if c.ndim > 1:
        raise ValueError("The coefficients must be a 1D array.")
    
    n = c.shape[0]
    pow = np.arange(n) if order == 'increasing' else np.arange(n)[::-1]

    return np.sum(c * np.power(x[:, np.newaxis], pow), axis=1)