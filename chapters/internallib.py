import numpy as np

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