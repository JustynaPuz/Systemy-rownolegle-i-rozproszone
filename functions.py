import numpy as np
import math
# ==============================================================================
# Functions
# ==============================================================================

def f1_quadratic(x):
    n = len(x)
    val = 0.0
    for i in range(2, n): # Loop i from 2 to n-1 (corresponds to i=3..n in 1-based)
        val += 100 * (x[i]**2 + x[i-1]**2) + x[i-2]**2
    return val

def f2_woods(x):
    n = len(x)
    val = 0.0
    for i in range(1, (n // 4) + 1):
        j = 4 * i - 4
        t1 = 100 * (x[j+1] - x[j]**2)**2
        t2 = (1 - x[j])**2
        t3 = 90 * (x[j+3] - x[j+2]**2)**2
        t4 = (1 - x[j+2])**2
        t5 = 10 * (x[j+1] + x[j+3] - 2)**2
        t6 = 0.1 * (x[j+1] - x[j+3])**2
        val += t1 + t2 + t3 + t4 + t5 + t6
    return val

def f3_powell_singular(x):
    n = len(x)
    val = 0.0
    for i in range(1, (n // 4) + 1):
        j = 4 * i - 4
        val += (x[j] + 10 * x[j+1])**2
        val += 5 * (x[j+2] - x[j+3])**2
        val += (x[j+1] - 2 * x[j+2])**4
        val += 10 * (x[j] - x[j+3])**4
    return val

def f4_extended_rosenbrock(x):
    n = len(x)
    val = 0.0
    for i in range(1, (n // 2) + 1):
        j = 2 * i - 2
        val += 100 * (x[j+1] - x[j]**2)**2 + (1 - x[j])**2
    return val

def f5_generalized_rosenbrock(x):
    n = len(x)
    val = 1.0
    for i in range(1, n):
        val += 100 * (x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2
    return val

def f6_shanno(x):
    n = len(x)
    val = (2 * x - 1)**2
    for i in range(1, n):
        val += (i + 1) * (2 * x[i-1] - x[i])**2
    return val

def f7_engval(x):
    n = len(x)
    val = 0.0
    for i in range(1, n):
        val += (x[i-1]**2 + x[i]**2)**2 - 4 * x[i-1] + 3
    return val

def f8_cragg_levy(x):
    n = len(x)
    val = 0.0
    for i in range(1, (n // 2) + 1):
        j = 2 * i - 2
        x_odd = x[j]
        x_even = x[j+1]
        
        term1 = (np.exp(x_odd) - x_even)**4
        term2 = 100 * (x_even - x_odd)**6
        term3 = (math.tan(x_odd - x_even))**4
        term4 = x_odd**8
        term5 = (x_even - 1)**2
        val += term1 + term2 + term3 + term4 + term5
    return val

def f9_freudenstein_roth(x):
    n = len(x)
    val = 0.0
    for i in range(1, (n // 2) + 1):
        j = 2 * i - 2
        x_odd = x[j]
        x_even = x[j+1]
        
        term1 = (-13 + x_odd + ((5 - x_even) * x_even - 2) * x_even)**2
        term2 = (-29 + x_odd + ((x_even + 1) * x_even - 14) * x_even)**2
        val += term1 + term2
    return val

def f10_discrete_boundary(x):
    n = len(x)
    h = 1.0 / (n + 1)
    val = 0.0
    x_ext = np.concatenate(([0.0], x, [0.0]))
    
    for i in range(1, n + 1):
        t_i = i * h
        term = 2*x_ext[i] - x_ext[i-1] - x_ext[i+1] + (h**2 / 2.0) * (x_ext[i] + t_i + 1)**3
        val += term**2
    return val

def f11_broyden_tridiagonal(x):
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    val = 0.0
    for i in range(1, n + 1):
        term = (3 - 2 * x_ext[i]) * x_ext[i] - x_ext[i-1] - 2 * x_ext[i+1] + 1
        val += term**2
    return val

def f12_arrowhead(x):
    n = len(x)
    val = 0.0
    for i in range(n - 1):
        val += (x[i]**2 + x[n-1]**2)**2 - 4 * x[i] + 3
    return val

def f13_nondiagonal_quartic(x):
    n = len(x)
    val = (x - x[10])**2
    for i in range(1, n - 1):
        term1 = (x[i-1] + x[i] + x[n-1])**4
        term2 = (x[i-1] - x[i])**2
        val += term1 + term2
    return val

def f14_banded_quartic(x):
    n = len(x)
    val = 0.0
    for i in range(3, n - 1):
        term = x[i-3] + 2*x[i-2] + 3*x[i-1] + 4*x[i] + 5*x[n-1]
        val += term**2 - 4*x[i-3] + 3
    return val

def f15_penalty(x):
    n = len(x)
    term1 = 0.0
    for i in range(n):
        term1 += (x[i] - 1)**2
    
    term2 = 0.0
    for j in range(n):
        term2 += x[j]**2
    term2 = (term2 - 0.25)**2
    
    return 1e-5 * term1 + term2

def f16_trigonometric(x):
    n = len(x)
    val = 0.0
    sum_cos = np.sum(np.cos(x))
    
    for i in range(n):
        idx_pdf = i + 1
        term = n - sum_cos + idx_pdf * (1 - math.cos(x[i])) - math.sin(x[i])
        val += term**2
    return val

def f17_quartic(x):
    n = len(x)
    inner_sum = 0.0
    for i in range(n):
        idx_pdf = i + 1
        inner_sum += idx_pdf * x[i]**2
    return inner_sum**2


def get_function_setup(func_id, n_dim):
    """
    Zwraca krotkę (funkcja, punkt_startowy_x0) dla danego ID i wymiaru.
    Dla niektórych funkcji PDF definiuje specyficzne x0.
    """
    if func_id == 1:
        return f1_quadratic, np.full(n_dim, 3.0)
    elif func_id == 2:
        x0 = np.array([-3.0, -1.0] * (n_dim // 2))
        return f2_woods, x0
    elif func_id == 3:
        x0 = np.array([3.0, -1.0, 0.0, 1.0] * (n_dim // 4))
        return f3_powell_singular, x0
    elif func_id == 4:
        x0 = np.array([-1.2, 1.0] * (n_dim // 2))
        return f4_extended_rosenbrock, x0
    elif func_id == 5:
        return f5_generalized_rosenbrock, np.full(n_dim, 1.0 / (n_dim + 1))
    elif func_id == 6:
        return f6_shanno, np.full(n_dim, -1.0)
    elif func_id == 7:
        return f7_engval, np.full(n_dim, 2.0)
    elif func_id == 8:
        return f8_cragg_levy, np.full(n_dim, 2.0)
    elif func_id == 9:
        x0 = np.zeros(n_dim)
        x0 = 0.5
        x0[10] = -2.0
        return f9_freudenstein_roth, x0
    elif func_id == 10:
        x0 = np.zeros(n_dim)
        h = 1.0 / (n_dim + 1)
        for k in range(n_dim):
            tk = (k + 1) * h
            x0[k] = tk * (tk - 1)
        return f10_discrete_boundary, x0
    elif func_id == 11:
        return f11_broyden_tridiagonal, np.full(n_dim, -1.0)
    elif func_id == 12:
        return f12_arrowhead, np.full(n_dim, 1.0)
    elif func_id == 13:
        x0 = np.array([1.0, -1.0] * (n_dim // 2))
        if len(x0) < n_dim: x0 = np.append(x0, 1.0)
        return f13_nondiagonal_quartic, x0[:n_dim]
    elif func_id == 14:
        return f14_banded_quartic, np.full(n_dim, 1.0)
    elif func_id == 15:
        x0 = np.arange(1, n_dim + 1, dtype=float)
        return f15_penalty, x0
    elif func_id == 16:
        return f16_trigonometric, np.full(n_dim, 1.0)
    elif func_id == 17:
        return f17_quartic, np.full(n_dim, 1.0)
    
    # In case of unknown func_id, default to f1
    return f1_quadratic, np.full(n_dim, 3.0)