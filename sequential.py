import argparse


# alpha = 1.5, beta = 0.5, eps = 0.001, można nadpisać z CLI
def parse_args():
    ap = argparse.ArgumentParser(
        description="Sequential Nelder–Mead variant."
    )
    ap.add_argument("--func", choices=["woods", "ext_rosenbrock"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--print-x", action="store_true")
    ap.add_argument("--csv-out", type=str, default="")
    return ap.parse_args()

#---------------------
def woods(x):
    n = len(x)
    if n % 4 != 0:
        raise ValueError("Woods requires n divisible by 4.")
    s = 0.0
    for i in range(n // 4):
        a, b, c, d = x[4*i:4*i+4]
        s += 100.0 * (b - a*a)**2 + (1.0 - a)**2
        s += 90.0 * (d - c*c)**2 + (1.0 - c)**2
        s += 10.0 * (b + d - 2.0)**2 + 0.1 * (b - d)**2
    return float(s)

def extended_rosenbrock(x):
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Extended Rosenbrock requires even n.")
    s = 0.0
    for i in range(n // 2):
        a = x[2*i]
        b = x[2*i + 1]
        s += 100.0 * (b - a*a)**2 + (1.0 - a)**2
    return float(s)

def get_function(name):
    if name == "woods":
        return woods
    if name == "ext_rosenbrock":
        return extended_rosenbrock
    raise ValueError("Unknown function name.")
