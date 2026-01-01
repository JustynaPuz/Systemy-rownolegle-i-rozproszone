import argparse
import time
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(
        description="Task 13: Sequential Nelder–Mead"
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

def woods(x):
    n = len(x)
    if n % 4 != 0:
        raise ValueError("Woods requires n divisible by 4")
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
        raise ValueError("Extended Rosenbrock requires even n")
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
    raise ValueError("Unknown function")


def starting_point(func_name, n):
    if func_name == "woods":
        if n % 4 != 0:
            raise ValueError("Woods requires n divisible by 4")
        x0 = np.empty(n, dtype=np.float64)
        for i in range(n // 2):
            x0[2*i] = -3.0
            x0[2*i + 1] = -1.0
        return x0

    if func_name == "ext_rosenbrock":
        if n % 2 != 0:
            raise ValueError("Extended Rosenbrock requires even n")
        x0 = np.empty(n, dtype=np.float64)
        for i in range(n // 2):
            x0[2*i] = -1.2
            x0[2*i + 1] = 1.0
        return x0

    raise ValueError("Unknown function")


def build_initial_simplex(x0, step):
    n = len(x0)
    simplex = np.tile(x0, (n + 1, 1))
    for l in range(1, n + 1):
        simplex[l, l - 1] += step
    return simplex


def eval_points_seq(f, points):
    vals = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        vals[i] = f(points[i])
    return vals


def max_pairwise_distance(simplex):
    m = simplex.shape[0]
    max_d2 = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = simplex[i] - simplex[j]
            d2 = float(np.dot(d, d))
            if d2 > max_d2:
                max_d2 = d2
    return float(np.sqrt(max_d2))


def nm_seq(f, x0, step, alpha, beta, eps, max_iter):
    n = len(x0)
    simplex = build_initial_simplex(x0, step)

    iters = 0
    while iters < max_iter:
        iters += 1

        fvals = eval_points_seq(f, simplex)
        best_idx = int(np.argmin(fvals))
        x_star = simplex[best_idx].copy()
        f_star = float(fvals[best_idx])

        others = np.delete(simplex, best_idx, axis=0)

        xr = 2.0 * x_star - others
        fr = eval_points_seq(f, xr)

        if float(np.min(fr)) < f_star:
            xe = alpha * xr + (1.0 - alpha) * x_star
            fe = eval_points_seq(f, xe)

            if float(np.min(fe)) < float(np.min(fr)):
                simplex = np.vstack((x_star, xe))
            else:
                simplex = np.vstack((x_star, xr))
        else:
            xk = beta * others + (1.0 - beta) * x_star
            simplex = np.vstack((x_star, xk))

            if max_pairwise_distance(simplex) < eps:
                break

    fvals = eval_points_seq(f, simplex)
    best_idx = int(np.argmin(fvals))
    x_best = simplex[best_idx].copy()
    f_best = float(fvals[best_idx])
    dist = max_pairwise_distance(simplex)
    return x_best, f_best, iters, dist

def run_once(func_name, n, step, alpha, beta, eps, max_iter):
    f = get_function(func_name)
    x0 = starting_point(func_name, n)

    t0 = time.perf_counter()
    x_best, f_best, iters, dist = nm_seq(f, x0, step, alpha, beta, eps, max_iter)
    t1 = time.perf_counter()

    return {
        "func": func_name,
        "n": n,
        "step": step,
        "alpha": alpha,
        "beta": beta,
        "eps": eps,
        "max_iter": max_iter,
        "time_s": (t1 - t0),
        "iters": iters,
        "f_best": f_best,
        "dist": dist,
        "x_best": x_best,
    }

def print_result(res, print_x):
    print(f"func={res['func']} n={res['n']} step={res['step']} alpha={res['alpha']} beta={res['beta']} eps={res['eps']}")
    print(f"iters={res['iters']} time_s={res['time_s']:.6f} f_best={res['f_best']:.12e} simplex_max_dist={res['dist']:.6e}")

    if print_x:
        x = res["x_best"]
        k = min(10, len(x))
        print("x_best_first =", " ".join(f"{x[i]:.6f}" for i in range(k)))

    print("csv: func,n,variant,p,time_s,iters,f_best,dist")
    print(f"csv: {res['func']},{res['n']},seq,1,{res['time_s']:.6f},{res['iters']},{res['f_best']:.12e},{res['dist']:.6e}")


def append_csv_line(path, res):
    line = f"{res['func']},{res['n']},seq,1,{res['time_s']:.6f},{res['iters']},{res['f_best']:.12e},{res['dist']:.6e}\n"
    header = "func,n,variant,p,time_s,iters,f_best,dist\n"
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(header)
            f.write(line)
    except FileExistsError:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def main():
    args = parse_args()

    res = run_once(
        func_name=args.func,
        n=args.n,
        step=args.step,
        alpha=args.alpha,
        beta=args.beta,
        eps=args.eps,
        max_iter=args.max_iter,
    )

    print_result(res, args.print_x)

    if args.csv_out:
        append_csv_line(args.csv_out, res)


if __name__ == "__main__":
    main()
