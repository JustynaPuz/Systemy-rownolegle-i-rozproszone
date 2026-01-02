import multiprocessing as mp
import numpy as np
import time
import functions

# GŁÓWNE ZMIENNE KONFIGURACYJNE

# ID funkcji testowej (1-17, zgodnie z functions.py i PDF)
FUNC_ID = 3  

# Uwaga: Niektóre funkcje wymagają n podzielnego przez 2 lub 4 (np. ID 2, 3, 4, 8)
N_DIM = 60    

# Parametry algorytmu Neldera-Meada
ALPHA = 1.5        # Współczynnik ekspansji (> 1)
BETA = 0.5         # Współczynnik kontrakcji (0 < beta < 1)
STEP_S = 1.0       # Krok początkowy do budowy sympleksu
EPSILON = 0.001    # Kryterium stopu (max odległość)
MAX_ITER = 2000    # Zabezpieczenie przed pętlą nieskończoną

# Liczba procesów (rdzeni) - ustawienie 'None' użyje wszystkich dostępnych
NUM_PROCESSES = 1 

# ==============================================================================
# FUNKCJE POMOCNICZE DLA MULTIPROCESSING
# ==============================================================================

def evaluate_wrapper(args):
    """
    Wrapper (funkcja opakowująca) potrzebna dla pool.map.
    Rozpakowuje argumenty i wywołuje odpowiednią funkcję matematyczną.
    args: (func_id, x_vector)
    """
    fid, x = args
    # Pobieramy samą funkcję (bez x0) z modułu functions
    func, _ = functions.get_function_setup(fid, len(x))
    return func(x)

def parallel_evaluate(pool, simplex, func_id):
    """
    Oblicza wartości funkcji celu dla listy punktów (sympleksu) równolegle.
    Zgodnie z zasadą zrównoleglenia danych.
    """
    tasks = [(func_id, point) for point in simplex]
    # Mapa rozsyła zadania do workerów
    values = pool.map(evaluate_wrapper, tasks)
    return np.array(values)

def check_stop_condition(simplex, epsilon):
    n_points = len(simplex)
    max_dist = 0.0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(simplex[i] - simplex[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist < epsilon, max_dist


def run_optimization():
    _, x0 = functions.get_function_setup(FUNC_ID, N_DIM)
    
    print(f"--- START: Równoległy Nelder-Mead ---")
    print(f"Funkcja ID: {FUNC_ID}, Wymiar: {N_DIM}")
    print(f"Start x0: {x0}")
    print(f"Procesy: {NUM_PROCESSES if NUM_PROCESSES else mp.cpu_count()}")

    pool = mp.Pool(processes=NUM_PROCESSES)
    
    start_time = time.time()

    simplex = [x0.copy()]
    for i in range(N_DIM):
        next_p = x0.copy()
        next_p[i] += STEP_S
        simplex.append(next_p)
    simplex = np.array(simplex)

    for k in range(MAX_ITER):
        f_values = parallel_evaluate(pool, simplex, FUNC_ID)
        sorted_indices = np.argsort(f_values)
        simplex = simplex[sorted_indices]
        f_values = f_values[sorted_indices]
        x_best = simplex[0]
        f_best = f_values[0]
        simplex_reflected = 2 * x_best - simplex
        f_reflected = parallel_evaluate(pool, simplex_reflected, FUNC_ID)
        
        mask = np.ones(len(f_reflected), dtype=bool)
        mask[0] = False
        min_f_refl = np.min(f_reflected[mask])
        action = ""
        if min_f_refl < f_best:
            simplex_expansion = ALPHA * simplex_reflected + (1 - ALPHA) * x_best
            f_expansion = parallel_evaluate(pool, simplex_expansion, FUNC_ID)
            min_f_exp = np.min(f_expansion[mask])
            
            if min_f_exp < min_f_refl:
                simplex = simplex_expansion
                simplex[0] = x_best
                action = "Ekspansja"
            else:
                # Ekspansja nie pomogła bardziej niż odbicie -> przyjmujemy odbicie
                simplex = simplex_reflected
                simplex[0] = x_best
                action = "Odbicie"
        else:
            simplex_contraction = BETA * simplex + (1 - BETA) * x_best
            stop, dist = check_stop_condition(simplex_contraction, EPSILON)
            
            if stop:
                simplex = simplex_contraction
                f_final = parallel_evaluate(pool, simplex, FUNC_ID)
                best_idx = np.argmin(f_final)
                print(f"\n--- KONIEC: Kryterium stopu spełnione w iteracji {k} ---")
                print(f"Max odległość: {dist:.6f}")
                print(f"Znalezione minimum f(x) = {f_final[best_idx]:.10f}")
                print(f"W punkcie x = {simplex[best_idx]}")
                break
            else:
                # Przejście do kroku 2 z nowym (skurczonym) sympleksem
                simplex = simplex_contraction
                action = "Kontrakcja"

        if k % 50 == 0:
            print(f"Iteracja {k}: f_best = {f_best:.6f} [{action}]")

    end_time = time.time()
    print(f"Czas obliczeń: {end_time - start_time:.4f} s")
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    run_optimization()