import numpy as np
import time
import functions
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# KONFIGURACJA
# ==============================================================================

FUNC_ID = 3  # numer funkcji z pliku functions.py
N_DIM = 60  # wielkość problemu
ALPHA = 1.5  # parametr ekspansji
BETA = 0.5  # parametr kontrakcji
STEP_S = 1.0  # wielkość początkowego sympleksu
EPSILON = 0.001  # dokładność (kiedy przestać)
MAX_ITER = 2000  # żeby pętla nie kręciła się w nieskończoność

# ile wątków
NUM_THREADS = 4

# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def evaluate_point_wrapper(args):
    """
    Wrapper potrzebny dla executora.
    Rozpakowuje (id, wektor), liczy wartość i zwraca wynik.
    """
    fid, x = args
    func, _ = functions.get_function_setup(fid, len(x))
    return func(x)


def parallel_evaluate_threads(executor, simplex, func_id):
    """
    Główna funkcja zrównoleglająca obliczanie wartości f(x).
    """
    # lista zadań: pary (id_funkcji, współrzędne punktu)
    tasks = [(func_id, point) for point in simplex]

    # mapowanie zadań na wątki i zbieranie wyników do listy
    results = list(executor.map(evaluate_point_wrapper, tasks))

    return np.array(results)


def check_stop_condition(simplex, epsilon):
    """
    Sprawdzamy czy punkty sympleksu są już wystarczająco blisko siebie.
    """
    n_points = len(simplex)
    max_dist = 0.0

    # zwykła pętla po wszystkich parach punktów
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(simplex[i] - simplex[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist < epsilon, max_dist


# ==============================================================================
# GŁÓWNA PĘTLA ALGORYTMU
# ==============================================================================

def run_optimization_threads():
    # bierzemy punkt startowy dla wybranej funkcji
    _, x0 = functions.get_function_setup(FUNC_ID, N_DIM)

    print(f"--- START: Nelder-Mead na WĄTKACH (Threading) ---")
    print(f"Funkcja ID: {FUNC_ID}, Wymiar: {N_DIM}")
    print(f"Liczba wątków: {NUM_THREADS}")

    # odpalamy pulę wątków
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:

        start_time = time.time()

        # Krok 1: Tworzymy sympleks startowy (x0 + przesunięcia)
        simplex = [x0.copy()]
        for i in range(N_DIM):
            next_p = x0.copy()
            next_p[i] += STEP_S
            simplex.append(next_p)
        simplex = np.array(simplex)

        # Główna pętla optymalizacji
        for k in range(MAX_ITER):

            # Krok 2: Liczymy wartości funkcji dla wszystkich punktów (równolegle)
            f_values = parallel_evaluate_threads(executor, simplex, FUNC_ID)

            # sortujemy punkty od najlepszego (najmniejsze f(x)) do najgorszego
            sorted_indices = np.argsort(f_values)
            simplex = simplex[sorted_indices]
            f_values = f_values[sorted_indices]

            # best
            x_best = simplex[0]
            f_best = f_values[0]

            # Krok 3: Odbicie (wszystkie punkty oprócz najlepszego)
            simplex_reflected = 2 * x_best - simplex

            # macierzowo liczymy f(x) dla wszystkich odbitych mimo że odbicie najlepszego punktu nas nie interesuje
            f_reflected = parallel_evaluate_threads(executor, simplex_reflected, FUNC_ID)

            # maska żeby ignorować wynik dla x_best (index 0)
            mask = np.ones(len(f_reflected), dtype=bool)
            mask[0] = False

            min_f_refl = np.min(f_reflected[mask])
            action = ""

            # sprawdzamy czy po odbiciu jest lepiej
            if min_f_refl < f_best:
                # Krok 4: Skoro jest lepiej, to próbujemy Ekspansji (idziemy dalej w tym kierunku)
                simplex_expansion = ALPHA * simplex_reflected + (1 - ALPHA) * x_best
                f_expansion = parallel_evaluate_threads(executor, simplex_expansion, FUNC_ID)
                min_f_exp = np.min(f_expansion[mask])

                if min_f_exp < min_f_refl:
                    # ekspansja się opłaciła
                    simplex = simplex_expansion
                    simplex[0] = x_best  # przywracamy starego mistrza na miejsce 0
                    action = "Ekspansja"
                else:
                    # ekspansja nie pomogła, zostajemy przy odbiciu
                    simplex = simplex_reflected
                    simplex[0] = x_best
                    action = "Odbicie"
            else:
                # Krok 5: Odbicie nic nie dało -> więc zmniejszamy sympleks
                simplex_contraction = BETA * simplex + (1 - BETA) * x_best

                # sprawdzamy czy można już kończyć (zanim policzymy wartości)
                stop, dist = check_stop_condition(simplex_contraction, EPSILON)

                if stop:
                    # liczymy ostatni raz wartości żeby wypisać wynik
                    f_final = parallel_evaluate_threads(executor, simplex_contraction, FUNC_ID)
                    best_idx = np.argmin(f_final)

                    print(f"\n--- KONIEC: Warunek stopu spełniony w iteracji {k} ---")
                    print(f"Max odległość: {dist:.6f}")
                    print(f"Znalezione minimum: {f_final[best_idx]:.10f}")
                    break
                else:
                    # lecimy dalej z mniejszym sympleksem
                    simplex = simplex_contraction
                    action = "Kontrakcja"

            if k % 50 == 0:
                print(f"Iteracja {k}: f_best = {f_best:.6f} [{action}]")

        end_time = time.time()
        print(f"Czas obliczeń (Wątki): {end_time - start_time:.4f} s")


if __name__ == "__main__":
    run_optimization_threads()