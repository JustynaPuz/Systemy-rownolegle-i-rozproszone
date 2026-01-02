from mpi4py import MPI
import numpy as np
import functions
import time

# ==============================================================================
# KONFIGURACJA
# ==============================================================================

FUNC_ID = 3      # Numer funkcji
N_DIM = 60       # Rozmiar problemu
ALPHA = 1.5      # Współczynnik ekspansji
BETA = 0.5       # Współczynnik kontrakcji
STEP_S = 1.0     # Rozmiar startowy
EPSILON = 0.001  # Kiedy przerywamy
MAX_ITER = 2000  # Bezpiecznik pętli


# ==============================================================================
# FUNKCJA DO KOMUNIKACJI (SERCE MPI)
# ==============================================================================

def distributed_evaluate(comm, simplex, func_id):
    """
    Kluczowa funkcja zrównoleglająca. Działa na zasadzie:
    Dziel i Rządź -> Policz -> Pozbieraj
    """
    rank = comm.Get_rank() # (0 = Szef, >0 = worker)
    size = comm.Get_size() # Ilu?

    chunks = None

    # KROK 1: Dzielenie dużej tablicy punktów na mniejsze kawałki dla każdego procesu
    if rank == 0:
        chunks = np.array_split(simplex, size, axis=0)

    # KROK 2: Rozsyłanie
    my_chunk = comm.scatter(chunks, root=0)

    # KROK 3: Każdy proces liczy u siebie
    my_results = []
    func, _ = functions.get_function_setup(func_id, len(my_chunk[0]) if len(my_chunk) > 0 else 0)

    # Liczymy f(x) dla przydzielonych punktów
    for point in my_chunk:
        val = func(point)
        my_results.append(val)

    my_results = np.array(my_results)

    # KROK 4: Zbieranie wyników
    # Wszyscy odsyłają wyniki do Szefa.
    # Szef dostaje listę tablic.
    all_results_list = comm.gather(my_results, root=0)

    # Szef skleja to w jedną długą tablicę i zwraca
    # Reszta procesów zwraca None
    if rank == 0:
        return np.concatenate(all_results_list)
    else:
        return None


def check_stop_condition_simple(simplex):
    """Pomocnicza funkcja: sprawdza czy trójkąt jest już wystarczająco mały"""
    n_points = len(simplex)
    max_dist = 0.0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(simplex[i] - simplex[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


# ==============================================================================
# GŁÓWNA PĘTLA
# ==============================================================================

def run_mpi():
    comm = MPI.COMM_WORLD   # Inicjalizacja świata procesów
    rank = comm.Get_rank()  # Numer ID procesu
    size = comm.Get_size()  # Liczba procesów

    # Inicjalizacja
    if rank == 0:
        print(f"--- START: Nelder-Mead na MPI ---")
        print(f"Liczba procesow: {size}")
        print(f"Funkcja ID: {FUNC_ID}, Wymiar: {N_DIM}")

        _, x0 = functions.get_function_setup(FUNC_ID, N_DIM)
        start_time = time.time()

        # Budowa pierwszego sympleksu
        simplex = [x0.copy()]
        for i in range(N_DIM):
            next_p = x0.copy()
            next_p[i] += STEP_S
            simplex.append(next_p)
        simplex = np.array(simplex)
    else:
        # Workerzy oczekują na rozkazy
        simplex = None

    # PĘTLA GŁÓWNA ALGORYTMU
    for k in range(MAX_ITER):

        # 1. Liczymy wartości funkcji dla sympleksu
        # Wszyscy pracownicy korzystają z tej funkcji, następuje wymiana danych
        f_values = distributed_evaluate(comm, simplex, FUNC_ID)

        # Decyzje podejmuje szef (rank 0)
        if rank == 0:
            # Sortujemy punkty od najlepszego
            sorted_indices = np.argsort(f_values)
            simplex = simplex[sorted_indices]
            f_values = f_values[sorted_indices]

            x_best = simplex[0]
            f_best = f_values[0]

            # Przygotowanie punktów odbitych
            simplex_reflected = 2 * x_best - simplex
        else:
            simplex_reflected = None

        # 2. Liczymy wartości dla punktów odbitych (workerzy pracują)
        f_reflected = distributed_evaluate(comm, simplex_reflected, FUNC_ID)

        stop_loop = False

        # Analiza wyników (szef)
        if rank == 0:
            mask = np.ones(len(f_reflected), dtype=bool)
            mask[0] = False # Ignorujemy najlepszy punkt (odbicie względem siebie)
            min_f_refl = np.min(f_reflected[mask])

            action = ""
            next_simplex_candidates = None
            mode = "NONE" # EXPANSION czy CONTRACTION?

            if min_f_refl < f_best:
                # Odbicie super -> próbujemy Ekspansji
                simplex_expansion = ALPHA * simplex_reflected + (1 - ALPHA) * x_best
                next_simplex_candidates = simplex_expansion
                mode = "EXPANSION"
            else:
                # Odbicie słabe -> robimy Kontrakcję
                simplex_contraction = BETA * simplex + (1 - BETA) * x_best

                # Sprawdzamy czy to już koniec (czy sympleks jest mały)
                max_dist = check_stop_condition_simple(simplex_contraction)

                if max_dist < EPSILON:
                    stop_loop = True
                    print(f"\n--- KONIEC: Warunek stopu w iteracji {k} ---")
                    print(f"Max odleglosc: {max_dist:.6f}")
                    print(f"Minimum: {f_best:.10f}")

                next_simplex_candidates = simplex_contraction
                mode = "CONTRACTION"
        else:
            # Pracownicy resetują zmienne pomocnicze
            mode = None
            next_simplex_candidates = None

        # szef daje sygnał zakończenia pracy
        # Broadcast (bcast) wysyła flagę 'stop_loop' do wszystkich
        stop_loop = comm.bcast(stop_loop, root=0)
        if stop_loop:
            break

        # 3. Liczymy wartości dla kandydatów (Ekspansja lub Kontrakcja)
        f_candidates = distributed_evaluate(comm, next_simplex_candidates, FUNC_ID)

        # Ostateczna aktualizacja sympleksu na nowy (szef)
        if rank == 0:
            if mode == "EXPANSION":
                mask = np.ones(len(f_candidates), dtype=bool)
                mask[0] = False
                min_f_exp = np.min(f_candidates[mask])

                if min_f_exp < min_f_refl:
                    simplex = next_simplex_candidates
                    simplex[0] = x_best
                    action = "Ekspansja"
                else:
                    simplex = simplex_reflected
                    simplex[0] = x_best
                    action = "Odbicie"

            elif mode == "CONTRACTION":
                simplex = next_simplex_candidates
                action = "Kontrakcja"

            if k % 50 == 0:
                print(f"Iteracja {k}: f_best = {f_best:.6f} [{action}]")

    # Koniec pracy - pomiar czasu
    if rank == 0:
        end_time = time.time()
        print(f"Czas MPI: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    run_mpi()