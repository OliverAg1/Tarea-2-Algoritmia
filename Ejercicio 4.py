import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias euclidianas entre las ciudades."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def tour_cost(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Calcula la distancia total de un tour cerrado."""
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))


def simulated_annealing_tsp(
    initial_tour: list[int],
    dist_matrix: np.ndarray,
    schedule_type: str,
    t_initial: float = 100.0,
    t_min: float = 1e-3,
    iters_per_temp: int = 30,
    seed: int = 42,
) -> dict:
    """
    Ejecuta Recocido Simulado usando 2-opt como operador de vecindad
    bajo distintos esquemas de enfriamiento.
    """
    rng = random.Random(seed)
    n = len(initial_tour)
    current_tour = initial_tour[:]
    current_cost = tour_cost(current_tour, dist_matrix)

    best_tour = current_tour[:]
    best_cost = current_cost

    temp = t_initial
    step = 0
    history_best_cost = [best_cost]
    history_temp = [temp]

    start_time = time.perf_counter()

    while temp > t_min:
        for _ in range(iters_per_temp):
            # Seleccionar dos índices i < j para movimiento 2-opt
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            if i == 0 and j == n - 1:
                continue

            a, b = current_tour[i], current_tour[i + 1]
            c, d = current_tour[j], current_tour[(j + 1) % n]

            # Costo incremental O(1) del movimiento 2-opt
            delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

            # Criterio de aceptación de Metropolis
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_tour[i + 1 : j + 1] = reversed(current_tour[i + 1 : j + 1])
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour[:]

        step += 1
        # Actualización de temperatura según el esquema de enfriamiento
        if schedule_type == "lineal":
            # Decaimiento constante en ~600 pasos térmicos
            temp = t_initial - (t_initial - t_min) * (step / 600.0)
        elif schedule_type == "exp_rapido":
            temp = t_initial * (0.990 ** step)
        elif schedule_type == "exp_lento":
            temp = t_initial * (0.9975 ** step)
        elif schedule_type == "cuadratico":
            temp = t_initial / (1.0 + 0.008 * (step ** 2))
        else:
            raise ValueError(f"Esquema desconocido: {schedule_type}")

        history_best_cost.append(best_cost)
        history_temp.append(max(temp, t_min))

    elapsed_time = time.perf_counter() - start_time

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "elapsed_time": elapsed_time,
        "total_iterations": step * iters_per_temp,
        "history_best_cost": history_best_cost,
        "history_temp": history_temp,
    }


def main():
    np.random.seed(42)
    num_cities = 30
    num_trials = 25
    coords = np.random.uniform(0, 100, size=(num_cities, 2))
    dist_matrix = compute_distance_matrix(coords)

    # Solución inicial aleatoria base para comparar curvas en igualdad de condiciones
    base_rng = random.Random(100)
    base_init_tour = list(range(num_cities))
    base_rng.shuffle(base_init_tour)

    schedules = [
        ("lineal", "Lineal ($T_k = T_0 - \\beta k$)", "#e15759"),
        ("cuadratico", "Cuadrático ($T_k = T_0 / (1 + \\gamma k^2)$)", "#f28e2b"),
        ("exp_rapido", "Exponencial Rápido ($\\alpha = 0.990$)", "#4e79a7"),
        ("exp_lento", "Exponencial Lento ($\\alpha = 0.9975$)", "#59a14f"),
    ]

    print("=" * 92)
    print(f"EJERCICIO 4: SIMULATED ANNEALING CON 2-OPT ({num_cities} ciudades, {num_trials} corridas por esquema)")
    print("=" * 92)
    print(f"{'Esquema de Enfriamiento':<32} | {'Costo Final (Media±DE)':^24} | {'Mejor Costo':^12} | {'Tiempo Medio (ms)':^17}")
    print("-" * 92)

    results_summary = {}
    sample_curves = {}

    for key, label, color in schedules:
        costs = []
        times_ms = []
        for trial in range(num_trials):
            # Misma permutación inicial aleatoria en cada trial para comparación justa
            trial_rng = random.Random(500 + trial)
            init_tour = list(range(num_cities))
            trial_rng.shuffle(init_tour)

            res = simulated_annealing_tsp(
                initial_tour=init_tour,
                dist_matrix=dist_matrix,
                schedule_type=key,
                seed=1000 + trial,
            )
            costs.append(res["best_cost"])
            times_ms.append(res["elapsed_time"] * 1000.0)

            if trial == 0:
                sample_curves[key] = res["history_best_cost"]

        results_summary[key] = {
            "label": label,
            "color": color,
            "costs": costs,
            "times_ms": times_ms,
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "min_cost": np.min(costs),
            "mean_time": np.mean(times_ms),
            "std_time": np.std(times_ms),
        }

        clean_name = label.split("(")[0].strip() + " (" + key + ")"
        print(
            f"{clean_name:<32} | "
            f"{np.mean(costs):>9.2f} ± {np.std(costs):<9.2f}   | "
            f"{np.min(costs):^12.2f} | "
            f"{np.mean(times_ms):>8.2f} ms"
        )

    print("=" * 92)

    # Visualización: 1) Curvas de Convergencia y 2) Tiempo vs. Calidad
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # Subgráfica 1: Curvas de convergencia (Costo vs Pasos de Temperatura)
    for key, label, color in schedules:
        curve = sample_curves[key]
        ax1.plot(range(len(curve)), curve, label=label, color=color, linewidth=2.2)

    ax1.set_title("1. Curvas de Convergencia (Mejor Costo vs. Pasos Térmicos)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pasos de Enfriamiento ($k$)", fontsize=10)
    ax1.set_ylabel("Mejor Costo del Tour Encontrado", fontsize=10)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subgráfica 2: Tiempo de Ejecución vs Calidad de la Solución Final
    for key, label, color in schedules:
        info = results_summary[key]
        # Graficar puntos individuales con transparencia
        ax2.scatter(info["times_ms"], info["costs"], color=color, alpha=0.25, s=28)
        # Graficar centroide (Media ± DE)
        ax2.errorbar(
            info["mean_time"],
            info["mean_cost"],
            xerr=info["std_time"],
            yerr=info["std_cost"],
            fmt="o",
            color=color,
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            markersize=9,
            markeredgecolor="black",
            label=f"{label.split('(')[0].strip()} ({info['mean_cost']:.1f}, {info['mean_time']:.1f}ms)",
        )

    ax2.set_title("2. Compromiso Tiempo de Ejecución vs. Calidad Final", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de Ejecución (ms)", fontsize=10)
    ax2.set_ylabel("Costo de la Solución Final (Menor = Mejor)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=8.5, title="Media (Costo, Tiempo)")

    fig.suptitle("Ejercicio 4: Ajuste de Parámetros en Simulated Annealing (Esquemas de Enfriamiento)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = "results/plots/ejercicio4_sa_tsp.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[OK] Gráfica guardada exitosamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()