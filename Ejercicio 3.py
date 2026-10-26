import os
import random
import numpy as np
import matplotlib.pyplot as plt


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias euclidianas entre todas las ciudades."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def tour_cost(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Calcula el costo total de un recorrido cerrado (regresando al inicio)."""
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))


def nearest_neighbor_tour(dist_matrix: np.ndarray, start_node: int) -> list[int]:
    """Construye una solución inicial voraz usando el Vecino Más Cercano."""
    n = len(dist_matrix)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    current = start_node

    while unvisited:
        next_node = min(unvisited, key=lambda node: dist_matrix[current, node])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return tour


def two_opt(initial_tour: list[int], dist_matrix: np.ndarray) -> tuple[list[int], float, int]:
    """
    Aplica búsqueda local 2-opt hasta alcanzar un óptimo local.
    Retorna: (mejor_tour, costo_final, numero_de_mejoras).
    """
    tour = initial_tour[:]
    n = len(tour)
    improvements = 0
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Evitar invertir el ciclo completo cuando i=0 y j=n-1
                if i == 0 and j == n - 1:
                    continue

                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]

                # Cambio en el costo si intercambiamos las aristas (a,b) y (c,d) por (a,c) y (b,d)
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

                if delta < -1e-9:
                    # Invertir el segmento entre i+1 y j
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improvements += 1
                    improved = True

    final_cost = tour_cost(tour, dist_matrix)
    return tour, final_cost, improvements


def main():
    # Semilla fija para reproducibilidad de la instancia de ciudades
    np.random.seed(42)
    rng = random.Random(42)

    num_cities = 30
    num_trials = 50
    coords = np.random.uniform(0, 100, size=(num_cities, 2))
    dist_matrix = compute_distance_matrix(coords)

    rand_init_costs, rand_final_costs, rand_improvements = [], [], []
    nn_init_costs, nn_final_costs, nn_improvements = [], [], []

    for _ in range(num_trials):
        # Escenario 1: Solución inicial aleatoria
        init_rand = list(range(num_cities))
        rng.shuffle(init_rand)
        rand_init_costs.append(tour_cost(init_rand, dist_matrix))
        _, c_rand, imp_rand = two_opt(init_rand, dist_matrix)
        rand_final_costs.append(c_rand)
        rand_improvements.append(imp_rand)

        # Escenario 2: Vecino Más Cercano desde un nodo inicial aleatorio
        start_node = rng.randint(0, num_cities - 1)
        init_nn = nearest_neighbor_tour(dist_matrix, start_node)
        nn_init_costs.append(tour_cost(init_nn, dist_matrix))
        _, c_nn, imp_nn = two_opt(init_nn, dist_matrix)
        nn_final_costs.append(c_nn)
        nn_improvements.append(imp_nn)

    print("=" * 86)
    print(f"EJERCICIO 3: BÚSQUEDA LOCAL 2-OPT PARA TSP ({num_cities} ciudades, {num_trials} ejecuciones)")
    print("=" * 86)
    print(f"{'Escenario de Inicio':<26} | {'Costo Inicial (Media)':^21} | {'Costo 2-opt (Media±DE)':^22} | {'Mejoras (Media±DE)':^18}")
    print("-" * 86)
    print(
        f"{'1. Inicio Aleatorio':<26} | "
        f"{np.mean(rand_init_costs):^21.2f} | "
        f"{np.mean(rand_final_costs):>8.2f} ± {np.std(rand_final_costs):<8.2f} | "
        f"{np.mean(rand_improvements):>6.1f} ± {np.std(rand_improvements):<6.1f}"
    )
    print(
        f"{'2. Vecino Más Cercano (NN)':<26} | "
        f"{np.mean(nn_init_costs):^21.2f} | "
        f"{np.mean(nn_final_costs):>8.2f} ± {np.std(nn_final_costs):<8.2f} | "
        f"{np.mean(nn_improvements):>6.1f} ± {np.std(nn_improvements):<6.1f}"
    )
    print("=" * 86)

    # Visualización: Diagramas de Caja (Boxplots)
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    labels = ["Inicio\nAleatorio", "Vecino Más\nCercano (NN)"]
    palette = ["#ff9da7", "#76b7b2"]

    # Subgráfica 1: Número de mejoras necesarias para converger
    bp1 = ax1.boxplot(
        [rand_improvements, nn_improvements],
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp1["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    ax1.set_title("Número de Mejoras (Intercambios 2-opt) hasta Converger", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cantidad de Mejoras Aplicadas", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    # Anotar promedios en Subgráfica 1
    for idx, data in enumerate([rand_improvements, nn_improvements], start=1):
        mean_val = np.mean(data)
        ax1.scatter(idx, mean_val, color="red", zorder=5, label="Media" if idx == 1 else "")
        ax1.annotate(
            f"Media: {mean_val:.1f}",
            xy=(idx, mean_val),
            xytext=(18, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="darkred",
        )
    ax1.legend(loc="upper right", fontsize=9)

    # Subgráfica 2: Calidad del óptimo local alcanzado (Costo final)
    bp2 = ax2.boxplot(
        [rand_final_costs, nn_final_costs],
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp2["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    ax2.set_title("Calidad del Óptimo Local Alcanzado (Menor Costo = Mejor)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Distancia Total del Tour (Óptimo Local)", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    for idx, data in enumerate([rand_final_costs, nn_final_costs], start=1):
        mean_val = np.mean(data)
        ax2.scatter(idx, mean_val, color="red", zorder=5, label="Media" if idx == 1 else "")
        ax2.annotate(
            f"Media: {mean_val:.1f}",
            xy=(idx, mean_val),
            xytext=(18, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="darkred",
        )
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Ejercicio 3: Heurística 2-opt — Inicio Aleatorio vs. Vecino Más Cercano ({num_trials} corridas)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = "results/plots/ejercicio3_2opt_tsp.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[OK] Gráfica guardada exitosamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()