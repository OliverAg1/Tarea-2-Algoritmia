import os
import math
import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias euclidianas entre las ciudades."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def tour_cost(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Calcula el costo total de un ciclo cerrado."""
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))


def precompute_unique_tours(n: int) -> np.ndarray:
    """
    Precalcula todas las permutaciones únicas de un ciclo de n ciudades
    fijando la ciudad 0 al inicio y eliminando simetrías inversas.
    Para n = 9: 8! / 2 = 20,160 tours únicos.
    """
    other_cities = list(range(1, n))
    unique_tours = []
    for p in itertools.permutations(other_cities):
        if p[0] < p[-1]:
            unique_tours.append((0,) + p)
    return np.array(unique_tours, dtype=np.int32)


def exact_tsp_cost(dist_matrix: np.ndarray, all_tours: np.ndarray) -> float:
    """Evalúa de forma vectorizada los 20,160 tours y retorna el costo óptimo global."""
    step_costs = dist_matrix[all_tours[:, :-1], all_tours[:, 1:]].sum(axis=1)
    close_costs = dist_matrix[all_tours[:, -1], 0]
    return float(np.min(step_costs + close_costs))


def mst_approx_tour(dist_matrix: np.ndarray) -> list[int]:
    """
    Algoritmo 2-aproximado basado en MST:
    1. Construye el Árbol Recubridor Mínimo (MST) completo.
    2. Retorna el recorrido en preorden (DFS) iniciando en el nodo 0.
    """
    n = len(dist_matrix)
    graph = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist_matrix[i, j]))

    mst = nx.minimum_spanning_tree(graph, weight="weight")
    preorder_tour = list(nx.dfs_preorder_nodes(mst, source=0))
    return preorder_tour


def two_opt(initial_tour: list[int], dist_matrix: np.ndarray) -> tuple[list[int], float]:
    """Aplica búsqueda local 2-opt hasta converger a un óptimo local."""
    tour = initial_tour[:]
    n = len(tour)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                if delta < -1e-9:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True

    return tour, tour_cost(tour, dist_matrix)


def simulated_annealing_fast(initial_tour: list[int], dist_matrix: np.ndarray, rng: random.Random) -> float:
    """Recocido Simulado con enfriamiento exponencial y operador 2-opt para N=9."""
    n = len(initial_tour)
    curr_tour = initial_tour[:]
    curr_cost = tour_cost(curr_tour, dist_matrix)
    best_cost = curr_cost

    temp = 80.0
    t_min = 0.05
    alpha = 0.96

    while temp > t_min:
        for _ in range(12):
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            if i == 0 and j == n - 1:
                continue
            a, b = curr_tour[i], curr_tour[i + 1]
            c, d = curr_tour[j], curr_tour[(j + 1) % n]
            delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

            if delta < 0 or rng.random() < math.exp(-delta / temp):
                curr_tour[i + 1 : j + 1] = reversed(curr_tour[i + 1 : j + 1])
                curr_cost += delta
                if curr_cost < best_cost:
                    best_cost = curr_cost
        temp *= alpha

    return best_cost


def main():
    np.random.seed(42)
    rng = random.Random(42)

    num_instances = 100
    num_cities = 9

    all_tours = precompute_unique_tours(num_cities)

    opt_costs = []
    mst_costs = []
    mst_2opt_costs = []
    local_2opt_costs = []
    sa_costs = []
    approx_ratios = []

    for _ in range(num_instances):
        coords = np.random.uniform(0, 100, size=(num_cities, 2))
        dist_matrix = compute_distance_matrix(coords)

        # 1. Óptimo Exacto (Fuerza bruta vectorizada)
        c_opt = exact_tsp_cost(dist_matrix, all_tours)
        opt_costs.append(c_opt)

        # 2. Aproximación MST Puro
        tour_mst = mst_approx_tour(dist_matrix)
        c_mst = tour_cost(tour_mst, dist_matrix)
        mst_costs.append(c_mst)
        approx_ratios.append(c_mst / c_opt)

        # 3. MST + Post-procesamiento 2-opt
        _, c_mst_2opt = two_opt(tour_mst, dist_matrix)
        mst_2opt_costs.append(c_mst_2opt)

        # 4. Búsqueda Local 2-opt (desde solución aleatoria)
        rand_tour = list(range(num_cities))
        rng.shuffle(rand_tour)
        _, c_loc_2opt = two_opt(rand_tour, dist_matrix)
        local_2opt_costs.append(c_loc_2opt)

        # 5. Recocido Simulado (SA)
        c_sa = simulated_annealing_fast(rand_tour, dist_matrix, rng)
        sa_costs.append(c_sa)

    # Resumen Estadístico en Terminal
    print("=" * 90)
    print(f"EJERCICIO 5: APROXIMACIÓN MST EN {num_instances} INSTANCIAS DE {num_cities} CIUDADES")
    print("=" * 90)
    print(f"Razón de Aproximación MST (C_MST / C_óptimo):")
    print(f"  -> Media ± DE : {np.mean(approx_ratios):.4f} ± {np.std(approx_ratios):.4f}")
    print(f"  -> Mínima     : {np.min(approx_ratios):.4f}")
    print(f"  -> Máxima     : {np.max(approx_ratios):.4f}  (Límite teórico garantizado <= 2.0000)")
    print("-" * 90)
    print(f"{'Método Evaluado':<28} | {'Costo Medio ± DE':^24} | {'Razón Media (C/C_opt)':^22} | {'Óptimos Exactos':^15}")
    print("-" * 90)

    methods_data = [
        ("Óptimo Global Exacto", opt_costs),
        ("1. MST Puro", mst_costs),
        ("2. MST + 2-opt", mst_2opt_costs),
        ("3. Búsqueda Local (2-opt)", local_2opt_costs),
        ("4. Recocido Simulado (SA)", sa_costs),
    ]

    for name, costs in methods_data:
        ratios = np.array(costs) / np.array(opt_costs)
        exact_hits = np.sum(np.isclose(costs, opt_costs, atol=1e-5))
        print(
            f"{name:<28} | "
            f"{np.mean(costs):>9.2f} ± {np.std(costs):<9.2f}   | "
            f"{np.mean(ratios):^22.4f} | "
            f"{exact_hits:>6d}/{num_instances}"
        )
    print("=" * 90)

    # Visualización: 1) Histograma de Razón de Aproximación y 2) Boxplots Comparativos
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # Subgráfica 1: Histograma de la Razón de Aproximación (C_MST / C_óptimo)
    ax1.hist(approx_ratios, bins=14, color="#4e79a7", edgecolor="black", alpha=0.85, label="Instancias (MST Puro)")
    ax1.axvline(1.0, color="#2ca02c", linestyle="-.", linewidth=2, label="Óptimo Global (Razón = 1.0)")
    ax1.axvline(np.mean(approx_ratios), color="#f28e2b", linestyle="-", linewidth=2.2, label=f"Media Observada ({np.mean(approx_ratios):.3f})")
    ax1.axvline(2.0, color="#e15759", linestyle="--", linewidth=2.5, label="Límite Teórico Garantizado (2.0)")

    ax1.set_title("1. Razón de Aproximación MST ($C_{obtenido} / C_{óptimo}$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Razón de Aproximación ($C_{MST} / C_{óptimo}$)", fontsize=10)
    ax1.set_ylabel("Frecuencia (Número de Instancias)", fontsize=10)
    ax1.set_xlim(0.95, 2.1)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subgráfica 2: Comparativa de Costos Obtenidos entre Métodos
    box_data = [opt_costs, mst_costs, mst_2opt_costs, local_2opt_costs, sa_costs]
    box_labels = ["Óptimo\nExacto", "MST\nPuro", "MST +\n2-opt", "Búsqueda\nLocal (2-opt)", "Recocido\nSimulado"]
    box_colors = ["#bab0ac", "#e15759", "#59a14f", "#76b7b2", "#4e79a7"]

    bp = ax2.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        widths=0.48,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    for idx, data in enumerate(box_data, start=1):
        mean_val = np.mean(data)
        ax2.scatter(idx, mean_val, color="darkred", zorder=5, s=30, label="Media" if idx == 1 else "")
        ax2.annotate(
            f"{mean_val:.1f}",
            xy=(idx, mean_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
        )

    ax2.set_title("2. Comparativa de Costos en 100 Instancias (9 Ciudades)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Costo Total del Tour", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Ejercicio 5: Razón de Aproximación MST y Comparativa Global de Heurísticas", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = "results/plots/ejercicio5_mst_tsp.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[OK] Gráfica guardada exitosamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()