import os
import time
import random
import multiprocessing as mp
import matplotlib.pyplot as plt


def monte_carlo_pi_chunk(num_samples: int, seed: int) -> int:
    """Cuenta cuántos puntos aleatorios caen dentro del círculo unitario."""
    rng = random.Random(seed)
    inside_circle = 0
    for _ in range(num_samples):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return inside_circle


def estimate_pi(total_samples: int, num_processes: int) -> tuple[float, float]:
    """
    Ejecuta la estimación de Pi repartiendo total_samples entre num_processes.
    Retorna (pi_estimado, tiempo_ejecucion_segundos).
    """
    samples_per_proc = total_samples // num_processes
    tasks = [(samples_per_proc, 1000 + i) for i in range(num_processes)]

    start_time = time.perf_counter()
    if num_processes == 1:
        total_inside = monte_carlo_pi_chunk(*tasks[0])
    else:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(monte_carlo_pi_chunk, tasks)
            total_inside = sum(results)
    elapsed_time = time.perf_counter() - start_time

    pi_estimate = 4.0 * total_inside / (samples_per_proc * num_processes)
    return pi_estimate, elapsed_time


def main():
    total_samples = 40_000_000
    process_counts = [1, 2, 4, 8]
    times = []
    speedups = []
    efficiencies = []

    print("=" * 68)
    print(f"EJERCICIO 1: ACELERACIÓN PARALELA (N = {total_samples:,} muestras)")
    print("=" * 68)
    print(f"{'Procesos':^10} | {'Pi Estimado':^14} | {'Tiempo (s)':^12} | {'Speedup':^10} | {'Eficiencia':^10}")
    print("-" * 68)

    t_serial = None
    for p in process_counts:
        best_time = float("inf")
        best_pi = 0.0
        for _ in range(3):
            pi_val, elapsed = estimate_pi(total_samples, p)
            if elapsed < best_time:
                best_time = elapsed
                best_pi = pi_val

        if p == 1:
            t_serial = best_time

        speedup = t_serial / best_time
        efficiency = (speedup / p) * 100.0

        times.append(best_time)
        speedups.append(speedup)
        efficiencies.append(efficiency)

        print(f"{p:^10d} | {best_pi:^14.6f} | {best_time:^12.4f} | {speedup:^10.2f}x | {efficiency:^9.1f}%")

    print("=" * 68)

    # Generación de la gráfica sin advertencias de secuencia de escape
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(8, 5.5))
    plt.plot(
        process_counts,
        process_counts,
        "k--",
        linewidth=2,
        label=r"Speedup Ideal (Lineal $S(p) = p$)",
    )
    plt.plot(
        process_counts,
        speedups,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        markersize=8,
        label="Speedup Observado (estimate_pi)",
    )

    for p, s in zip(process_counts, speedups):
        plt.annotate(
            f"{s:.2f}x",
            (p, s),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.title(
        r"Ejercicio 1: Speedup Paralelo vs. Número de Procesos (Monte Carlo $\pi$)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(r"Número de Procesos ($p$)", fontsize=11)
    plt.ylabel(r"Aceleración / Speedup ($S(p) = T_1 / T_p$)", fontsize=11)
    plt.xticks(process_counts)
    plt.yticks(process_counts)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    output_path = "results/plots/ejercicio1_speedup.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[OK] Gráfica guardada exitosamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()