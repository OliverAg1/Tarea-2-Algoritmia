import os
import matplotlib.pyplot as plt
import numpy as np


def greedy_knapsack(items: list[dict], capacity: float, criterion: str) -> dict:
    """
    Resuelve el problema de la mochila 0/1 usando una heurística voraz.
    - criterion == 'lightest': prioriza menor peso primero (w_i ascendente).
    - criterion == 'ratio': prioriza mayor valor/peso primero (v_i / w_i descendente).
    """
    if criterion == "lightest":
        sorted_items = sorted(items, key=lambda x: (x["weight"], -x["value"]))
    elif criterion == "ratio":
        sorted_items = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)
    else:
        raise ValueError("Criterio no válido. Usa 'lightest' o 'ratio'.")

    selected = []
    used_capacity = 0.0
    total_value = 0.0

    for item in sorted_items:
        if used_capacity + item["weight"] <= capacity:
            selected.append(item)
            used_capacity += item["weight"]
            total_value += item["value"]

    return {
        "criterion": criterion,
        "selected_items": selected,
        "selected_names": [it["name"] for it in selected],
        "used_capacity": used_capacity,
        "total_value": total_value,
    }


def main():
    # Instancia pequeña diseñada como contraejemplo (Capacidad W = 10 kg)
    capacity = 10
    items = [
        {"name": "A (Cable)",  "weight": 2, "value": 6},
        {"name": "B (Libro)",  "weight": 3, "value": 9},
        {"name": "C (Bocina)", "weight": 4, "value": 12},
        {"name": "D (Laptop)", "weight": 5, "value": 80},
        {"name": "E (Dron)",   "weight": 5, "value": 75},
    ]

    res_lightest = greedy_knapsack(items, capacity, criterion="lightest")
    res_ratio = greedy_knapsack(items, capacity, criterion="ratio")

    print("=" * 82)
    print(f"EJERCICIO 2: ANÁLISIS ALGORÍTMICO - GREEDY KNAPSACK (Capacidad W = {capacity} kg)")
    print("=" * 82)
    print("INSTANCIA DE ENTRADA:")
    print(f"{'Objeto':<14} | {'Peso (kg)':^10} | {'Valor ($)':^10} | {'Razón Valor/Peso ($/kg)':^24}")
    print("-" * 66)
    for it in items:
        ratio = it["value"] / it["weight"]
        print(f"{it['name']:<14} | {it['weight']:^10d} | {it['value']:^10d} | {ratio:^24.2f}")

    print("\n" + "=" * 82)
    print("COMPARACIÓN DE RESULTADOS ENTRE AMBOS MÉTODOS VORACES:")
    print("=" * 82)
    print(f"{'Método Voraz':<24} | {'Objetos Seleccionados':<28} | {'Cap. Usada':^12} | {'Valor Total':^11}")
    print("-" * 82)
    print(
        f"{'1. Menor Peso Primero':<24} | "
        f"{', '.join(res_lightest['selected_names']):<28} | "
        f"{res_lightest['used_capacity']:>4.0f}/{capacity} kg   | "
        f"${res_lightest['total_value']:>8.0f}"
    )
    print(
        f"{'2. Mayor Valor/Peso':<24} | "
        f"{', '.join(res_ratio['selected_names']):<28} | "
        f"{res_ratio['used_capacity']:>4.0f}/{capacity} kg   | "
        f"${res_ratio['total_value']:>8.0f}"
    )
    print("=" * 82)

    # Generación de la visualización comparativa
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    methods = ["Menor Peso\nPrimero", "Mayor Razón\nValor / Peso"]
    values = [res_lightest["total_value"], res_ratio["total_value"]]
    colors_val = ["#e15759", "#2ca02c"]

    # Subgráfica 1: Valor Total Obtenido
    bars1 = ax1.bar(methods, values, color=colors_val, width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_title("Comparación de Valor Total Obtenido ($)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Valor Total Acumulado ($)", fontsize=10)
    ax1.set_ylim(0, max(values) * 1.2)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    for bar, res in zip(bars1, [res_lightest, res_ratio]):
        height = bar.get_height()
        items_str = ", ".join([it["name"].split()[0] for it in res["selected_items"]])
        ax1.annotate(
            f"${height:.0f}\n(Objs: {items_str})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # Subgráfica 2: Desglose de Capacidad Usada por Objeto Seleccionado
    item_colors = {
        "A (Cable)": "#a6cee3",
        "B (Libro)": "#1f78b4",
        "C (Bocina)": "#b2df8a",
        "D (Laptop)": "#33a02c",
        "E (Dron)": "#fb9a99",
    }

    for idx, res in enumerate([res_lightest, res_ratio]):
        bottom = 0
        for it in res["selected_items"]:
            w = it["weight"]
            v = it["value"]
            ax2.bar(
                methods[idx],
                w,
                bottom=bottom,
                width=0.5,
                color=item_colors[it["name"]],
                edgecolor="black",
                linewidth=1,
                label=f"{it['name']} ({w}kg, ${v})",
            )
            ax2.text(
                idx,
                bottom + w / 2,
                f"{it['name']}\n({w}kg | ${v})",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="black",
            )
            bottom += w

    ax2.axhline(capacity, color="black", linestyle="--", linewidth=1.8, label=f"Capacidad Máxima ({capacity} kg)")
    ax2.set_title("Desglose de Objetos y Capacidad Usada (kg)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Peso Acumulado en la Mochila (kg)", fontsize=10)
    ax2.set_ylim(0, capacity * 1.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Ejercicio 2: Greedy Knapsack — Menor Peso vs. Mayor Valor/Peso", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = "results/plots/ejercicio2_knapsack.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[OK] Gráfica guardada exitosamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()