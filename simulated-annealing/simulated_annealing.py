import random
import math
import matplotlib.pyplot as plt

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tour_length(points, tour):
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance(points[tour[i]], points[tour[(i + 1) % n]])
    return total

def two_opt_neighbor(tour):
    """Return a new tour by reversing a random segment (2-opt move)."""
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    new_tour = tour[:]
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour

def simulated_annealing_tsp(points, T0=1.0, alpha=0.995, steps=20000):
    n = len(points)
    current = list(range(n))
    random.shuffle(current)

    best = current[:]
    curr_E = tour_length(points, current)
    best_E = curr_E

    T = T0
    energies = [curr_E]
    temps = [T]

    for _ in range(steps):
        candidate = two_opt_neighbor(current)
        cand_E = tour_length(points, candidate)
        dE = cand_E - curr_E

        # Metropolis criterion
        if dE <= 0 or random.random() < math.exp(-dE / T):
            current = candidate
            curr_E = cand_E

            if curr_E < best_E:
                best = current[:]
                best_E = curr_E

        T *= alpha
        energies.append(curr_E)
        temps.append(T)

    return best, best_E, energies, temps, current

def plot_tour(points, tour, title):
    xs = [points[i][0] for i in tour] + [points[tour[0]][0]]
    ys = [points[i][1] for i in tour] + [points[tour[0]][1]]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

def main():
    random.seed(7)

    # Generate a random TSP instance
    n_cities = 35
    points = [(random.random(), random.random()) for _ in range(n_cities)]

    # Initial random tour
    initial_tour = list(range(n_cities))
    random.shuffle(initial_tour)
    initial_len = tour_length(points, initial_tour)

    # Run SA
    best_tour, best_len, energies, temps, final_state = simulated_annealing_tsp(
        points,
        T0=0.3,        # try 0.1 to 2.0 depending on scale
        alpha=0.995,   # slower cooling => better but slower
        steps=5000
    )

    print(f"Initial tour length: {initial_len:.4f}")
    print(f"Best tour length:    {best_len:.4f}")

    # Plot initial and best tours
    plot_tour(points, initial_tour, f"Initial tour (len={initial_len:.4f})")
    plot_tour(points, best_tour, f"Best SA tour (len={best_len:.4f})")

    # Plot energy + temperature traces (two separate figures)
    plt.figure()
    plt.plot(energies)
    plt.title("Cost (tour length) vs iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Tour length")
    plt.tight_layout()

    plt.figure()
    plt.plot(temps)
    plt.title("Temperature vs iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
