#Min Max ACO with 2 opt with constr - gpt


import numpy as np
import random
import time

# --------------------------- EVRP Parsing ---------------------------
def parse_evrp_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    nodes, demands, stations = {}, {}, []
    capacity = energy_capacity = energy_consumption = depot = None
    section = None

    for line in lines:
        line = line.strip()
        if not line or line == 'EOF':
            continue
        if line.startswith('NODE_COORD_SECTION'):
            section = 'coords'
            continue
        if line.startswith('DEMAND_SECTION'):
            section = 'demands'
            continue
        if line.startswith('DEPOT_SECTION'):
            section = 'depot'
            continue
        if line.startswith('STATIONS_COORD_SECTION'):
            section = 'stations'
            continue
        if line.startswith('CAPACITY:'):
            capacity = int(line.split()[-1])
        elif line.startswith('ENERGY_CAPACITY:'):
            energy_capacity = int(line.split()[-1])
        elif line.startswith('ENERGY_CONSUMPTION:'):
            energy_consumption = float(line.split()[-1])

        if section == 'coords':
            parts = line.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                nodes[nid] = (float(parts[1]), float(parts[2]))
        elif section == 'demands':
            parts = line.split()
            if len(parts) == 2:
                demands[int(parts[0])] = float(parts[1])
        elif section == 'stations':
            if line.isdigit():
                stations.append(int(line))
        elif section == 'depot':
            if line.isdigit():
                depot = int(line)

    return nodes, demands, depot, stations, capacity, energy_capacity, energy_consumption

# --------------------------- Distance Matrix ---------------------------
def calculate_distance_matrix(nodes, id_to_index):
    n = len(nodes)
    matrix = np.zeros((n, n))
    for i in nodes:
        for j in nodes:
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            matrix[id_to_index[i]][id_to_index[j]] = np.hypot(xi - xj, yi - yj)
    return matrix

# --------------------------- Route Cost ---------------------------
def route_cost(route, dist_matrix, id_to_index):
    return sum(dist_matrix[id_to_index[route[i]]][id_to_index[route[i+1]]] for i in range(len(route) - 1))

# --------------------------- 2-opt ---------------------------
def two_opt(route, dist_matrix, id_to_index):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_cost(new_route, dist_matrix, id_to_index) < route_cost(best, dist_matrix, id_to_index):
                    best = new_route
                    improved = True
    return best

# --------------------------- Visualization ---------------------------
def plot_routes(routes, nodes, depot):
    plt.figure(figsize=(8, 6))
    for route in routes:
        x = [nodes[n][0] for n in route]
        y = [nodes[n][1] for n in route]
        plt.plot(x, y, marker='o')
    plt.plot(nodes[depot][0], nodes[depot][1], 'rs', label='Depot')
    plt.title('Min-Max ACO Routes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_improvement(costs):
    plt.figure(figsize=(10, 4))
    plt.plot(costs, marker='o')
    plt.title("Cost Improvement Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

# --------------------------- Export ---------------------------
def export_solution(routes, filename='solution.sol'):
    with open(filename, 'w') as f:
        for i, route in enumerate(routes):
            route_str = 'Route #{0}: {1}\n'.format(i+1, ' -> '.join(map(str, route)))
            f.write(route_str)

# --------------------------- Min-Max ACO Solver ---------------------------
def solve_minmax_aco(filepath, num_ants=100, iterations=150, alpha=1.0, beta=2.0, rho=0.05, q=250, tau_min=0.1, tau_max=10, max_route_len=100, max_recharges=5):
    nodes, demands, depot, stations, cap, energy_cap, energy_rate = parse_evrp_file(filepath)

    node_ids = list(nodes.keys())
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    dist_matrix = calculate_distance_matrix(nodes, id_to_index)
    n = len(node_ids)
    pheromone = np.ones((n, n))

    best_solution = None
    best_cost = float('inf')
    worst_cost = float('-inf')
    all_costs = []
    stagnation_counter = 0
    stagnation_limit = 50

    start_time = time.time()
    trend = []

    for iteration in range(iterations):
        iteration_best_cost = float('inf')
        iteration_costs = []
        iteration_best_solution = None

        for _ in range(num_ants):
            unvisited = set(demands.keys())
            ant_routes = []
            total_cost = 0

            while unvisited:
                route = [depot]
                current, load, energy, recharges = depot, cap, energy_cap, 0

                while True:
                    candidates = [j for j in unvisited if demands[j] <= load and energy >= dist_matrix[id_to_index[current]][id_to_index[j]] * energy_rate]

                    if len(route) > max_route_len:
                        break

                    if not candidates:
                        station_options = [s for s in stations if energy >= dist_matrix[id_to_index[current]][id_to_index[s]] * energy_rate]
                        if station_options and recharges < max_recharges:
                            next_station = min(station_options, key=lambda s: dist_matrix[id_to_index[current]][id_to_index[s]])
                            route.append(next_station)
                            energy = energy_cap
                            recharges += 1
                            current = next_station
                            continue
                        else:
                            break

                    probs = []
                    for j in candidates:
                        tau = pheromone[id_to_index[current]][id_to_index[j]] ** alpha
                        eta = (1 / (dist_matrix[id_to_index[current]][id_to_index[j]] + 1e-6)) ** beta
                        probs.append(tau * eta)

                    probs = np.array(probs)
                    probs /= probs.sum()
                    next_node = np.random.choice(candidates, p=probs)

                    route.append(next_node)
                    load -= demands[next_node]
                    energy -= dist_matrix[id_to_index[current]][id_to_index[next_node]] * energy_rate
                    current = next_node
                    unvisited.remove(next_node)

                if route[-1] != depot:
                    route.append(depot)

                route = two_opt(route, dist_matrix, id_to_index)
                ant_routes.append(route)
                total_cost += route_cost(route, dist_matrix, id_to_index)

            iteration_costs.append(total_cost)
            if total_cost < iteration_best_cost:
                iteration_best_cost = total_cost
                iteration_best_solution = ant_routes

        all_costs.append(iteration_best_cost)
        trend.append(iteration_best_cost)

        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_solution = iteration_best_solution
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_limit:
            print(f"Early stopping at iteration {iteration+1} due to stagnation.")
            break

        worst_cost = max(worst_cost, max(iteration_costs))

        rho = max(0.01, rho - 0.001)
        alpha += 0.01
        beta = max(1.0, beta - 0.01)

        pheromone = np.clip((1 - rho) * pheromone, tau_min, tau_max)
        for route in best_solution:
            for i in range(len(route) - 1):
                pheromone[id_to_index[route[i]]][id_to_index[route[i+1]]] += q / route_cost(route, dist_matrix, id_to_index)

    duration = round(time.time() - start_time, 2)

    print("Best Solution Cost:", best_cost)
    print("Worst Solution Cost:", worst_cost)
    print("Average Solution Cost:", round(np.mean(all_costs), 2))
    print("Execution Time (s):", duration)

    export_solution(best_solution)
    plot_routes(best_solution, nodes, depot)
    plot_improvement(trend)

    return best_solution, best_cost, duration, worst_cost

def run_multiple_trials(filepath, trials=10):
    best_costs = []
    worst_costs = []
    avg_costs = []
    exec_times = []

    for i in range(trials):
        print(f"\n--- Trial {i+1} ---")
        solution, best_cost, exec_time, worst_cost = solve_minmax_aco(filepath)
        best_costs.append(best_cost)
        worst_costs.append(worst_cost)
        exec_times.append(exec_time)

        # Use last average from cost trend for average cost
        avg_cost = sum(best_costs) / len(best_costs)
        avg_costs.append(avg_cost)

    summary = {
        "Min Best Cost": min(best_costs),
        "Max Worst Cost": max(worst_costs),
        "Avg of Avg Cost": round(sum(avg_costs) / len(avg_costs), 2),
        "Avg Time (s)": round(sum(exec_times) / len(exec_times), 2)
    }

    print("\n====== Summary Over", trials, "Trials ======")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary

# Call this instead of single solve
summary = run_multiple_trials("E-n89-k7-s13.evrp", trials=10)
