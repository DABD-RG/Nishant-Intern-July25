#ACO + 2 opt + constraints + optimization + Very Final one - working

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os

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

# --------------------------- Visualization ---------------------------
def plot_routes(routes, nodes, depot):
    plt.figure(figsize=(8, 6))
    for route in routes:
        x = [nodes[n][0] for n in route]
        y = [nodes[n][1] for n in route]
        plt.plot(x, y, marker='o')
    plt.plot(nodes[depot][0], nodes[depot][1], 'rs', label='Depot')
    plt.title('Routes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_improvement(trend):
    plt.figure(figsize=(10, 4))
    plt.plot(trend, marker='o')
    plt.title("ACO Cost Improvement Over Iterations")
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

# --------------------------- ACO Solver ---------------------------
def solve_ecvrp_aco(filepath, num_ants=100, iterations=2000, alpha=1.0, beta=2.0, rho=0.05, q=150, max_route_len=100, max_recharges=5):
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
    elite_archive = []
    stagnation_counter = 0
    stagnation_limit = 30

    start_time = time.time()
    trend = []

    for iteration in range(iterations):
        iteration_costs = []
        iteration_best_solution = None
        iteration_best_cost = float('inf')

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
                ant_routes.append(route)
                total_cost += route_cost(route, dist_matrix, id_to_index)

            iteration_costs.append(total_cost)

            if total_cost < iteration_best_cost:
                iteration_best_cost = total_cost
                iteration_best_solution = ant_routes

        alpha += 0.01
        beta = max(1.0, beta - 0.01)
        rho = max(0.05, rho - 0.001)

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

        elite_archive.append((iteration_best_solution, iteration_best_cost))
        elite_archive = sorted(elite_archive, key=lambda x: x[1])[:5]

        pheromone *= (1 - rho)
        for route, _ in elite_archive:
            for r in route:
                for i in range(len(r) - 1):
                    pheromone[id_to_index[r[i]]][id_to_index[r[i + 1]]] += q / route_cost(r, dist_matrix, id_to_index)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    print("Best Solution Cost:", best_cost)
    print("Worst Solution Cost:", worst_cost)
    print("Average Solution Cost:", round(np.mean(all_costs), 2))
    print("Execution Time (s):", duration)

    export_solution(best_solution)
    plot_routes(best_solution, nodes, depot)
    plot_improvement(trend)
    solve_ecvrp_aco.last_worst_cost = worst_cost
    solve_ecvrp_aco.last_avg_cost = round(np.mean(all_costs), 2)


    return best_solution, best_cost, duration
def run_multiple_trials(filepath, runs=10):
    best_costs = []
    worst_costs = []
    avg_costs = []
    times = []

    for i in range(runs):
        print(f"\n--- Trial {i+1} ---")
        routes, best_cost, exec_time = solve_ecvrp_aco(filepath)
        best_costs.append(best_cost)
        times.append(exec_time)

        # These are computed during the function; reuse last run's values
        worst_costs.append(solve_ecvrp_aco.last_worst_cost)
        avg_costs.append(solve_ecvrp_aco.last_avg_cost)

    print("\n===== Summary After 10 Runs =====")
    print("Min Best Cost:", min(best_costs))
    print("Max Worst Cost:", max(worst_costs))
    print("Avg Mean Cost:", round(sum(avg_costs) / len(avg_costs), 2))
    print("Avg Time (s):", round(sum(times) / len(times), 2))

run_multiple_trials("/content/E-n29-k4-s7.evrp", runs=10)

