# Optimized Q-Learning ECVRP Solver - Enhanced for Better Performance

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from collections import defaultdict
import pickle

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
    if len(route) < 2:
        return 0
    return sum(dist_matrix[id_to_index[route[i]]][id_to_index[route[i+1]]] for i in range(len(route) - 1))

# --------------------------- Nearest Neighbor Heuristic ---------------------------
def nearest_neighbor_heuristic(nodes, demands, depot, stations, capacity, energy_capacity, energy_rate, dist_matrix, id_to_index):
    """Generate initial good solution using nearest neighbor"""
    unvisited = set(demands.keys())
    routes = []
    
    while unvisited:
        route = [depot]
        current = depot
        load = capacity
        energy = energy_capacity
        recharges = 0
        
        while True:
            # Find feasible customers
            feasible = [node for node in unvisited 
                       if demands[node] <= load and 
                       energy >= dist_matrix[id_to_index[current]][id_to_index[node]] * energy_rate]
            
            if not feasible:
                # Try charging if needed
                if recharges < 3:
                    nearest_station = None
                    min_dist = float('inf')
                    for station in stations:
                        if energy >= dist_matrix[id_to_index[current]][id_to_index[station]] * energy_rate:
                            dist = dist_matrix[id_to_index[current]][id_to_index[station]]
                            if dist < min_dist:
                                min_dist = dist
                                nearest_station = station
                    
                    if nearest_station:
                        route.append(nearest_station)
                        current = nearest_station
                        energy = energy_capacity
                        recharges += 1
                        continue
                break
            
            # Choose nearest feasible customer
            nearest = min(feasible, key=lambda x: dist_matrix[id_to_index[current]][id_to_index[x]])
            route.append(nearest)
            load -= demands[nearest]
            energy -= dist_matrix[id_to_index[current]][id_to_index[nearest]] * energy_rate
            current = nearest
            unvisited.remove(nearest)
        
        route.append(depot)
        routes.append(route)
    
    return routes

# --------------------------- Enhanced State Representation ---------------------------
class EVRPState:
    def __init__(self, current_node, unvisited, load, energy, recharges, route_length):
        self.current_node = current_node
        self.unvisited = frozenset(unvisited)
        self.load = load
        self.energy = energy
        self.recharges = recharges
        self.route_length = route_length
    
    def __hash__(self):
        # More precise discretization
        energy_level = int(self.energy / 5)   # Finer energy discretization
        load_level = int(self.load / 3)       # Finer load discretization
        # Include route length for better state differentiation
        route_level = min(self.route_length // 3, 10)
        return hash((self.current_node, len(self.unvisited), energy_level, load_level, self.recharges, route_level))
    
    def __eq__(self, other):
        return (self.current_node == other.current_node and 
                self.unvisited == other.unvisited and
                abs(self.energy - other.energy) < 5 and
                abs(self.load - other.load) < 3 and
                self.recharges == other.recharges)

# --------------------------- Optimized Q-Learning Agent ---------------------------
class OptimizedQLearningEVRP:
    def __init__(self, nodes, demands, depot, stations, capacity, energy_capacity, energy_rate, dist_matrix, id_to_index):
        self.nodes = nodes
        self.demands = demands
        self.depot = depot
        self.stations = stations
        self.capacity = capacity
        self.energy_capacity = energy_capacity
        self.energy_rate = energy_rate
        self.dist_matrix = dist_matrix
        self.id_to_index = id_to_index
        
        # Enhanced Q-learning parameters
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.2        # Higher learning rate
        self.discount_factor = 0.9      # Slightly lower discount
        self.epsilon = 0.9              # Start with more exploration
        self.epsilon_decay = 0.9995     # Slower decay
        self.epsilon_min = 0.05         # Higher minimum for continued exploration
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_costs = []
        self.best_cost = float('inf')
        self.best_routes = None
        
        # Initialize with heuristic solution
        self.initialize_q_values()
        
    def initialize_q_values(self):
        """Initialize Q-values using nearest neighbor heuristic knowledge"""
        heuristic_routes = nearest_neighbor_heuristic(
            self.nodes, self.demands, self.depot, self.stations, 
            self.capacity, self.energy_capacity, self.energy_rate, 
            self.dist_matrix, self.id_to_index
        )
        
        # Set initial Q-values based on heuristic performance
        heuristic_cost = sum(route_cost(route, self.dist_matrix, self.id_to_index) for route in heuristic_routes)
        print(f"Heuristic baseline cost: {heuristic_cost:.2f}")
        
        # Use this to initialize some Q-values
        for route in heuristic_routes:
            for i in range(len(route) - 1):
                # Simulate state and action
                current = route[i]
                next_node = route[i + 1]
                action = ('visit', next_node) if next_node in self.demands else ('depot', next_node)
                
                # Create dummy state
                dummy_state = EVRPState(current, set(), self.capacity, self.energy_capacity, 0, i)
                initial_value = 50.0 / (1 + self.dist_matrix[self.id_to_index[current]][self.id_to_index[next_node]])
                self.q_table[dummy_state][action] = initial_value
    
    def get_valid_actions(self, state):
        """Get all valid actions from current state with improved logic"""
        actions = []
        
        # Action 1: Visit unvisited customers (prioritize by distance and demand)
        customer_actions = []
        for node in state.unvisited:
            if (self.demands[node] <= state.load and 
                state.energy >= self.dist_matrix[self.id_to_index[state.current_node]][self.id_to_index[node]] * self.energy_rate):
                customer_actions.append(('visit', node))
        
        # Sort customer actions by attractiveness (closer + higher demand)
        customer_actions.sort(key=lambda x: (
            self.dist_matrix[self.id_to_index[state.current_node]][self.id_to_index[x[1]]] / 
            max(self.demands[x[1]], 1)
        ))
        actions.extend(customer_actions)
        
        # Action 2: Go to charging station (only if really needed)
        if state.recharges < 3 and state.energy < self.energy_capacity * 0.3:  # Only when energy is low
            for station in self.stations:
                if (state.energy >= self.dist_matrix[self.id_to_index[state.current_node]][self.id_to_index[station]] * self.energy_rate and
                    station != state.current_node):
                    actions.append(('charge', station))
                    break  # Only consider nearest charging station
        
        # Action 3: Return to depot
        if state.current_node != self.depot:
            actions.append(('depot', self.depot))
        
        return actions if actions else [('depot', self.depot)]
    
    def calculate_reward(self, state, action, next_state):
        """Enhanced reward function with better incentives"""
        action_type, target_node = action
        
        # Distance cost (normalized)
        distance = self.dist_matrix[self.id_to_index[state.current_node]][self.id_to_index[target_node]]
        max_distance = np.max(self.dist_matrix)
        distance_penalty = -(distance / max_distance) * 10
        
        if action_type == 'visit':
            # Strong reward for visiting customers
            base_reward = 100
            # Bonus for high demand customers
            demand_bonus = (self.demands[target_node] / max(self.demands.values())) * 20
            # Efficiency bonus (prefer closer customers)
            efficiency_bonus = max(0, 10 - distance / max_distance * 10)
            # Progress bonus
            progress_bonus = (len(self.demands) - len(state.unvisited)) * 2
            return base_reward + demand_bonus + efficiency_bonus + progress_bonus + distance_penalty
            
        elif action_type == 'charge':
            # Penalty for charging, but less if energy is really low
            energy_ratio = state.energy / self.energy_capacity
            if energy_ratio < 0.2:
                return -5 + distance_penalty  # Necessary charging
            else:
                return -20 + distance_penalty  # Unnecessary charging
            
        elif action_type == 'depot':
            if len(state.unvisited) == 0:
                # Big reward for completing all customers
                completion_reward = 200
                # Efficiency bonus for shorter routes
                route_efficiency = max(0, 50 - state.route_length * 2)
                return completion_reward + route_efficiency + distance_penalty
            else:
                # Penalty for returning with unvisited customers
                penalty = -50 - len(state.unvisited) * 10
                return penalty + distance_penalty
        
        return distance_penalty
    
    def choose_action(self, state, valid_actions):
        """Enhanced action selection with better exploration"""
        if random.random() < self.epsilon:
            # Intelligent exploration: bias towards visiting customers
            customer_actions = [a for a in valid_actions if a[0] == 'visit']
            if customer_actions and random.random() < 0.7:
                return random.choice(customer_actions)
            else:
                return random.choice(valid_actions)
        else:
            # Exploitation: choose best action with tie-breaking
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if abs(q - max_q) < 1e-6]
            
            # Tie-breaking: prefer visiting customers
            customer_best = [a for a in best_actions if a[0] == 'visit']
            if customer_best:
                return random.choice(customer_best)
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, valid_next_actions):
        """Enhanced Q-value update with experience replay"""
        current_q = self.q_table[state][action]
        
        if valid_next_actions:
            max_next_q = max([self.q_table[next_state][next_action] for next_action in valid_next_actions])
        else:
            max_next_q = 0
        
        # Enhanced update with adaptive learning rate
        td_error = reward + self.discount_factor * max_next_q - current_q
        adaptive_lr = self.learning_rate * (1 + abs(td_error) / 100)  # Higher LR for larger errors
        
        new_q = current_q + min(adaptive_lr, 0.5) * td_error
        self.q_table[state][action] = new_q
    
    def generate_episode(self):
        """Generate episode with improved termination conditions"""
        unvisited = set(self.demands.keys())
        all_routes = []
        total_cost = 0
        total_reward = 0
        
        route_count = 0
        max_routes = 8  # Limit number of routes
        
        while unvisited and route_count < max_routes:
            route = [self.depot]
            current_node = self.depot
            load = self.capacity
            energy = self.energy_capacity
            recharges = 0
            route_reward = 0
            
            trajectory = []
            steps = 0
            max_steps = 30  # Limit steps per route
            
            while steps < max_steps:
                state = EVRPState(current_node, unvisited, load, energy, recharges, len(route))
                valid_actions = self.get_valid_actions(state)
                
                if not valid_actions:
                    break
                
                action = self.choose_action(state, valid_actions)
                action_type, target_node = action
                
                route.append(target_node)
                
                # Update state
                if action_type == 'visit':
                    load -= self.demands[target_node]
                    unvisited.remove(target_node)
                elif action_type == 'charge':
                    energy = self.energy_capacity
                    recharges += 1
                
                energy -= self.dist_matrix[self.id_to_index[current_node]][self.id_to_index[target_node]] * self.energy_rate
                current_node = target_node
                
                next_state = EVRPState(current_node, unvisited, load, energy, recharges, len(route))
                next_valid_actions = self.get_valid_actions(next_state)
                
                reward = self.calculate_reward(state, action, next_state)
                trajectory.append((state, action, reward, next_state, next_valid_actions))
                
                route_reward += reward
                steps += 1
                
                # End route conditions
                if (action_type == 'depot' or 
                    not any(act[0] == 'visit' for act in next_valid_actions) or
                    energy < 0):
                    break
            
            # Update Q-values for this route
            for state, action, reward, next_state, next_valid_actions in trajectory:
                self.update_q_value(state, action, reward, next_state, next_valid_actions)
            
            route_cost_val = route_cost(route, self.dist_matrix, self.id_to_index)
            total_cost += route_cost_val
            total_reward += route_reward
            
            all_routes.append(route)
            route_count += 1
        
        # Track best solution
        if total_cost < self.best_cost and len(unvisited) == 0:
            self.best_cost = total_cost
            self.best_routes = all_routes.copy()
        
        return all_routes, total_cost, total_reward
    
    def train(self, episodes=2000):
        """Enhanced training with adaptive parameters"""
        print(f"Training optimized Q-learning agent for {episodes} episodes...")
        
        no_improvement = 0
        last_best = float('inf')
        
        for episode in range(episodes):
            routes, cost, reward = self.generate_episode()
            
            self.episode_costs.append(cost)
            self.episode_rewards.append(reward)
            
            # Adaptive parameter adjustment
            if episode > 100:
                recent_improvement = np.mean(self.episode_costs[-50:]) < np.mean(self.episode_costs[-100:-50])
                if not recent_improvement:
                    no_improvement += 1
                    if no_improvement > 100:
                        # Increase exploration
                        self.epsilon = min(0.3, self.epsilon * 1.1)
                        no_improvement = 0
                else:
                    no_improvement = 0
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % 200 == 0:
                avg_cost = np.mean(self.episode_costs[-100:]) if len(self.episode_costs) >= 100 else np.mean(self.episode_costs)
                print(f"Episode {episode}, Avg Cost: {avg_cost:.2f}, Best Cost: {self.best_cost:.2f}, Epsilon: {self.epsilon:.3f}")
        
        print(f"Training completed! Best cost found: {self.best_cost:.2f}")
    
    def get_best_solution(self):
        """Get best solution found during training"""
        if self.best_routes:
            return self.best_routes, self.best_cost
        else:
            # Fallback: generate solution with pure exploitation
            old_epsilon = self.epsilon
            self.epsilon = 0.0
            routes, cost, _ = self.generate_episode()
            self.epsilon = old_epsilon
            return routes, cost

# --------------------------- 2-opt Local Search ---------------------------
def two_opt_improvement(routes, dist_matrix, id_to_index, max_iterations=100):
    """Apply 2-opt local search to improve routes"""
    improved_routes = []
    
    for route in routes:
        if len(route) <= 4:  # Too short for 2-opt
            improved_routes.append(route)
            continue
            
        best_route = route[:]
        best_cost = route_cost(route, dist_matrix, id_to_index)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Skip if nodes are adjacent
                    if j - i == 1:
                        continue
                    
                    # Create new route by reversing segment
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = route_cost(new_route, dist_matrix, id_to_index)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
            
            route = best_route
            iterations += 1
        
        improved_routes.append(best_route)
    
    return improved_routes

# --------------------------- Visualization ---------------------------
def plot_routes(routes, nodes, depot, title="Q-Learning Routes"):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(routes)))
    
    for i, route in enumerate(routes):
        x = [nodes[n][0] for n in route]
        y = [nodes[n][1] for n in route]
        plt.plot(x, y, marker='o', color=colors[i], linewidth=2, markersize=6, label=f'Route {i+1}')
    
    plt.plot(nodes[depot][0], nodes[depot][1], 'rs', markersize=12, label='Depot')
    plt.title(title, fontsize=14)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_progress(agent):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cost improvement
    ax1.plot(agent.episode_costs, alpha=0.7)
    if len(agent.episode_costs) > 100:
        window = 100
        moving_avg = [np.mean(agent.episode_costs[max(0, i-window):i+1]) for i in range(len(agent.episode_costs))]
        ax1.plot(moving_avg, color='red', linewidth=2, label='Moving Average')
        ax1.axhline(y=383, color='green', linestyle='--', label='Optimal (383)')
        ax1.legend()
    ax1.set_title('Cost per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Cost')
    ax1.grid(True, alpha=0.3)
    
    # Plot rewards
    ax2.plot(agent.episode_rewards, alpha=0.7, color='green')
    if len(agent.episode_rewards) > 100:
        window = 100
        moving_avg = [np.mean(agent.episode_rewards[max(0, i-window):i+1]) for i in range(len(agent.episode_rewards))]
        ax2.plot(moving_avg, color='darkgreen', linewidth=2, label='Moving Average')
        ax2.legend()
    ax2.set_title('Reward per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --------------------------- Export ---------------------------
def export_solution(routes, filename='optimized_qlearning_solution.sol'):
    total_cost = 0
    with open(filename, 'w') as f:
        f.write("Optimized Q-Learning ECVRP Solution\n")
        f.write("=" * 40 + "\n")
        for i, route in enumerate(routes):
            route_str = 'Route #{0}: {1}\n'.format(i+1, ' -> '.join(map(str, route)))
            f.write(route_str)
        f.write(f"\nTotal Routes: {len(routes)}\n")

# --------------------------- Main Optimized Q-Learning Solver ---------------------------
def solve_ecvrp_qlearning_optimized(filepath, episodes=2000, runs=10):
    """Optimized Q-learning solver for ECVRP"""
    print("Loading ECVRP instance...")
    nodes, demands, depot, stations, capacity, energy_capacity, energy_rate = parse_evrp_file(filepath)
    
    print(f"Instance details:")
    print(f"- Nodes: {len(nodes)}")
    print(f"- Customers: {len(demands)}")  
    print(f"- Charging stations: {len(stations)}")
    print(f"- Capacity: {capacity}")
    print(f"- Energy capacity: {energy_capacity}")
    print(f"- Target optimal: 383")
    
    node_ids = list(nodes.keys())
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    dist_matrix = calculate_distance_matrix(nodes, id_to_index)
    
    best_cost = float('inf')
    best_routes = None
    all_costs = []
    all_times = []
    
    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        start_time = time.time()
        
        # Create and train optimized agent
        agent = OptimizedQLearningEVRP(nodes, demands, depot, stations, capacity, 
                                     energy_capacity, energy_rate, dist_matrix, id_to_index)
        
        agent.train(episodes)
        
        # Get best solution and apply local search
        routes, cost = agent.get_best_solution()
        
        # Apply 2-opt improvement
        improved_routes = two_opt_improvement(routes, dist_matrix, id_to_index)
        improved_cost = sum(route_cost(route, dist_matrix, id_to_index) for route in improved_routes)
        
        end_time = time.time()
        duration = end_time - start_time
        
        all_costs.append(improved_cost)
        all_times.append(duration)
        
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_routes = improved_routes
            best_agent = agent
        
        print(f"Run {run + 1} - Original: {cost:.2f}, Improved: {improved_cost:.2f}, Time: {duration:.2f}s")
        print(f"Gap from optimal: {((improved_cost - 383) / 383 * 100):.1f}%")
    
    # Results summary
    print("\n" + "="*60)
    print("OPTIMIZED Q-LEARNING ECVRP RESULTS")
    print("="*60)
    print(f"Optimal Cost (Target):     383.00")
    print(f"Best Found Cost:          {best_cost:.2f}")
    print(f"Gap from Optimal:         {((best_cost - 383) / 383 * 100):.1f}%")
    print(f"Average Cost:             {np.mean(all_costs):.2f}")
    print(f"Worst Cost:               {max(all_costs):.2f}")
    print(f"Average Time:             {np.mean(all_times):.2f}s")
    print(f"Number of Routes:         {len(best_routes)}")
    print("="*60)
    
    # Export and visualize
    export_solution(best_routes)
    plot_routes(best_routes, nodes, depot, f"Best Q-Learning Solution (Cost: {best_cost:.2f})")
    plot_learning_progress(best_agent)
    print(f"Mean of all run costs: {np.mean(all_costs):.2f}")

    

    return best_routes, best_cost, np.mean(all_times), np.mean(all_costs)


# --------------------------- Usage Example ---------------------------
if __name__ == "__main__":
    # Example usage - replace with your file path
    filepath = "/content/E-n29-k4-s7.evrp"  # Use your EVRP file path
    
    # Solve with optimized Q-learning
    routes, cost, avg_time, avg_cost = solve_ecvrp_qlearning_optimized(filepath, episodes=2000, runs=10)

    
    print(f"\nFINAL RESULTS:")
    print(f"Best solution cost: {cost:.2f}")
    print(f"Mean of average costs across runs: {avg_cost:.2f}")
    print(f"Gap from optimal (383): {((cost - 383) / 383 * 100):.1f}%")
    print(f"Number of routes: {len(routes)}")
    print(f"Average execution time: {avg_time:.2f} seconds")
