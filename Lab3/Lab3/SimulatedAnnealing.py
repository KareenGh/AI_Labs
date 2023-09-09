# import numpy as np
#
#
# class SimulatedAnnealing:
#     def __init__(self, cvrp, T=100, alpha=0.99, max_iter=1000):
#         self.cvrp = cvrp
#         self.T = T
#         self.alpha = alpha
#         self.max_iter = max_iter
#
#     def solve(self):
#         # Initialize solution randomly
#         solution = np.random.permutation(self.cvrp.location - 1) + 1  # +1 because depot is 0
#         best_solution, best_cost = self.cvrp.calc_path_cost_sa(solution)
#
#         # ensure each route starts and ends with the depot
#         for route in best_solution:
#             if route[0] != 0:  # add depot at the start if not present
#                 route.insert(0, 0)
#             if route[-1] != 0:  # add depot at the end if not present
#                 route.append(0)
#
#         for _ in range(self.max_iter):
#             # Generate neighbor solution
#             neighbor = solution.copy()
#
#             # Exclude depot from the swap
#             depot_indices = [i for i, x in enumerate(neighbor) if x == 0]
#             city_indices = list(set(range(len(neighbor))) - set(depot_indices))
#             i, j = np.random.choice(city_indices, size=2, replace=False)
#
#             neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap two cities
#
#             # Calculate cost difference
#             neighbor_solution, neighbor_cost = self.cvrp.calc_path_cost_sa(neighbor)
#             cost_diff = neighbor_cost - best_cost
#
#             # Decide whether to move to neighbor
#             if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / self.T):
#                 solution = neighbor
#                 if neighbor_cost < best_cost:
#                     best_solution = neighbor_solution
#                     best_cost = neighbor_cost
#
#             # Decrease temperature
#             self.T *= self.alpha
#
#         # ensure each route starts and ends with the depot
#         for route in best_solution:
#             if route[0] != 0:  # add depot at the start if not present
#                 route.insert(0, 0)
#             if route[-1] != 0:  # add depot at the end if not present
#                 route.append(0)
#
#         return best_solution, best_cost
#
#
# import numpy as np
# import math
# import random
#
# from helpers import initial_solution
#
#
# class SimulatedAnnealing:
#     def __init__(self, cvrp, T=100, alpha=0.99, max_iter=1000):
#         self.cvrp = cvrp
#         self.T = T
#         self.alpha = alpha
#         self.max_iter = max_iter
#
#     def solve(self, init_solution):
#         # Initialize solution using the nearest neighbor heuristic
#         solution = initial_solution(self.cvrp)
#         # solution = init_solution
#         best_solution = solution
#         best_cost = self.cvrp.calculate_total_cost(solution)
#
#         for _ in range(self.max_iter):
#             neighbor = self.generate_neighbor(best_solution.copy())
#             neighbor_cost = self.cvrp.calculate_total_cost(neighbor)
#             cost_diff = neighbor_cost - best_cost
#
#             # Decide whether to move to neighbor
#             if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / self.T):
#                 best_solution = neighbor
#                 best_cost = neighbor_cost
#
#             # Decrease temperature
#             self.T *= self.alpha
#
#         # ensure each route starts and ends with the depot
#         for route in best_solution:
#             if route[0] != 0:
#                 route.insert(0, 0)
#             if route[-1] != 0:
#                 route.append(0)
#
#         return best_solution, best_cost
#
#     def generate_neighbor(self, solution):
#         # Make a copy of the solution to avoid modifying the original
#         solution = [list(route) for route in solution]
#
#         # Pick a random route
#         route_index = random.randint(0, len(solution) - 1)
#         route = solution[route_index]
#
#         # Ensure there are at least 2 cities in the route to swap
#         if len(route) > 3:  # must have at least 1 city (excluding depot)
#             # Randomly choose two cities within the same route and swap them
#             i, j = random.sample(range(1, len(route) - 1), 2)  # exclude depot
#             route[i], route[j] = route[j], route[i]
#
#         return solution
#
#


import numpy as np
import random
from helpers import initial_solution


# class SimulatedAnnealing:
#     def __init__(self, cvrp, T=100, alpha=0.99, max_iter=1000, restarts=5):
#         self.cvrp = cvrp
#         self.T = T
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.restarts = restarts
#
#     def solve(self, init_solution):
#         best_solution = initial_solution(self.cvrp)
#         best_cost = self.cvrp.calculate_total_cost(best_solution)
#
#         for _ in range(self.restarts):
#             solution = initial_solution(self.cvrp)
#             for _ in range(self.max_iter):
#                 neighbor = self.generate_neighbor(solution.copy())
#                 neighbor_cost = self.cvrp.calculate_total_cost(neighbor)
#                 cost_diff = neighbor_cost - best_cost
#
#                 # Decide whether to move to neighbor
#                 if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / self.T):
#                     solution = neighbor
#                     if neighbor_cost < best_cost:
#                         best_solution = neighbor
#                         best_cost = neighbor_cost
#
#                 # Decrease temperature
#                 self.T *= self.alpha
#
#             # ensure each route starts and ends with the depot
#             for route in best_solution:
#                 if route[0] != 0:
#                     route.insert(0, 0)
#                 if route[-1] != 0:
#                     route.append(0)
#
#         return best_solution, best_cost
#
#     def generate_neighbor(self, solution):
#         # Make a copy of the solution to avoid modifying the original
#         solution = [list(route) for route in solution]
#
#         # Pick a random route
#         route_index = random.randint(0, len(solution) - 1)
#         route = solution[route_index]
#
#         # Ensure there are at least 2 cities in the route to swap
#         if len(route) > 3:  # must have at least 1 city (excluding depot)
#             # Randomly choose two cities within the same route and swap them
#             i, j = random.sample(range(1, len(route) - 1), 2)  # exclude depot
#             route[i], route[j] = route[j], route[i]
#
#         return solution
#
class SimulatedAnnealing:
    def __init__(self, cvrp, T=100, alpha=0.99, max_iter=1000, restarts=5, tabu_tenure=10):
        self.cvrp = cvrp
        self.T = T
        self.alpha = alpha
        self.max_iter = max_iter
        self.restarts = restarts
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
        self.best_costs = []

    def solve(self, init_solution):
        best_solution = initial_solution(self.cvrp)
        best_cost = self.cvrp.calculate_total_cost(best_solution)

        for _ in range(self.restarts):
            solution = initial_solution(self.cvrp)
            for _ in range(self.max_iter):
                neighbor = self.generate_neighbor(solution.copy())
                neighbor_cost = self.cvrp.calculate_total_cost(neighbor)
                cost_diff = neighbor_cost - best_cost

                # Decide whether to move to neighbor
                if (cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / self.T)) and neighbor not in self.tabu_list:
                    solution = neighbor
                    if neighbor_cost < best_cost:
                        best_solution = neighbor
                        best_cost = neighbor_cost
                        self.best_costs.append(best_cost)

                # Add to tabu list and remove oldest entries if it exceeds tenure
                self.tabu_list.append(neighbor)
                if len(self.tabu_list) > self.tabu_tenure:
                    self.tabu_list.pop(0)

                # Decrease temperature
                self.T *= self.alpha

            # ensure each route starts and ends with the depot
            for route in best_solution:
                if route[0] != 0:
                    route.insert(0, 0)
                if route[-1] != 0:
                    route.append(0)

        return best_solution, best_cost

    def generate_neighbor(self, solution):
        solution = [list(route) for route in solution]
        neighborhood = random.choice([self.swap_in_route, self.swap_between_routes, self.reverse_subroute])
        solution = neighborhood(solution)

        return solution

    def swap_in_route(self, solution):
        route_index = random.randint(0, len(solution) - 1)
        route = solution[route_index]
        # Ensure there are at least 2 cities in the route to swap
        if len(route) > 3:
            # Randomly choose two cities within the same route and swap them
            i, j = random.sample(range(1, len(route) - 1), 2)
            route[i], route[j] = route[j], route[i]
        return solution

    def swap_between_routes(self, solution):
        # Ensure there are at least 2 routes
        if len(solution) > 2:
            # Randomly choose two routes and two cities, then swap the cities
            i, j = random.sample(range(len(solution)), 2)
            route1, route2 = solution[i], solution[j]
            if len(route1) > 2 and len(route2) > 2:  # each route must have at least 1 city
                city1, city2 = random.choice(route1[1:-1]), random.choice(route2[1:-1])
                index1, index2 = route1.index(city1), route2.index(city2)
                route1[index1], route2[index2] = city2, city1
        return solution

    def reverse_subroute(self, solution):
        # Pick a random route
        route_index = random.randint(0, len(solution) - 1)
        route = solution[route_index]
        if len(route) > 4:
            # Randomly choose a subroute and reverse it
            i, j = sorted(random.sample(range(1, len(route) - 1), 2))  # exclude depot
            route[i:j] = reversed(route[i:j])
        return solution
