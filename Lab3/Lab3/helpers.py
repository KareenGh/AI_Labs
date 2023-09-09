import random

import numpy as np
from scipy.optimize import differential_evolution


def get_neighborhood(solution):
    neighborhood = []
    for i in range(1, len(solution) - 1):
        for j in range(i + 1, len(solution)):
            new_solution = solution[:i] + solution[i:j][::-1] + solution[j:]
            neighborhood.append(new_solution)
    return neighborhood


# def initial_solution(cvrp):
#     routes = []
#     visited = set()
#     for i in range(cvrp.location):
#         if i not in visited:
#             route = [i]
#             load = cvrp.demands[i]
#             while True:
#                 j = nearest_neighbor(route[-1], cvrp, visited)
#                 if j is None or load + cvrp.demands[j] > cvrp.capacity:
#                     break
#                 route.append(j)
#                 load += cvrp.demands[j]
#                 visited.add(j)
#             routes.append(route)
#     return routes
#
#
# def nearest_neighbor(i, cvrp, visited):
#     neighbors = list(set(range(cvrp.location)) - visited - {i})
#     return min(neighbors, key=lambda j: cvrp.distance_matrix[i][j]) if neighbors else None
#

# def initial_solution(cvrp):
#     routes = []
#     visited = set()
#     for i in range(cvrp.location):  # changed from cvrp.location
#         if i not in visited:
#             route = [i]
#             load = cvrp.demands[i]
#             while True:
#                 j = nearest_neighbor(route[-1], cvrp, visited)
#                 if j is None or load + cvrp.demands[j] > cvrp.capacity:
#                     break
#                 route.append(j)
#                 load += cvrp.demands[j]
#                 visited.add(j)
#             routes.append(route)
#
#     return routes

def initial_solution(cvrp):
    # Generate a random permutation of the nodes (excluding the depot)
    nodes = np.random.permutation(range(1, cvrp.location))

    # Split this solution into routes based on the vehicle capacity constraint
    routes = []
    route = [0]  # start from depot
    load = 0
    for node in nodes:
        if load + cvrp.demands[node] <= cvrp.capacity:
            route.append(node)
            load += cvrp.demands[node]
        else:
            route.append(0)  # return to depot
            routes.append(route)
            route = [0, node]  # start a new route from depot
            load = cvrp.demands[node]
    route.append(0)  # return to depot
    routes.append(route)

    return routes


def nearest_neighbor(i, cvrp, visited):
    neighbors = list(set(range(cvrp.location)) - visited - {i})  # changed from cvrp.location
    return min(neighbors, key=lambda j: cvrp.distance_matrix[i][j]) if neighbors else None

# def nearest_neighbor(i, cvrp, visited, k=1):
#     neighbors = list(set(range(cvrp.location)) - visited - {i})
#     if neighbors:
#         sorted_neighbors = sorted(neighbors, key=lambda j: cvrp.distance_matrix[i][j])
#         top_k_neighbors = sorted_neighbors[:k]
#         return random.choice(top_k_neighbors)
#     else:
#         return None


def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(sum(x**2) / d))
    sum2 = -np.exp(sum(np.cos(c*x) / d))
    return sum1 + sum2 + a + np.exp(1)
