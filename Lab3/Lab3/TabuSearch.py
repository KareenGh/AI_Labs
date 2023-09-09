# # import random
# # import numpy as np
# #
# #
# # class TabuSearch:
# #     def __init__(self, TS_problem):
# #         self.cvrp = TS_problem
# #         self.frequency_matrix = np.zeros((self.cvrp.location, self.cvrp.location))
# #
# #     def get_neighborhood(self, solution):
# #         neighborhood = []
# #         for i in range(1, len(solution) - 1):
# #             for j in range(i + 1, len(solution)):
# #                 new_solution = solution[:i] + solution[i:j][::-1] + solution[j:]
# #                 neighborhood.append(new_solution)
# #         return neighborhood
# #
# #     def calculate_cost_with_penalty(self, route):
# #         cost = self.cvrp.calculate_route_cost(route)
# #         penalty = sum(self.frequency_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
# #         return cost + 0.1 * penalty
# #
# #     def tabu_search(self, initial_solution, min_tabu_tenure, max_tabu_tenure, max_iterations):
# #         best_solution = initial_solution
# #         best_cost = self.cvrp.calculate_total_cost(initial_solution)
# #
# #         current_solution = initial_solution
# #         tabu_list = {}
# #
# #         for iteration in range(max_iterations):
# #             best_neighbor = None
# #             best_neighbor_cost = float('inf')
# #             best_neighbor_route_index = None
# #             best_neighbor_move = None
# #
# #             for route_index, route in enumerate(current_solution):
# #                 neighborhood = self.get_neighborhood(route)
# #                 for neighbor in neighborhood:
# #                     i, j = self.get_move(route, neighbor)
# #                     if i is None or j is None:  # skip neighbors that don't involve a swap
# #                         continue
# #                     cost = self.calculate_cost_with_penalty(neighbor)
# #
# #                     tabu_tenure = random.randint(min_tabu_tenure, max_tabu_tenure)
# #
# #                     if (i, j) not in tabu_list or iteration - tabu_list[(i, j)] > tabu_tenure or cost < best_cost:
# #                         if np.random.rand() < 0.1:  # 10% chance to select a random neighbor
# #                             best_neighbor = random.choice(neighborhood)
# #                             best_neighbor_cost = self.calculate_cost_with_penalty(best_neighbor)
# #                             best_neighbor_route_index = route_index
# #                             best_neighbor_move = i, j
# #                         else:  # 90% chance to select the best neighbor
# #                             if cost < best_neighbor_cost:
# #                                 best_neighbor = neighbor
# #                                 best_neighbor_cost = cost
# #                                 best_neighbor_route_index = route_index
# #                                 best_neighbor_move = i, j
# #
# #             if best_neighbor is None:
# #                 break
# #
# #             current_solution[best_neighbor_route_index] = best_neighbor
# #             tabu_list[best_neighbor_move] = iteration
# #
# #             for i in range(len(best_neighbor) - 1):
# #                 self.frequency_matrix[best_neighbor[i]][best_neighbor[i + 1]] += 1
# #
# #             current_cost = self.cvrp.calculate_total_cost(current_solution)
# #             if current_cost < best_cost:
# #                 best_solution = current_solution.copy()
# #                 best_cost = current_cost
# #
# #         # ensure each route starts and ends with the depot
# #         for route in best_solution:
# #             if route[0] != 0:  # add depot at the start if not present
# #                 route.insert(0, 0)
# #             if route[-1] != 0:  # add depot at the end if not present
# #                 route.append(0)
# #
# #         return best_solution
# #
# #     def get_move(self, route, neighbor):
# #         # find the indices of the first and last elements that are different
# #         diff = [i for i in range(len(route)) if route[i] != neighbor[i]]
# #         if len(diff) >= 2:
# #             return diff[0], diff[-1]
# #         else:
# #             return None, None
# #
# #
# #
# import random
# import numpy as np
#
#
# class TabuSearch:
#     def __init__(self, TS_problem):
#         self.cvrp = TS_problem
#         self.frequency_matrix = np.zeros((self.cvrp.location, self.cvrp.location))
#
#     def get_neighborhood(self, solution):
#         neighborhood = []
#         for route_index, route in enumerate(solution):
#             for i in range(1, len(route) - 1):
#                 for j in range(i + 1, len(route)):
#                     new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
#                     new_solution = solution[:route_index] + [new_route] + solution[route_index + 1:]
#                     neighborhood.append(new_solution)
#         return neighborhood
#
#     def calculate_cost_with_penalty(self, route):
#         cost = self.cvrp.calculate_route_cost(route)
#         penalty = sum(self.frequency_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
#         return cost + 0.1 * penalty
#
#     def tabu_search(self, initial_solution, min_tabu_tenure, max_tabu_tenure, max_iterations):
#         best_solution = initial_solution
#         best_cost = self.cvrp.calculate_total_cost(initial_solution)
#
#         current_solution = initial_solution
#         tabu_list = {}
#
#         for iteration in range(max_iterations):
#             best_neighbor = None
#             best_neighbor_cost = float('inf')
#             best_neighbor_move = None
#
#             neighborhood = self.get_neighborhood(current_solution)
#             for neighbor in neighborhood:
#                 for route_index, (route, neighbor_route) in enumerate(zip(current_solution, neighbor)):
#                     i, j = self.get_move(route, neighbor_route)
#                     if i is None or j is None:  # skip neighbors that don't involve a swap
#                         continue
#                     cost = self.calculate_cost_with_penalty(neighbor_route)
#
#                     tabu_tenure = random.randint(min_tabu_tenure, max_tabu_tenure)
#
#                     if (i, j) not in tabu_list or iteration - tabu_list[(i, j)] > tabu_tenure or cost < best_cost:
#                         if cost < best_neighbor_cost:
#                             best_neighbor = neighbor
#                             best_neighbor_cost = cost
#                             best_neighbor_move = i, j
#
#             if best_neighbor is None:
#                 break
#
#             current_solution = best_neighbor
#             tabu_list[best_neighbor_move] = iteration
#
#             for route in best_neighbor:
#                 for i in range(len(route) - 1):
#                     self.frequency_matrix[route[i]][route[i + 1]] += 1
#
#             current_cost = self.cvrp.calculate_total_cost(current_solution)
#             if current_cost < best_cost:
#                 best_solution = current_solution.copy()
#                 best_cost = current_cost
#
#         # ensure each route starts and ends with the depot
#         for route in best_solution:
#             if route[0] != 0:  # add depot at the start if not present
#                 route.insert(0, 0)
#             if route[-1] != 0:  # add depot at the end if not present
#                 route.append(0)
#
#         return best_solution
#
#     def get_move(self, route, neighbor_route):
#         # find the indices of the first and last elements that are different
#         diff = [i for i in range(len(route)) if route[i] != neighbor_route[i]]
#         if len(diff) >= 2:
#             return diff[0], diff[-1]
#         else:
#             return None, None
#


import random
import numpy as np


class TabuSearch:
    def __init__(self, TS_problem):
        self.cvrp = TS_problem
        self.frequency_matrix = np.zeros((self.cvrp.location, self.cvrp.location))
        self.best_costs = []

    def get_neighborhood(self, solution):
        neighborhood = []
        for i in range(1, len(solution) - 1):
            for j in range(i + 1, len(solution)):
                new_solution = solution.copy()
                new_solution[i:j] = reversed(solution[i:j])  # this is a 2-opt Swap
                neighborhood.append(new_solution)
        return neighborhood

    def calculate_cost_with_penalty(self, route):
        cost = self.cvrp.calculate_route_cost(route)
        penalty = sum(self.frequency_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        return cost + 0.1 * penalty

    def tabu_search(self, initial_solution, min_tabu_tenure, max_tabu_tenure, max_iterations):
        best_solution = initial_solution
        best_cost = self.cvrp.calculate_total_cost(initial_solution)

        current_solution = initial_solution
        tabu_list = {}

        for iteration in range(max_iterations):
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_neighbor_route_index = None
            best_neighbor_move = None

            for route_index, route in enumerate(current_solution):
                neighborhood = self.get_neighborhood(route)
                for neighbor in neighborhood:
                    i, j = self.get_move(route, neighbor)
                    if i is None or j is None:  # skip neighbors that don't involve a swap
                        continue
                    cost = self.calculate_cost_with_penalty(neighbor)

                    tabu_tenure = tabu_tenure = self.adaptive_tabu_tenure(cost, best_cost, min_tabu_tenure, max_tabu_tenure)
                                                #random.randint(min_tabu_tenure, max_tabu_tenure)

                    if (i, j) not in tabu_list or iteration - tabu_list[(i, j)] > tabu_tenure or cost < best_cost:
                        if cost < best_cost:
                            tabu_tenure = min_tabu_tenure
                        else:
                            tabu_tenure = max_tabu_tenure
                        if np.random.rand() < 0.1:  # 10% chance to select a random neighbor
                            best_neighbor = random.choice(neighborhood)
                            best_neighbor_cost = self.calculate_cost_with_penalty(best_neighbor)
                            best_neighbor_route_index = route_index
                            best_neighbor_move = i, j
                        else:  # 90% chance to select the best neighbor
                            if cost < best_neighbor_cost:
                                best_neighbor = neighbor
                                best_neighbor_cost = cost
                                best_neighbor_route_index = route_index
                                best_neighbor_move = i, j

            if best_neighbor is None:
                break

            current_solution[best_neighbor_route_index] = best_neighbor
            tabu_list[best_neighbor_move] = iteration + tabu_tenure  # add tabu_tenure to the current iteration

            for i in range(len(best_neighbor) - 1):
                self.frequency_matrix[best_neighbor[i]][best_neighbor[i + 1]] += 1

            current_cost = self.cvrp.calculate_total_cost(current_solution)
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                self.best_costs.append(best_cost)
            self.best_costs.append(best_cost)

        # ensure each route starts and ends with the depot
        for route in best_solution:
            if route[0] != 0:
                route.insert(0, 0)
            if route[-1] != 0:
                route.append(0)

        return best_solution

    def get_move(self, route, neighbor):
        # find the indices of the first and last elements that are different
        diff = [i for i in range(len(route)) if route[i] != neighbor[i]]
        if len(diff) >= 2:
            return diff[0], diff[-1]
        else:
            return None, None

    def adaptive_tabu_tenure(self, current_cost, best_cost, min_tabu_tenure, max_tabu_tenure):
        if current_cost > 1.2 * best_cost:
            return max_tabu_tenure
        elif current_cost > best_cost:
            return min_tabu_tenure + (max_tabu_tenure - min_tabu_tenure) * (current_cost - best_cost) / (
                        0.2 * best_cost)
        else:
            return min_tabu_tenure

#
# import random
# import numpy as np
#
#
# class TabuSearch:
#     def __init__(self, TS_problem):
#         self.cvrp = TS_problem
#         self.frequency_matrix = np.zeros((self.cvrp.location, self.cvrp.location))
#
#     def get_neighborhood(self, solution, operator):
#         neighborhood = []
#         for i in range(1, len(solution) - 1):
#             for j in range(i + 1, len(solution)):
#                 if operator == "reversion":
#                     new_solution = solution[:i] + solution[i:j][::-1] + solution[j:]
#                 elif operator == "relocate":
#                     new_solution = solution[:i] + solution[j:j + 1] + solution[i:j] + solution[j + 1:] \
#                         if i < j else solution[:j] + solution[i + 1:i + 2] + solution[j:i] + solution[i + 2:]
#                 elif operator == "swap":
#                     new_solution = solution[:i] + solution[j:j + 1] + solution[i + 1:j] \
#                                    + solution[i:i + 1] + solution[j + 1:]
#                 neighborhood.append(new_solution)
#         return neighborhood
#
#     def calculate_cost_with_penalty(self, route):
#         cost = self.cvrp.calculate_route_cost(route)
#         penalty = sum(self.frequency_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
#         return cost + 0.1 * penalty
#
#     def tabu_search(self, initial_solution, min_tabu_tenure, max_tabu_tenure, max_iterations):
#         best_solution = initial_solution
#         best_cost = self.cvrp.calculate_total_cost(initial_solution)
#
#         current_solution = initial_solution
#         tabu_list = {}
#
#         for iteration in range(max_iterations):
#             best_neighbor = None
#             best_neighbor_cost = float('inf')
#             best_neighbor_route_index = None
#             best_neighbor_move = None
#
#             for route_index, route in enumerate(current_solution):
#                 operator = np.random.choice(["reversion", "relocate", "swap"], p=[0.4, 0.3, 0.3])
#                 neighborhood = self.get_neighborhood(route, operator)
#                 for neighbor in neighborhood:
#                     i, j = self.get_move(route, neighbor)
#                     if i is None or j is None:  # skip neighbors that don't involve a swap
#                         continue
#                     cost = self.calculate_cost_with_penalty(neighbor)
#
#                     tabu_tenure = random.randint(min_tabu_tenure, max_tabu_tenure)
#
#                     if (i, j) not in tabu_list or iteration - tabu_list[(i, j)] > tabu_tenure or cost < best_cost:
#                         if cost < best_cost:
#                             tabu_tenure = min_tabu_tenure
#                         else:
#                             tabu_tenure = max_tabu_tenure
#                         if np.random.rand() < 0.1:  # 10% chance to select a random neighbor
#                             best_neighbor = random.choice(neighborhood)
#                             best_neighbor_cost = self.calculate_cost_with_penalty(best_neighbor)
#                             best_neighbor_route_index = route_index
#                             best_neighbor_move = i, j
#                         else:  # 90% chance to select the best neighbor
#                             if cost < best_neighbor_cost:
#                                 best_neighbor = neighbor
#                                 best_neighbor_cost = cost
#                                 best_neighbor_route_index = route_index
#                                 best_neighbor_move = i, j
#
#             if best_neighbor is None:
#                 break
#
#             current_solution[best_neighbor_route_index] = best_neighbor
#             tabu_list[best_neighbor_move] = iteration + tabu_tenure  # add tabu_tenure to the current iteration
#
#             for i in range(len(best_neighbor) - 1):
#                 self.frequency_matrix[best_neighbor[i]][best_neighbor[i + 1]] += 1
#
#             current_cost = self.cvrp.calculate_total_cost(current_solution)
#             if current_cost < best_cost:
#                 best_solution = current_solution.copy()
#                 best_cost = current_cost
#
#         # ensure each route starts and ends with the depot
#         for route in best_solution:
#             if route[0] != 0:  # add depot at the start if not present
#                 route.insert(0, 0)
#             if route[-1] != 0:  # add depot at the end if not present
#                 route.append(0)
#
#         return best_solution
#
#     def get_move(self, route, neighbor):
#         # find the indices of the first and last elements that are different
#         diff = [i for i in range(len(route)) if route[i] != neighbor[i]]
#         if len(diff) >= 2:
#             return diff[0], diff[-1]
#         else:
#             return None, None


#
# import random
#
#
# class TabuSearch:
#     def __init__(self, ackley):
#         self.ackley = ackley
#         self.dimensions = ackley.dimensions
#         self.bounds = ackley.bounds
#
#     def initial_solution(self):
#         return [random.uniform(bound[0], bound[1]) for bound in self.bounds]
#
#     def get_neighborhood(self, solution):
#         neighborhood = []
#         for i in range(self.dimensions):
#             for _ in range(10):  # generate 10 neighbors per dimension
#                 new_solution = solution.copy()
#                 new_solution[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
#                 neighborhood.append(new_solution)
#         return neighborhood
#
#     def tabu_search(self, min_tabu_tenure, max_tabu_tenure, max_iterations):
#         initial_solution = self.initial_solution()
#         best_solution = initial_solution
#         best_cost = self.ackley.evaluate(initial_solution)
#
#         current_solution = initial_solution
#         tabu_list = {}
#
#         for iteration in range(max_iterations):
#             best_neighbor = None
#             best_neighbor_cost = float('inf')
#
#             neighborhood = self.get_neighborhood(current_solution)
#             for neighbor in neighborhood:
#                 cost = self.ackley.evaluate(neighbor)
#
#                 tabu_tenure = random.randint(min_tabu_tenure, max_tabu_tenure)
#
#                 if tuple(neighbor) not in tabu_list or iteration - tabu_list[tuple(neighbor)] > tabu_tenure or cost < best_cost:
#                     if cost < best_neighbor_cost:
#                         best_neighbor = neighbor
#                         best_neighbor_cost = cost
#
#             if best_neighbor is None:
#                 break
#
#             current_solution = best_neighbor
#             tabu_list[tuple(best_neighbor)] = iteration
#
#             current_cost = self.ackley.evaluate(current_solution)
#             if current_cost < best_cost:
#                 best_solution = current_solution.copy()
#                 best_cost = current_cost
#
#         return best_solution



