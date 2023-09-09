import numpy as np
import helpers


class ALNS:
    def __init__(self, cvrp, destroy_methods, repair_methods, weights, iterations, acceptance_criteria, initial_temperature, cooling_rate):
        self.cvrp = cvrp
        self.destroy_methods = destroy_methods
        self.repair_methods = repair_methods
        self.destroy_weights = dict(zip(destroy_methods, weights))
        self.repair_weights = dict(zip(repair_methods, weights))
        self.iterations = iterations
        self.acceptance_criteria = acceptance_criteria
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.no_improve = 0
        self.best_costs = []

    def select_method(self, methods, weights):
        return np.random.choice(methods, p=weights / np.sum(weights))

    def update_weights(self, method, success, weights, rate=0.2):
        weights[method] += rate if success else -rate

    def solve(self):
        # Generate an initial solution
        solution = helpers.initial_solution(self.cvrp)
        best_solution = solution
        best_cost = self.cvrp.calculate_total_cost(solution)
        temperature = self.initial_temperature

        for iteration in range(self.iterations):
            # Decrease the temperature
            if self.no_improve > 10:
                temperature *= 2 * self.cooling_rate
            else:
                temperature *= self.cooling_rate

            # Select destroy and repair methods
            destroy = self.select_method(self.destroy_methods, list(self.destroy_weights.values()))
            repair = self.select_method(self.repair_methods, list(self.repair_weights.values()))

            # Apply destroy and repair methods
            destroyed_solution = destroy(solution, self.cvrp)
            new_solution = repair(destroyed_solution, self.cvrp)

            new_cost = self.cvrp.calculate_total_cost(new_solution)
            if self.acceptance_criteria(new_cost, best_cost, temperature):
                solution = new_solution
                self.update_weights(destroy, new_cost < best_cost, self.destroy_weights)
                self.update_weights(repair, new_cost < best_cost, self.repair_weights)

                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
                    self.best_costs.append(best_cost)
                    self.no_improve = 0  # reset no_improve counter
                else:
                    self.no_improve += 1
            else:
                self.update_weights(destroy, False, self.destroy_weights)
                self.update_weights(repair, False, self.repair_weights)
                self.no_improve += 1  # increase no_improve counter
            self.best_costs.append(best_cost)

        return best_solution, best_cost


def acceptance_criteria(new_cost, best_cost, temperature):
    delta = new_cost - best_cost
    probability = np.exp(-delta / temperature)
    return np.random.rand() < probability


def worst_removal(solution, cvrp, remove_percentage=0.2):
    """Remove customers that contribute most to the total cost."""
    solution_copy = solution.copy()
    num_remove = int(remove_percentage * len(solution))
    for _ in range(num_remove):
        worst_route = None
        worst_node_index = None
        worst_cost = float('-inf')
        for route in solution_copy:
            # ensure the depot and at least one node remains
            if len(route) > 4:
                for i in range(1, len(route) - 1):
                    route_copy = route.copy()
                    node = route_copy.pop(i)
                    cost_reduction = cvrp.calculate_total_cost([route]) - cvrp.calculate_total_cost([route_copy])
                    if cost_reduction > worst_cost:
                        worst_route = route
                        worst_node_index = i
                        worst_cost = cost_reduction
        # Check if worst_route and worst_node_index is not None
        if worst_route is not None and worst_node_index is not None:
            worst_route.pop(worst_node_index)
    return solution_copy


def greedy_insertion(solution, cvrp):
    """Insert customers back into the solution in a greedy way."""
    removed_customers = [i for i in range(1, cvrp.location) if i not in [customer for route in solution for customer in route]]
    route_capacities = {i: sum(cvrp.demands[customer] for customer in route) for i, route in enumerate(solution)}

    for customer in removed_customers:
        best_route = None
        best_route_index = None
        best_insert_pos = None
        best_insert_cost = float('inf')
        for route_index, route in enumerate(solution):
            # ensure capacity is not exceeded
            if route_capacities[route_index] + cvrp.demands[customer] <= cvrp.capacity:
                for i in range(1, len(route)):
                    route_copy = route.copy()
                    route_copy.insert(i, customer)
                    insert_cost = cvrp.calculate_total_cost([route_copy]) - cvrp.calculate_total_cost([route])
                    if insert_cost < best_insert_cost:
                        best_route = route
                        best_route_index = route_index
                        best_insert_pos = i
                        best_insert_cost = insert_cost
        # If a customer cannot be inserted into any route, create a new route
        if best_route is None and best_insert_pos is None:
            solution.append([0, customer, 0])
            route_capacities[len(solution)-1] = cvrp.demands[customer]
        else:
            best_route.insert(best_insert_pos, customer)
            route_capacities[best_route_index] += cvrp.demands[customer]

    return solution


def acceptance_criteria(new_cost, best_cost, temperature):
    if new_cost < best_cost:
        return True
    else:
        delta = new_cost - best_cost
        probability = np.exp(-delta / temperature)
        return np.random.rand() < probability

# import numpy as np
#
#
# class ALNS:
#     def __init__(self, problem, destroy_methods, repair_methods, weights, iterations, initial_temperature, cooling_rate):
#         self.problem = problem
#         self.destroy_methods = destroy_methods
#         self.repair_methods = repair_methods
#         self.destroy_weights = dict(zip(destroy_methods, weights))
#         self.repair_weights = dict(zip(repair_methods, weights))
#         self.iterations = iterations
#         self.initial_temperature = initial_temperature
#         self.cooling_rate = cooling_rate
#         self.no_improve = 0
#
#     def select_method(self, methods, weights):
#         return np.random.choice(methods, p=weights / np.sum(weights))
#
#     def update_weights(self, method, success, weights, rate=0.2):
#         weights[method] += rate if success else -rate
#
#     def solve(self):
#         # Generate an initial solution
#         solution = np.random.uniform(self.problem.bounds[0][0], self.problem.bounds[0][1], size=self.problem.dimensions)
#         best_solution = solution
#         best_cost = self.problem.evaluate(solution)
#         temperature = self.initial_temperature
#
#         for iteration in range(self.iterations):
#             if self.no_improve > 10:
#                 temperature *= 2 * self.cooling_rate
#             else:
#                 temperature *= self.cooling_rate
#
#             destroy = self.select_method(self.destroy_methods, list(self.destroy_weights.values()))
#             repair = self.select_method(self.repair_methods, list(self.repair_weights.values()))
#
#             destroyed_solution = destroy(solution, self.problem)
#             new_solution = repair(destroyed_solution, self.problem)
#
#             new_cost = self.problem.evaluate(new_solution)
#             if self.acceptance_criteria(new_cost, best_cost, temperature):
#                 solution = new_solution
#                 self.update_weights(destroy, new_cost < best_cost, self.destroy_weights)
#                 self.update_weights(repair, new_cost < best_cost, self.repair_weights)
#
#                 if new_cost < best_cost:
#                     best_solution = new_solution
#                     best_cost = new_cost
#                     self.no_improve = 0
#                 else:
#                     self.no_improve += 1
#             else:
#                 self.update_weights(destroy, False, self.destroy_weights)
#                 self.update_weights(repair, False, self.repair_weights)
#                 self.no_improve += 1
#
#         return best_solution, best_cost
#
#     def acceptance_criteria(self, new_cost, best_cost, temperature):
#         delta = new_cost - best_cost
#         probability = np.exp(-delta / temperature)
#         return np.random.rand() < probability
#
#
# def random_perturbation(solution, problem):
#     """Add some random noise to the solution vector."""
#     noise = np.random.normal(0, 0.1, size=problem.dimensions)  # 0.1 is the standard deviation of the noise
#     return solution + noise
#
#
# def bounding(solution, problem):
#     """Ensure that the solution lies within the problem's search space."""
#     lower_bounds, upper_bounds = problem.bounds
#     return np.clip(solution, lower_bounds, upper_bounds)
#
#
# def worst_removal(solution, cvrp, remove_percentage=0.2):
#     """Remove customers that contribute most to the total cost."""
#     solution_copy = solution.copy()
#     num_remove = int(remove_percentage * len(solution))
#     for _ in range(num_remove):
#         worst_route = None
#         worst_node_index = None
#         worst_cost = float('-inf')
#         for route in solution_copy:
#             # ensure the depot and at least one node remains
#             if len(route) > 4:
#                 for i in range(1, len(route) - 1):
#                     route_copy = route.copy()
#                     node = route_copy.pop(i)
#                     cost_reduction = cvrp.calculate_total_cost([route]) - cvrp.calculate_total_cost([route_copy])
#                     if cost_reduction > worst_cost:
#                         worst_route = route
#                         worst_node_index = i
#                         worst_cost = cost_reduction
#         # Check if worst_route and worst_node_index is not None
#         if worst_route is not None and worst_node_index is not None:
#             worst_route.pop(worst_node_index)
#     return solution_copy
#
#
# def greedy_insertion(solution, cvrp):
#     """Insert customers back into the solution in a greedy way."""
#     removed_customers = [i for i in range(1, cvrp.location) if i not in [customer for route in solution for customer in route]]
#     route_capacities = {i: sum(cvrp.demands[customer] for customer in route) for i, route in enumerate(solution)}
#
#     for customer in removed_customers:
#         best_route = None
#         best_route_index = None
#         best_insert_pos = None
#         best_insert_cost = float('inf')
#         for route_index, route in enumerate(solution):
#             # ensure capacity is not exceeded
#             if route_capacities[route_index] + cvrp.demands[customer] <= cvrp.capacity:
#                 for i in range(1, len(route)):
#                     route_copy = route.copy()
#                     route_copy.insert(i, customer)
#                     insert_cost = cvrp.calculate_total_cost([route_copy]) - cvrp.calculate_total_cost([route])
#                     if insert_cost < best_insert_cost:
#                         best_route = route
#                         best_route_index = route_index
#                         best_insert_pos = i
#                         best_insert_cost = insert_cost
#         # If a customer cannot be inserted into any route, create a new route
#         if best_route is None and best_insert_pos is None:
#             solution.append([0, customer, 0])
#             route_capacities[len(solution)-1] = cvrp.demands[customer]
#         else:
#             best_route.insert(best_insert_pos, customer)
#             route_capacities[best_route_index] += cvrp.demands[customer]
#
#     return solution
#
#
# def acceptance_criteria(new_cost, best_cost, temperature):
#     if new_cost < best_cost:
#         return True
#     else:
#         delta = new_cost - best_cost
#         probability = np.exp(-delta / temperature)
#         return np.random.rand() < probability