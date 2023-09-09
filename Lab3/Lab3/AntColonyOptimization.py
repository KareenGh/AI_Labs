import numpy as np


class AntColonyOptimization:
    def __init__(self, cvrp, n_ants, alpha, beta, rho, q, max_iter):
        self.cvrp = cvrp
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iter = max_iter
        self.best_costs = []

        # Initialize pheromone levels
        self.tau = np.ones(self.cvrp.distance_matrix.shape)  # pheromone levels

        # Prevent division by zero
        distance_matrix_with_zeros = np.where(self.cvrp.distance_matrix == 0, 1e-10, self.cvrp.distance_matrix)
        self.eta = 1 / distance_matrix_with_zeros  # heuristic information

    def solve(self):
        best_cost = float('inf')
        best_solution = []

        for i in range(self.max_iter):
            routes, costs = self.construct_solutions()
            if min(costs) < best_cost:
                best_cost = min(costs)
                self.best_costs.append(best_cost)
                best_solution = routes[np.argmin(costs)]
            self.update_pheromone(routes, costs)
            # self.best_costs.append(best_cost)

        return best_solution, best_cost

    def construct_solutions(self):
        all_routes = []
        all_costs = []
        for ant in range(self.n_ants):
            routes = []
            costs = []
            visited = [0]  # start at the depot
            route = [0]
            capacity = self.cvrp.capacity
            while len(visited) < len(self.cvrp.demands):  # ensure all nodes are visited
                p = (self.tau[route[-1]] ** self.alpha) * (self.eta[route[-1]] ** self.beta)
                p = p / np.where(p.sum() == 0, 1e-10, p.sum())  # normalize

                # Convert demands to a numpy array for the comparison operation
                demands_np = np.array(self.cvrp.demands)
                p[demands_np > capacity] = 0  # cannot satisfy the demand

                # If p.sum() is zero or contains NaN/infinite values, replace them with a small value
                if p.sum() == 0 or np.isnan(p).any() or np.isinf(p).any():
                    p = np.where(np.isnan(p) | np.isinf(p), 1e-10, p)

                # exclude already visited nodes
                p[visited] = 0

                if p.sum() == 0:  # no feasible nodes, go back to the depot
                    route.append(0)
                    routes.append(route)
                    costs.append(self.cvrp.calculate_route_cost(route))
                    route = [0]
                    capacity = self.cvrp.capacity
                else:
                    # Normalizing the probabilities again to ensure they sum to 1
                    p = p / np.where(p.sum() == 0, 1e-10, p.sum())
                    next_node = np.random.choice(range(len(self.cvrp.demands)), 1, p=p)[0]
                    route.append(next_node)
                    visited.append(next_node)
                    capacity -= self.cvrp.demands[next_node]

            # return to the depot
            route.append(0)
            routes.append(route)
            costs.append(self.cvrp.calculate_route_cost(route))
            all_routes.append(routes)
            all_costs.append(sum(costs))

        return all_routes, all_costs

    # def construct_solutions(self):
    #     all_routes = []
    #     all_costs = []
    #
    #     for ant in range(self.n_ants):
    #         routes = []
    #         costs = []
    #         visited = [0]  # start at the depot
    #         route = [0]
    #         capacity = self.cvrp.capacity
    #
    #         while len(visited) < len(self.cvrp.demands):  # ensure all nodes are visited
    #             unvisited = list(set(range(len(self.cvrp.demands))) - set(visited))
    #             feasible_nodes = [node for node in unvisited if self.cvrp.demands[node] <= capacity]
    #
    #             if not feasible_nodes:  # if no feasible nodes, start a new route
    #                 route.append(0)  # return to depot
    #                 routes.append(route)
    #                 costs.append(self.cvrp.calculate_route_cost(route))
    #                 route = [0]  # start at the depot for the new route
    #                 capacity = self.cvrp.capacity
    #                 continue
    #
    #             # Choose the nearest feasible node as the next node to visit
    #             nearest_node = min(feasible_nodes, key=lambda node: self.cvrp.distance_matrix[route[-1]][node])
    #             route.append(nearest_node)
    #             visited.append(nearest_node)
    #             capacity -= self.cvrp.demands[nearest_node]
    #
    #         # return to the depot for the last route
    #         route.append(0)
    #         routes.append(route)
    #         costs.append(self.cvrp.calculate_route_cost(route))
    #         all_routes.append(routes)
    #         all_costs.append(sum(costs))
    #
    #     return all_routes, all_costs

    def update_pheromone(self, routes, costs):
        self.tau = (1 - self.rho) * self.tau
        for routes_list, cost in zip(routes, costs):
            for route in routes_list:
                for i in range(len(route) - 1):
                    self.tau[route[i]][route[i + 1]] += self.q / (cost + 1e-10)

