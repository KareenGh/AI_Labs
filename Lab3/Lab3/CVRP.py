import math
import random
import time

import numpy as np
from matplotlib import pyplot as plt


class CVRP:
    def __init__(self, dimension, capacity, node_coords, demands):
        self.location = dimension
        self.capacity = capacity
        self.node_coords = node_coords
        self.demands = demands
        self.distance_matrix = self.calculate_distance_matrix()

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.location, self.location))
        for i in range(self.location):
            for j in range(i+1, self.location):
                distance_matrix[i][j] = self.euclidean_distance(self.node_coords[i], self.node_coords[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_total_cost(self, routes):
        return sum(self.calculate_route_cost(route) for route in routes)

    def calculate_route_cost(self, route):
        return sum(self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + \
               self.distance_matrix[route[-1]][0] + self.distance_matrix[0][route[0]]

    def calc_path_cost(self, path):
        total_cost = 0
        capacity = self.capacity
        index = 0
        total_cost += self.distance_matrix[0][path[index]]
        capacity -= self.demands[path[index]]

        vehicles = 1
        while index < (len(path) - 1):
            city1 = path[index]
            city2 = path[index + 1]
            if self.demands[city2] <= capacity:
                cost = self.distance_matrix[city1][city2]
                capacity -= self.demands[city2]
                total_cost += cost
            else:
                total_cost += self.distance_matrix[city1][0]
                capacity = self.capacity
                vehicles += 1
                total_cost += self.distance_matrix[0][city2]
                capacity -= self.demands[city2]
            index += 1

        total_cost += self.distance_matrix[path[index]][0]
        return total_cost, vehicles

    def calc_path_cost_sa(self, path):
        total_cost = 0
        capacity = self.capacity
        index = 0
        route = [0]
        routes = []

        total_cost += self.distance_matrix[0][path[index]]
        capacity -= self.demands[path[index]]

        while index < (len(path) - 1):
            city1 = path[index]
            city2 = path[index + 1]
            if self.demands[city2] <= capacity:
                cost = self.distance_matrix[city1][city2]
                capacity -= self.demands[city2]
                total_cost += cost
                route.append(city2)
            else:
                route.append(0)
                routes.append(route)
                total_cost += self.distance_matrix[city1][0]
                capacity = self.capacity
                total_cost += self.distance_matrix[0][city2]
                capacity -= self.demands[city2]
                route = [0, city2]
            index += 1

        route.append(0)
        routes.append(route)
        total_cost += self.distance_matrix[path[index]][0]
        return routes, total_cost

    def display_results(self, routes, start_time, problem_name):
        # Check if routes is a list of lists or just a list
        if type(routes[0]) is not list:
            routes = [routes]  # Convert to a list of lists

        total_cost = self.calculate_total_cost(routes)
        print(f"Total cost: {total_cost}")
        for i, route in enumerate(routes):
            print(f"Vehicle {i + 1}: {route}")
        elapsed_time = time.time() - start_time
        cpu_time = time.process_time()
        print(f"Clock Ticks: {cpu_time}s")
        print(f"Time Elapsed: {elapsed_time}s")
        self.plot_routes(routes, problem_name)

    def plot_routes(self, routes, problem_name):
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the depot
        depot_x, depot_y = self.node_coords[0]
        ax.scatter(depot_x, depot_y, c='green', s=200)
        ax.text(depot_x, depot_y, "Depot", fontsize=12)

        # Plot the customers and their demands
        for i in range(1, self.location):
            x, y = self.node_coords[i]
            demand = self.demands[i]
            ax.scatter(x, y, c='orange', s=200)
            ax.text(x, y, f"Customer {i}\nDemand: {demand}", fontsize=12)

            # Define a list of colors to cycle through
            colors = ['red', 'blue', 'green', 'purple', 'cyan', 'magenta', 'yellow']

            # Plot the routes with different colors
            for i, route in enumerate(routes):
                x_values = [self.node_coords[node][0] for node in route]
                y_values = [self.node_coords[node][1] for node in route]
                x_values.append(self.node_coords[0][0])  # Return to depot
                y_values.append(self.node_coords[0][1])
                color = colors[i % len(colors)]  # Cycle through the colors
                ax.plot(x_values, y_values, c=color, label=f"Route {i + 1}")

        # Set axis labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Vehicle Routes for " + problem_name)

        # Show the plot
        plt.show()

    def random_solution(self):
        nodes = list(range(1, self.location))  # create a list of the nodes, excluding the depot (0)
        random.shuffle(nodes)  # randomize the nodes

        routes = []
        route = []
        capacity = self.capacity

        for node in nodes:
            if self.demands[node] <= capacity:
                route.append(node)
                capacity -= self.demands[node]
            else:
                routes.append(route)
                route = [node]
                capacity = self.capacity - self.demands[node]
            if nodes.index(node) == len(nodes) - 1:  # if this is the last node
                routes.append(route)  # append the current route to the list of routes even if it's not full

        return routes

    def calculate_route_demand(self, route):
        return sum(self.demands[i] for i in route)

    def plot_hist(self, algo, problem_name):
        plt.plot(algo.best_costs)
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.title('Improvement of fitness in ' + problem_name)
        plt.show()

