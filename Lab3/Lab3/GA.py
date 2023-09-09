import math
import random

from multiprocessing import Pool

import numpy as np

PENALTY_RATIO = 10000


class GeneticAlgorithm:
    def __init__(self, cvrp, pop_size=100, elite_size=5, mutation_rate=0.01, generations=500, num_islands=4,
                 migration_interval=100, migration_size=1, tournament_size=3):
        self.cvrp = cvrp
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.tournament_size = tournament_size
        self.best_costs = []

    def fitness_function(self, individual):
        individual = self.split_into_routes(individual)  # repair routes before evaluation

        total_distance = 0
        num_trucks = 0
        for route in individual:
            route_demand = sum(self.cvrp.demands[i] for i in route)
            total_distance += self.cvrp.calculate_route_cost(route)
            num_trucks += 1
        # weights for the objectives
        weight_distance = 0.8
        weight_trucks = 0.2
        # calculate the weighted sum
        weighted_sum = weight_distance * total_distance + weight_trucks * num_trucks
        return 1 / weighted_sum  # We use 1/total_

    def split_into_routes(self, individual):
        routes = []
        capacity = self.cvrp.capacity

        for route in individual:  # Here route is a list of cities
            current_route = []
            current_load = 0

            for city in route:
                if self.cvrp.demands[city] + current_load <= capacity:
                    current_route.append(city)
                    current_load += self.cvrp.demands[city]
                else:
                    routes.append(current_route)
                    current_route = [city]
                    current_load = self.cvrp.demands[city]

            if current_route:  # append the last route if it is not empty
                routes.append(current_route)

        return routes

    def initial_population(self):
        initial_population = []
        for _ in range(self.num_islands):
            island_population = []
            for _ in range(self.pop_size):
                individual = self.nearest_neighbor_heuristic()
                island_population.append(individual)
            initial_population.append(island_population)
        return initial_population

    def nearest_neighbor_heuristic(self):
        individual = []
        unvisited_cities = list(range(1, len(self.cvrp.node_coords)))
        random.shuffle(unvisited_cities)
        current_city = unvisited_cities.pop()
        while unvisited_cities:
            nearest_city = min(unvisited_cities, key=lambda city: self.cvrp.distance_matrix[current_city][city])
            individual.append(nearest_city)
            unvisited_cities.remove(nearest_city)
            current_city = nearest_city
        return [individual]  # individual is a list of lists

    def clarke_wright_savings_heuristic(self):
        savings = [
            (i, j, self.cvrp.distance_matrix[0][i] + self.cvrp.distance_matrix[0][j] - self.cvrp.distance_matrix[i][j])
            for i in range(1, self.cvrp.location) for j in range(i + 1, self.cvrp.location)]
        savings.sort(key=lambda x: x[2], reverse=True)

        routes = [[0, city, 0] for city in range(1, self.cvrp.location)]
        city_to_route = {city: route for city, route in zip(range(1, self.cvrp.location), routes)}

        for i, j, _ in savings:
            route_i = city_to_route[i]
            route_j = city_to_route[j]

            if route_i == route_j:
                continue

            if self.is_valid_merge(route_i, route_j, i, j):
                merged_route = self.merge_routes(route_i, route_j, i, j)
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(merged_route)
                for city in merged_route[1:-1]:
                    city_to_route[city] = merged_route

        return routes

    def is_valid_merge(self, route_i, route_j, i, j):
        if route_i[1] == i and route_j[-2] == j:
            new_route = route_i[:-1] + route_j[1:]
        elif route_i[-2] == i and route_j[1] == j:
            new_route = route_i[:-1] + route_j[1:]
        elif route_i[1] == i and route_j[1] == j:
            new_route = route_i[:-1] + route_j[::-1][1:]
        elif route_i[-2] == i and route_j[-2] == j:
            new_route = route_i[:-1] + route_j[::-1][1:]
        else:
            return False

        total_demand = sum(self.cvrp.demands[city] for city in new_route[1:-1])
        return total_demand <= self.cvrp.capacity

    def merge_routes(self, route_i, route_j, i, j):
        if route_i[1] == i and route_j[-2] == j:
            return route_i[:-1] + route_j[1:]
        elif route_i[-2] == i and route_j[1] == j:
            return route_i[:-1] + route_j[1:]
        elif route_i[1] == i and route_j[1] == j:
            return route_i[:-1] + route_j[::-1][1:]
        else:  # route_i[-2] == i and route_j[-2] == j
            return route_i[:-1] + route_j[::-1][1:]

    def pmx_crossover(self, parent1, parent2):
        num_cities = len(self.cvrp.node_coords)
        child1, child2 = [], []
        for route1, route2 in zip(parent1, parent2):
            p1, p2 = [-1] * num_cities, [-1] * num_cities

            # Initialize the position of each index in the individuals
            for i in range(len(route1)):
                p1[route1[i]] = i
            for i in range(len(route2)):
                p2[route2[i]] = i
            # Choose crossover points
            cxpoint1 = random.randint(0, len(route1))
            cxpoint2 = random.randint(0, len(route2) - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            # Apply crossover between cx points
            for i in range(cxpoint1, cxpoint2):
                # Keep track of the selected values
                temp1 = route1[i]
                temp2 = route2[i]
                # Swap the matched value
                route1[i], route1[p1[temp2]] = temp2, temp1
                route2[i], route2[p2[temp1]] = temp1, temp2
                # Position bookkeeping
                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
            child1.append(route1)
            child2.append(route2)
        return child1, child2

    def inversion_mutation(self, individual):
        if len(individual) < 2:
            return individual
        start, end = sorted(random.sample(range(len(individual)), 2))
        individual[start:end + 1] = reversed(individual[start:end + 1])
        return individual

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def roulette_wheel_selection(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probs = [fitness / total_fitness for fitness in fitnesses]
        selected_individual = np.random.choice(population, p=selection_probs)
        return selected_individual

    def perform_crossover(self, population):
        next_gen = list(population[:self.elite_size])
        for i in range(self.elite_size, len(population) - 1, 2):  # Subtract 1 from the range end
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = self.ox_crossover(parent1, parent2)
            next_gen.extend([child1, child2])

        # Handle the last individual separately if population size is odd
        if len(population) % 2 == 1:
            next_gen.append(population[-1])

        return next_gen

    def perform_mutation(self, next_gen):
        for i in range(len(next_gen)):
            if random.random() < self.mutation_rate:
                for j in range(len(next_gen[i])):
                    next_gen[i][j] = self.inversion_mutation(next_gen[i][j])
        return next_gen

    def perform_migration(self, populations, generation):
        if generation % self.migration_interval == 0:
            for i in range(self.num_islands):
                j = i
                while j == i:
                    j = random.randint(0, self.num_islands - 1)

                populations[i].sort(key=self.fitness_function, reverse=True)
                populations[j].sort(key=self.fitness_function, reverse=True)

                populations[i][:self.migration_size], populations[j][:self.migration_size] = \
                    populations[j][:self.migration_size], populations[i][:self.migration_size]
        return populations

    def ox_crossover(self, parent1, parent2):
        if len(parent1) < 2 or len(parent2) < 2:
            # Skip crossover if either parent has less than 2 routes
            return parent1, parent2
        size = min(len(parent1), len(parent2))
        start, end = sorted(random.sample(range(size), 2))
        child1 = [-1] * size
        child2 = [-1] * size
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        for i in range(size):
            if parent2[i] not in child1:
                for j in range(size):
                    if child1[(end + j) % size] == -1:
                        child1[(end + j) % size] = parent2[i]
                        break
            if parent1[i] not in child2:
                for j in range(size):
                    if child2[(end + j) % size] == -1:
                        child2[(end + j) % size] = parent1[i]
                        break
        return child1, child2

    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)

        # Choose crossover points
        slice1 = np.random.randint(size)
        slice2 = np.random.randint(size)

        # Ensure slice1 is the smaller index
        if slice2 < slice1:
            slice1, slice2 = slice2, slice1

        child1 = [None] * size
        child2 = [None] * size

        # Copy segment from first parent
        child1[slice1:slice2] = parent1[slice1:slice2]
        child2[slice1:slice2] = parent2[slice1:slice2]

        # Fill remaining spots with genes from second parent
        for i in range(size):
            if child1[i] is None and parent2[i] not in child1:
                child1[i] = parent2[i]
            if child2[i] is None and parent1[i] not in child2:
                child2[i] = parent1[i]

        # If there are still None values, replace them with the remaining genes
        for i in range(size):
            if child1[i] is None:
                child1[i] = next(gene for gene in parent2 if gene not in child1)
            if child2[i] is None:
                child2[i] = next(gene for gene in parent1 if gene not in child2)

        return child1, child2

    def adaptive_crossover_rate(self, generation, max_generations):
        start_rate = 0.8  # start crossover rate
        end_rate = 0.2  # end crossover rate
        return start_rate - generation * ((start_rate - end_rate) / max_generations)

    def swap_mutation(self, individual):
        size = len(individual)
        if size < 2:
            return individual
        idx1, idx2 = random.sample(range(size), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def solve(self):
        populations = self.initial_population()

        for generation in range(self.generations):
            self.elite_size = max(1, int(self.pop_size * (1 - generation / self.generations)))
            # reduce mutation rate over time using exponential decay
            self.mutation_rate = 1 * math.exp(-0.05 * generation)
            for population in populations:
                fitnesses = [self.fitness_function(individual) for individual in population]
                self.best_costs.append(max(fitnesses))
                population = self.tournament_selection(population, fitnesses)
                population, fitnesses = zip(*sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True))
                next_gen = self.perform_crossover(population)
                population = self.perform_mutation(next_gen)
            populations = self.perform_migration(populations, generation)
        best_solution = max([individual for population in populations for individual in population],
                            key=self.fitness_function)
        best_routes = self.split_into_routes(best_solution)
        return best_routes, self.cvrp.calculate_total_cost(best_routes)




