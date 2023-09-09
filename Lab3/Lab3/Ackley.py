# import math
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# class AckleyFunction:
#     def __init__(self, dimensions):
#         self.dimensions = dimensions
#         self.bounds = [(-32.768, 32.768)] * dimensions
#
#     def evaluate(self, solution):
#         a = 20
#         b = 0.2
#         c = 2 * math.pi
#
#         sum_sq_term = -b * math.sqrt((1 / self.dimensions) * sum(x**2 for x in solution))
#         cos_term = (1 / self.dimensions) * sum(math.cos(c * x) for x in solution)
#
#         result = -a * math.exp(sum_sq_term) - math.exp(cos_term) + a + math.exp(1)
#         return result
#
#     def plotingAckley(self):
#         x = np.arange(-32.768, 32.768, 0.001)
#         y = np.arange(-32.768, 32.768, 0.001)
#         X, Y = np.meshgrid(x, y)
#         Z = self.evaluate([X, Y])
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#         ax.set_xlabel('X Label')
#         ax.set_ylabel('Y Label')
#         ax.set_zlabel('Z Label')
#
#         plt.show()
#
#
# class AckleyCVRP:
#     def __init__(self, ackley, grid_size):
#         self.ackley = ackley
#         self.grid_size = grid_size
#         self.bounds = ackley.bounds
#         self.distance_matrix = self.create_distance_matrix()
#
#     def create_distance_matrix(self):
#         points = np.linspace(self.bounds[0][0], self.bounds[0][1], self.grid_size)
#         distance_matrix = np.zeros((self.grid_size, self.grid_size))
#
#         for i in range(self.grid_size):
#             for j in range(self.grid_size):
#                 point_i = points[i]
#                 point_j = points[j]
#                 distance = abs(self.ackley.evaluate([point_i]) - self.ackley.evaluate([point_j]))
#                 distance_matrix[i, j] = distance
#         return distance_matrix
#
#
# class CVRP_AckleyWrapper:
#     def __init__(self, n_dimensions):
#         self.location = n_dimensions
#
#     @staticmethod
#     def initial_solution(cvrp):
#         # Initialize with a random solution
#         return [[0] + [np.random.uniform(-32.768, 32.768) for _ in range(cvrp.n_dimensions)] + [0]]
#
#     @staticmethod
#     def calculate_total_cost(solution):
#         # The "cost" here is the Ackley function evaluated at the current solution
#         return -20*np.exp(-0.2*np.sqrt(np.sum(np.square(solution[0][1:-1]))/len(solution[0][1:-1]))) - np.exp(np.sum(np.cos(2*np.pi*np.array(solution[0][1:-1])))/len(solution[0][1:-1])) + 20 + np.e
# import numpy as np
#
#
# class SimulatedAnnealing:
#     def __init__(self, T=100, alpha=0.99, max_iter=1000, restarts=5, tabu_tenure=10):
#         self.T = T
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.restarts = restarts
#         self.tabu_tenure = tabu_tenure
#         self.tabu_list = []
#
#     def ackley(self, x, a=20, b=0.2, c=2*np.pi):
#         d = len(x)
#         sum1 = -a * np.exp(-b * np.sqrt(sum(x**2) / d))
#         sum2 = -np.exp(sum(np.cos(c*x) / d))
#         return sum1 + sum2 + a + np.exp(1)
#
#     def solve(self, init_solution):
#         best_solution = init_solution
#         best_cost = self.ackley(best_solution)
#
#         for _ in range(self.restarts):
#             solution = init_solution
#             for _ in range(self.max_iter):
#                 neighbor = self.generate_neighbor(solution.copy())
#                 neighbor_cost = self.ackley(neighbor)
#                 cost_diff = neighbor_cost - best_cost
#
#                 # Decide whether to move to neighbor
#                 if (cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / self.T)) and neighbor.tolist() not in self.tabu_list:
#                     solution = neighbor
#                     if neighbor_cost < best_cost:
#                         best_solution = neighbor
#                         best_cost = neighbor_cost
#
#                 # Add to tabu list and remove oldest entries if it exceeds tenure
#                 self.tabu_list.append(neighbor.tolist())
#                 if len(self.tabu_list) > self.tabu_tenure:
#                     self.tabu_list.pop(0)
#
#                 # Decrease temperature
#                 self.T *= self.alpha
#
#         return best_solution, best_cost
#
#     def generate_neighbor(self, solution):
#         # Make a small random change to the solution
#         solution += np.random.uniform(-1, 1, len(solution))
#         # Clip the new solution to the bounds
#         solution = np.clip(solution, -32.768, 32.768)
#         return solution
#
#
# # initial guess
# x0 = np.random.uniform(-32.768, 32.768, 10)
#
# sa = SimulatedAnnealing()
# best_solution, best_cost = sa.solve(x0)
#
# print("Best solution found: ", best_solution)
# print("Ackley function value at this solution: ", best_cost)


# import random
# import numpy as np
# import math
#
# # define constants for the Ackley function
# a = 20
# b = 0.2
# c = 2 * math.pi
# d = 10
#
#
# # define the Ackley function
# def ackley(x):
#     term1 = -a * np.exp(-b * np.sqrt(1 / d * sum([i ** 2 for i in x])))
#     term2 = -np.exp(1 / d * sum([np.cos(c * i) for i in x]))
#     return term1 + term2 + a + np.exp(1)
#
#
# class TabuSearch:
#     def __init__(self, dimensions, lower_bound, upper_bound, tenure):
#         self.dimensions = dimensions
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self.tenure = tenure
#         self.tabu_list = []
#         self.best_solution = []
#         self.best_cost = float('inf')
#
#     def create_random_solution(self):
#         return [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dimensions)]
#
#     def get_neighborhood(self, solution):
#         neighborhood = []
#         for i in range(self.dimensions):
#             for j in [0.05, -0.05]:
#                 neighbor = solution.copy()
#                 neighbor[i] += j
#                 neighbor[i] = max(min(neighbor[i], self.upper_bound), self.lower_bound)
#                 neighborhood.append(neighbor)
#         return neighborhood
#
#     def tabu_search(self, max_iterations):
#         current_solution = self.create_random_solution()
#         current_cost = ackley(current_solution)
#
#         for _ in range(max_iterations):
#             neighborhood = self.get_neighborhood(current_solution)
#             neighborhood_costs = [ackley(x) for x in neighborhood]
#
#             best_neighbor_index = np.argmin(neighborhood_costs)
#
#             if len(self.tabu_list) >= self.tenure:
#                 self.tabu_list.pop(0)
#
#             if neighborhood[best_neighbor_index] not in self.tabu_list and neighborhood_costs[
#                 best_neighbor_index] < current_cost:
#                 self.tabu_list.append(neighborhood[best_neighbor_index])
#                 current_solution = neighborhood[best_neighbor_index]
#                 current_cost = neighborhood_costs[best_neighbor_index]
#
#             if current_cost < self.best_cost:
#                 self.best_cost = current_cost
#                 self.best_solution = current_solution
#
#         return self.best_solution, self.best_cost
#
#
# ts = TabuSearch(10, -32.768, 32.768, 5)
# best_solution, best_cost = ts.tabu_search(1000)
# print(f"Best Solution: {best_solution}")
# print(f"Best Cost: {best_cost}")

import random
import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithmForAckley:
    def __init__(self, dimension, population_size=300, generations=100, mutation_rate=0.1, elitism_ratio=0.1):
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio

    def generate_population(self):
        return np.random.uniform(-32.768, 32.768, (self.population_size, self.dimension))

    def ackley_fitness(self, x):
        firstSum = np.sum(np.square(x))
        secondSum = np.sum(np.cos(2.0*np.pi*x))
        return -20.0*np.exp(-0.2*np.sqrt(firstSum/x.size)) - np.exp(secondSum/x.size) + 20 + np.e

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.dimension - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def select_parents(self, population, fitnesses):
        # Implementing tournament selection
        tournament_size = 10  # You can adjust this
        indices = np.random.choice(range(self.population_size), size=tournament_size)
        tournament_fitnesses = fitnesses[indices]
        best_index = indices[np.argmax(tournament_fitnesses)]
        return population[best_index]

    def mutate(self, individual, generation):
        # Adaptive mutation rate
        mutation_rate = self.mutation_rate * (1 - generation/self.generations)
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, self.dimension - 1)
            individual[mutation_point] += np.random.uniform(-1, 1)
        return individual

    def run(self):
        population = self.generate_population()
        best_individual = None
        best_fitness = float('-inf')
        for generation in range(self.generations):
            fitnesses = np.array([self.ackley_fitness(individual) for individual in population])
            # Elitism
            elites = np.argsort(fitnesses)[-int(self.elitism_ratio*self.population_size):]
            new_population = [population[i] for i in elites]
            while len(new_population) < self.population_size:
                parent1 = self.select_parents(population, fitnesses)
                parent2 = self.select_parents(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1, generation), self.mutate(child2, generation)])
            population = np.array(new_population)
            max_fitness_index = np.argmax(fitnesses)
            if fitnesses[max_fitness_index] > best_fitness:
                best_fitness = fitnesses[max_fitness_index]
                best_individual = population[max_fitness_index]
        return best_individual, best_fitness


# Assume you have set the problem dimension as 10 and number of generations as 1000
dim = 10
generations = 1000

# Create an instance of GeneticAlgorithmForAckley with the problem dimensions and number of generations
ga = GeneticAlgorithmForAckley(dim, generations=generations)

# Call the run() method to execute the Genetic Algorithm
best_solution, best_cost = ga.run()

# Print the best solution and its cost
print("Best solution found by the GA:")
print(best_solution)

print("\nCost of the best solution:")
print(best_cost)
