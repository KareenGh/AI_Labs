# # import random
# #
# # import matplotlib.pyplot as plt
# # import time
# #
# # from Genetic_Algorithms import generate_population
# # from crossover import cx, pmx
# # from genetic_diversity import average_genetic_distance, unique_alleles
# # from selection import roulette_wheel_selection, stochastic_universal_sampling, ranking_and_tournament_selection
# # from utils import inversion_mutation
# #
# #
# # # Define the genetic algorithm function for the N-Queens problem
# # def genetic_algorithm_NQueens(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
# #                               K=None, exchange_operator="CX", mutation_rate=0.01, elitism_rate=0.1, aging_rate=0.1):
# #
# #     # Initialize the population with random individuals
# #     population = generate_population(pop_size, num_genes)
# #
# #     # Evolve the population for a fixed number of generations
# #     start_time = time.time()
# #     start_clock = time.process_time()
# #     solution_found = False
# #     generation_times = []
# #     best_fitnesses = []
# #     for generation in range(max_generations):
# #         # Calculate the fitness of each individual
# #         fitnesses = [fitness_func(individual) for individual in population]
# #
# #         # Check if a solution is found
# #         if min(fitnesses) == 0:
# #             solution_found = True
# #             break
# #
# #         elite_size = int(pop_size * elitism_rate)
# #         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
# #         elites = [population[i] for i in elite_indices]
# #
# #         # Generate new individuals by applying crossover and mutation operators
# #         offspring = []
# #         while len(offspring) < pop_size - elite_size:
# #             if selection_method == "RWS":
# #                 parent1 = roulette_wheel_selection(population, fitnesses, fitnesses)
# #                 parent2 = roulette_wheel_selection(population, fitnesses, fitnesses)
# #             elif selection_method == "SUS":
# #                 parent1, parent2 = stochastic_universal_sampling(population, fitnesses, fitnesses, 2)
# #             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
# #                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
# #
# #             if exchange_operator == "CX":
# #                 child1, child2 = cx(parent1, parent2)
# #             elif exchange_operator == "PMX":
# #                 child1, child2 = pmx(parent1, parent2)
# #
# #             for child in [child1, child2]:
# #                 if random.random() < mutation_rate:
# #                     child = inversion_mutation(child)
# #             offspring.append(child1)
# #             offspring.append(child2)
# #
# #         population = elites + offspring
# #
# #         # Calculate genetic diversification metrics
# #         avg_gen_dist = average_genetic_distance(population)
# #         num_unique_alleles = unique_alleles(population)
# #
# #         # Store the statistics and running time for the current generation
# #         generation_times.append(time.process_time() - start_clock)
# #         best_fitnesses.append(min(fitnesses))
# #
# #     # Find the individual with the highest fitness
# #     best_individual = min(population, key=fitness_func)
# #     best_fitness = fitness_func(best_individual)
# #
# #     if solution_found:
# #         print("Solution found.")
# #     else:
# #         print("Solution not found within the given generations.")
# #
# #     # Plot the performance metrics
# #     plt.plot(range(max_generations), best_fitnesses, label="Best Fitness")
# #     plt.plot(range(max_generations), generation_times, label="Generation Time")
# #     plt.legend()
# #     plt.xlabel("Generation")
# #     plt.ylabel("Performance")
# #     plt.title("Performance of Genetic Algorithm for N-Queens Problem")
# #     plt.show()
# #
# #     return best_individual, best_fitness
# #
# #
# # # Define the fitness function for the N-Queens problem
# # def n_queens_fitness(individual):
# #     size = len(individual)
# #     conflicts = 0
# #
# #     for i in range(size):
# #         for j in range(i+1, size):
# #             if individual[i] == individual[j]:
# #                 conflicts += 1
# #             elif abs(individual[i] - individual[j]) == abs(i - j):
# #                 conflicts += 1
# #
# #     return conflicts
# #
# #
# # pop_sizes = [50, 100, 200]
# # mutation_rates = [0.01, 0.02, 0.05]
# # selection_methods = ["RWS", "SUS", "RANKING_TOURNAMENT"]
# # crossover_types = ["CX", "PMX"]
# # elitism_rates = [0.1, 0.2, 0.3]
# # aging_rates = [0.1, 0.2, 0.3]
# #
#
# import time
# import random
# import math
# import matplotlib.pyplot as plt
#
# # Parameters for the genetic algorithm
# GA_POPSIZE = 100
# GA_MAXGENS = 100
# GA_MUTATIONRATE = 0.01
# GA_ELITRATE = 0.1
# GA_AGINGRATE = 0.1
# GA_SELECTIONMETHOD = "RWS"
# GA_CROSSOVERTYPE = "CX"
# GA_K = None
# GA_EXCHANGEOPERATOR = "CX"
#
# # Functions for the N-Queens problem
# def generate_population(pop_size, num_genes):
#     population = []
#     for _ in range(pop_size):
#         individual = list(range(num_genes))
#         random.shuffle(individual)
#         population.append(individual)
#     return population
#
# def roulette_wheel_selection(population, fitnesses, probabilities):
#     return population[random.choices(range(len(population)), weights=probabilities)[0]]
#
# def stochastic_universal_sampling(population, fitnesses, probabilities, num_parents):
#     step = sum(probabilities) / num_parents
#     start = random.uniform(0, step)
#     return [roulette_wheel_selection(population, fitnesses, [p - i * step for p in probabilities])
#             for i in range(num_parents)]
#
# def cx(parent1, parent2):
#     child1, child2 = parent1.copy(), parent2.copy()
#     cx_point1 = random.randint(0, len(parent1) - 1)
#     cx_point2 = random.randint(0, len(parent1) - 1)
#     if cx_point2 < cx_point1:
#         cx_point1, cx_point2 = cx_point2, cx_point1
#     for i in range(cx_point1, cx_point2 + 1):
#         temp = child1[i]
#         child1[i] = child2[i]
#         child2[i] = temp
#     return child1, child2
#
# def inversion_mutation(individual):
#     mutation_point1 = random.randint(0, len(individual) - 1)
#     mutation_point2 = random.randint(0, len(individual) - 1)
#     if mutation_point2 < mutation_point1:
#         mutation_point1, mutation_point2 = mutation_point2, mutation_point1
#     individual[mutation_point1:mutation_point2+1] = reversed(individual[mutation_point1:mutation_point2+1])
#     return individual
#
# def average_genetic_distance(population):
#     total_dist = 0
#     num_pairs = 0
#     for i in range(len(population)):
#         for j in range(i+1, len(population)):
#             dist = sum(1 for a, b in zip(population[i], population[j]) if a != b)
#             total_dist += dist
#             num_pairs += 1
#     return total_dist / num_pairs
#
# def unique_alleles(population):
#     alleles = set()
#     for individual in population:
#         alleles.update(individual)
#     return len(alleles)
#
# def print_generation_stats_NQueens(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
#                                     num_unique_alleles):
#     best_fitness = min(fitness_func(individual) for individual in population)
#     mean_fitness = sum(fitness_func(individual) for individual in population) / len(population)
#     elapsed_time = time.time() - start_time
#     cpu_time = time.process_time() - start_clock
#     print(f"Generation: {generation}")
#     print(f"Best Fitness: {best_fitness}")
#     print(f"Mean Fitness: {mean_fitness}")
#     print(f"Elapsed Time: {elapsed_time:.2f} seconds")
#     print(f"CPU Time: {cpu_time:.2f} seconds")
#     print(f"Average Genetic Distance: {avg_gen_dist:.2f}")
#     print(f"Number of Unique Alleles: {num_unique_alleles}")
#     print()
#
#
# def ranking_and_tournament_selection(population, fitnesses, K):
#     ranking_probs = [K / (i + 1) for i in range(len(population))]
#     probs_sum = sum(ranking_probs)
#     ranking_probs = [p / probs_sum for p in ranking_probs]
#
#     parent1_index = random.choices(range(len(population)), weights=ranking_probs)[0]
#
#     temp = population[:parent1_index] + population[parent1_index + 1:]
#     temp_fitnesses = fitnesses[:parent1_index] + fitnesses[parent1_index + 1:]
#
#     ranking_probs = [K / (i + 1) for i in range(len(temp))]
#     probs_sum = sum(ranking_probs)
#     ranking_probs = [p / probs_sum for p in ranking_probs]
#
#     parent2_index = random.choices(range(len(temp)), weights=ranking_probs)[0]
#     parent2 = temp[parent2_index]
#
#     return population[parent1_index], parent2
#
#
# # Functions for the Bin Packing problem
# def generate_bin_packing_individual(num_items, max_item_size):
#     individual = []
#     total_size = 0
#     while total_size < num_items * max_item_size:
#         bin_size = random.randint(1, max_item_size)
#         total_size += bin_size
#         individual.append(bin_size)
#     return individual
#
#
# def print_generation_stats_BinPacking(generation, population, fitness_func, start_time, start_clock, item_sizes,
#                                       bin_capacity, avg_gen_dist, num_unique_alleles):
#     best_fitness = min(fitness_func(individual, item_sizes, bin_capacity) for individual in population)
#     mean_fitness = sum(fitness_func(individual, item_sizes, bin_capacity) for individual in population) / len(
#         population)
#     elapsed_time = time.time() - start_time
#     cpu_time = time.process_time() - start_clock
#     print(f"Generation: {generation}")
#     print(f"Best Fitness: {best_fitness}")
#     print(f"Mean Fitness: {mean_fitness}")
#     print(f"Elapsed Time: {elapsed_time:.2f} seconds")
#     print(f"CPU Time: {cpu_time:.2f} seconds")
#     print(f"Average Genetic Distance: {avg_gen_dist:.2f}")
#     print(f"Number of Unique Alleles: {num_unique_alleles}")
#     print()
#
#
# def ranking_and_tournament_selection(population, fitnesses, K):
#     ranking_prob = [i / (len(population) - 1) for i in range(len(population))]
#     ranking_prob_sum = [sum(ranking_prob[:i+1]) for i in range(len(population))]
#     parents = []
#     for _ in range(2):
#         competitors = random.choices(population, k=K)
#         ranks = sorted(range(K), key=lambda i: fitnesses[competitors[i]])
#         selected_index = competitors[ranks[0]]
#         for rank in ranks[1:]:
#             if random.random() < ranking_prob_sum[rank]:
#                 selected_index = competitors[rank]
#                 break
#         parents.append(selected_index)
#     return parents
#
# def cx_first_point(parent1, parent2):
#     child1, child2 = parent1.copy(), parent2.copy()
#     cx_point1 = random.randint(0, len(parent1) - 1)
#     temp = child1[cx_point1]
#     child1[cx_point1] = child2[cx_point1]
#     child2[cx_point1] = temp
#     return child1, child2
#
#
# def cx_two_point(parent1, parent2):
#     child1, child2 = parent1.copy(), parent2.copy()
#     cx_point1 = random.randint(0, len(parent1) - 1)
#     cx_point2 = random.randint(0, len(parent1) - 1)
#     if cx_point2 < cx_point1:
#         cx_point1, cx_point2 = cx_point2, cx_point1
#     for i in range(cx_point1, cx_point2 +
#
from Genetic_Algorithms import genetic_algorithm_NQueens, genetic_algorithm_BinPacking
from LoadFiles import read_binpack_input_file
from fitness import bin_packing_fitness, n_queens_fitness


def sensitivity_analysis_NQueens(pop_sizes, mutation_rates, selection_methods, survival_strategies, exchange_operators):
    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            for selection_method in selection_methods:
                for survival_strategy in survival_strategies:
                    for exchange_operator in exchange_operators:
                        print(f"Running NQueens with pop_size={pop_size}, mutation_rate={mutation_rate}, "
                              f"selection_method={selection_method}, survival_strategy={survival_strategy}, "
                              f"exchange_operator={exchange_operator}...")

                        # Run the genetic algorithm with the given parameters
                        genetic_algorithm_NQueens(pop_size=pop_size, num_genes=8, fitness_func=n_queens_fitness,
                                                 max_generations=100, crossover_type="PMX",
                                                 selection_method=selection_method, K=None,
                                                 exchange_operator=exchange_operator)


def sensitivity_analysis_BinPacking(pop_sizes, mutation_rates, selection_methods, survival_strategies, crossover_types):
    item_sizes, bin_capacity = read_binpack_input_file("binpack1.txt")

    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            for selection_method in selection_methods:
                for survival_strategy in survival_strategies:
                    for crossover_type in crossover_types:
                        print(f"Running BinPacking with pop_size={pop_size}, mutation_rate={mutation_rate}, "
                              f"selection_method={selection_method}, survival_strategy={survival_strategy}, "
                              f"crossover_type={crossover_type}...")

                        # Run the genetic algorithm with the given parameters
                        genetic_algorithm_BinPacking(pop_size=pop_size, num_genes=len(item_sizes),
                                                     fitness_func=bin_packing_fitness, max_generations=100,
                                                     crossover_type=crossover_type, selection_method=selection_method,
                                                     K=None, item_sizes=item_sizes, bin_capacity=bin_capacity)


# def sensitivity_analysis_NQueens(pop_sizes, mutation_rates, selection_methods, survival_strategies, exchange_operators):
#     for pop_size in pop_sizes:
#         print(f"Running NQueens with pop_size={pop_size}...")
#         genetic_algorithm_NQueens(pop_size=pop_size, num_genes=8, fitness_func=n_queens_fitness, max_generations=100,
#                                   crossover_type="PMX", selection_method=selection_methods[0], K=None,
#                                   exchange_operator=exchange_operators[0])
