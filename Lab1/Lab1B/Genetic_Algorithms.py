import math
import random
import statistics
import time

from matplotlib import pyplot as plt

from crossover import crossover, cx, pmx
from fitness import fitness_with_age
from genetic_diversity import average_genetic_distance, unique_alleles
from plotting import print_generation_stats, plot_fitness_histogram, print_generation_stats_NQueens, \
    print_generation_stats_BinPacking
from selection import scale_fitness, roulette_wheel_selection, stochastic_universal_sampling, \
    ranking_and_tournament_selection
from utils import inversion_mutation

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = int(GA_MUTATIONRATE * 100)
GA_TARGET = "Hello, world!"


# Define the genetic algorithm with selection_method parameter
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method, K=None):
    # Initialize the population with random individuals
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    for generation in range(max_generations):
        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Scale fitness values based on the current generation
        scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # genetic_algorithm function snippet
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses, scaled_fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses, scaled_fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, scaled_fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < GA_MUTATION:
                        child[i] = chr(random.randint(32, 126))
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                               num_unique_alleles)

        # Plot the fitness histogram every 10 generations
        if (generation + 1) % 10 == 0:
            plot_fitness_histogram(generation, population, fitness_func)

        # Find the individual with the highest fitness
        best_individual = max(population, key=lambda individual: fitness_func(individual))
        best_fitness = fitness_func(best_individual)

        # Check if the target solution is found
        if ''.join(population[generation]) == GA_TARGET:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_fitness_histogram(generation, population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

    # # Find the individual with the highest fitness
    # best_individual = max(population, key=lambda individual: fitness_func(individual))
    # best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


# Define the genetic algorithm with selection_method parameter
def genetic_algorithm_with_age(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                               K=None, age_alpha=0.5):

    # Initialize the population with random individuals and their ages
    population = []
    ages = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)
        ages.append(0)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False
    for generation in range(max_generations):
        # fitness of each individual considering age
        fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
                     i, individual in enumerate(population)]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Scale fitness values based on the current generation
        scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # genetic_algorithm function snippet
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses, scaled_fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses, scaled_fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, scaled_fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < GA_MUTATION:
                        child[i] = chr(random.randint(32, 126))
            offspring.append(child1)
            offspring.append(child2)

        # Update the ages of the population
        ages = [age + 1 for age in ages]

        # Add new offspring to the population and reset their ages
        population = elites + offspring
        ages[:elite_size] = [0] * elite_size

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                               num_unique_alleles)

        # Plot the fitness histogram every 10 generations
        if (generation + 1) % 10 == 0:
            plot_fitness_histogram(generation, population, fitness_func)

        # Calculate the best individual in the current generation
        best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
            individual)], fitness_func, age_alpha, max_age=max_generations))

        # Check if a solution is found
        if solution_found(best_individual):
            solution_found_flag = True
            break

    # Find the individual with the highest fitness considering age
    best_individual = max(population, key=lambda individual: fitness_with_age(individual,
                                                                              ages[population.index(individual)],
                                                                              fitness_func, age_alpha,
                                                                              max_age=max_generations))
    best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
                                    age_alpha, max_age=max_generations)

    if solution_found_flag:
        print("Solution found.")
    else:
        print("Solution not found within the given generations.")

    return best_individual, best_fitness


def generate_individual(n):
    return random.sample(range(n), n)


def generate_population(pop_size, n):
    return [generate_individual(n) for _ in range(pop_size)]


def genetic_algorithm_NQueens(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                              K=None, exchange_operator="CX"):

    # Initialize the population with random individuals
    population = generate_population(pop_size, num_genes)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False
    for generation in range(max_generations):
        # Calculate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]

        # Check if a solution is found
        if min(fitnesses) == 0:
            solution_found = True
            break

        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses, fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            if exchange_operator == "CX":
                child1, child2 = cx(parent1, parent2)
            elif exchange_operator == "PMX":
                child1, child2 = pmx(parent1, parent2)

            for child in [child1, child2]:
                if random.random() < GA_MUTATIONRATE:
                    child = inversion_mutation(child)
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        # Print the statistics and running time for the current generation
        print_generation_stats_NQueens(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                                       num_unique_alleles)

    # Find the individual with the highest fitness
    best_individual = min(population, key=fitness_func)
    best_fitness = fitness_func(best_individual)

    if solution_found:
        print("Solution found.")
    else:
        print("Solution not found within the given generations.")

    return best_individual, best_fitness


# # Define the genetic algorithm function for the N-Queens problem
# def genetic_algorithm_NQueens_analysis(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
#                               K=None, exchange_operator="CX", mutation_rate=0.01, elitism_rate=0.1, aging_rate=0.1):
#
#     # Initialize the population with random individuals
#     population = generate_population(pop_size, num_genes)
#
#     # Evolve the population for a fixed number of generations
#     start_time = time.time()
#     start_clock = time.process_time()
#     solution_found = False
#     generation_times = []
#     best_fitnesses = []
#     for generation in range(max_generations):
#         # Calculate the fitness of each individual
#         fitnesses = [fitness_func(individual) for individual in population]
#
#         # Check if a solution is found
#         if min(fitnesses) == 0:
#             solution_found = True
#             break
#
#         elite_size = int(pop_size * elitism_rate)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             if selection_method == "RWS":
#                 parent1 = roulette_wheel_selection(population, fitnesses, fitnesses)
#                 parent2 = roulette_wheel_selection(population, fitnesses, fitnesses)
#             elif selection_method == "SUS":
#                 parent1, parent2 = stochastic_universal_sampling(population, fitnesses, fitnesses, 2)
#             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
#                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
#
#             if exchange_operator == "CX":
#                 child1, child2 = cx(parent1, parent2)
#             elif exchange_operator == "PMX":
#                 child1, child2 = pmx(parent1, parent2)
#
#             for child in [child1, child2]:
#                 if random.random() < mutation_rate:
#                     child = inversion_mutation(child)
#             offspring.append(child1)
#             offspring.append(child2)
#
#         population = elites + offspring
#
#         # Calculate genetic diversification metrics
#         avg_gen_dist = average_genetic_distance(population)
#         num_unique_alleles = unique_alleles(population)
#
#         # Store the statistics and running time for the current generation
#         generation_times.append(time.process_time() - start_clock)
#         best_fitnesses.append(min(fitnesses))
#
#     # Find the individual with the highest fitness
#     best_individual = min(population, key=fitness_func)
#     best_fitness = fitness_func(best_individual)
#
#     if solution_found:
#         print("Solution found.")
#     else:
#         print("Solution not found within the given generations.")
#
#     if len(best_fitnesses) == len(generation_times):
#         plt.plot(range(max_generations), best_fitnesses, label="Best Fitness")
#         plt.plot(range(max_generations), generation_times, label="Generation Time")
#         plt.legend()
#         plt.xlabel("Generation")
#         plt.ylabel("Performance")
#         plt.title("Performance of Genetic Algorithm for N-Queens Problem")
#         plt.show()
#     else:
#         print("Data lists have different lengths.")
#         print(f"Best Fitnesses length: {len(best_fitnesses)}")
#         print(f"Generation Times length: {len(generation_times)}")
#
#     return best_individual, best_fitness


def generate_bin_packing_individual(num_items, num_bins):
    return [random.randint(0, num_bins - 1) for _ in range(num_items)]


def genetic_algorithm_BinPacking(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                                 K=None, item_sizes=None, bin_capacity=None):

    # Initialize the population with random individuals
    population = [generate_bin_packing_individual(num_genes, max(item_sizes)) for _ in range(pop_size)]

    # Calculate the lower bound of bins needed
    lower_bound = math.ceil(sum(item_sizes) / bin_capacity)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False
    for generation in range(max_generations):
        # Calculate the fitness of each individual
        fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]

        # Check if a solution is found
        if min(fitnesses) <= lower_bound:
            solution_found = True
            break

        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses, fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)

            for child in [child1, child2]:
                if random.random() < GA_MUTATIONRATE:
                    child = inversion_mutation(child)
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        # Print the statistics, genetic diversification metrics, and running time for the current generation
        print_generation_stats_BinPacking(generation, population, fitness_func, start_time, start_clock, item_sizes,
                                          bin_capacity, avg_gen_dist, num_unique_alleles)

    # Find the individual with the highest fitness
    best_individual = min(population, key=lambda x: fitness_func(x, item_sizes, bin_capacity))
    best_fitness = fitness_func(best_individual, item_sizes, bin_capacity)

    if solution_found:
        print("Solution found.")
    else:
        print("Solution not found within the given generations.")

    return best_individual, best_fitness
