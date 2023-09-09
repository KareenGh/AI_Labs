import math
import random
import statistics
import time

from matplotlib import pyplot as plt

from Metrics import average_kendall_tau_distance, non_attacking_pairs_ratio
from Niching import speciation_with_clustering, fitness_sharing, crowding
from crossover import crossover, cx, pmx, crossover_binary
from fitness import fitness_with_age
from genetic_diversity import average_genetic_distance, unique_alleles
from mutation import update_mutation_probability, mutate_binary, mutate_individual_species
from plotting import print_generation_stats, plot_fitness_histogram, print_generation_stats_NQueens, \
    print_generation_stats_BinPacking
from selection import scale_fitness, roulette_wheel_selection, stochastic_universal_sampling, \
    ranking_and_tournament_selection, roulette_wheel_selection_species
from utils import inversion_mutation, adjust_mutation_rate

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = int(GA_MUTATIONRATE * 100)
GA_TARGET = "Hello, world!"
INCREASE_FACTOR = 1.1
DECREASE_FACTOR = 0.9
MIN_MUTATION_RATE = 0.01
MAX_MUTATION_RATE = 0.5
GA_TARGET_DECODED = "".join(format(ord(c), '07b') for c in GA_TARGET)


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
                parent1 = roulette_wheel_selection(population, scaled_fitnesses)
                parent2 = roulette_wheel_selection(population, scaled_fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
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
                               K=None, age_alpha=0.5, mutation_method="ADAPTIVE", threshold=0.8, stagnation_gen=0,
                               decay_rate=0.99):
    character_ga_times = []

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

    # Initialize the mutation rate
    mutation_rate = GA_MUTATIONRATE
    avg_gen_dist_list = []
    num_unique_alleles_list = []

    for generation in range(max_generations):
        # fitness of each individual considering age
        fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
                     i, individual in enumerate(population)]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Update mutation rate based on the selected method
        mutation_rate = update_mutation_probability(mutation_method, generation, mutation_rate, population, fitnesses,
                                                    fitness_func, threshold, stagnation_gen, decay_rate)

        # Scale fitness values based on the current generation
        scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # genetic_algorithm function snippet
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, scaled_fitnesses)
                parent2 = roulette_wheel_selection(population, scaled_fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < mutation_rate:
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

        avg_gen_dist_list.append(avg_gen_dist)
        num_unique_alleles_list.append(num_unique_alleles)

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                               num_unique_alleles)

        # # Plot the fitness histogram every 10 generations
        # if (generation + 1) % 10 == 0:
        #     plot_fitness_histogram(generation, population, fitness_func)

        # Calculate the best individual in the current generation
        best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
            individual)], fitness_func, age_alpha, max_age=max_generations))

        # Check if the target solution is found
        if ''.join(population[generation]) == GA_TARGET:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_fitness_histogram(generation, population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

    # Find the individual with the highest fitness considering age
    best_individual = max(population, key=lambda individual: fitness_with_age(individual,
                                                                              ages[population.index(individual)],
                                                                              fitness_func, age_alpha,
                                                                              max_age=max_generations))
    best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
                                    age_alpha, max_age=max_generations)

    generation_time = time.time() - start_time
    character_ga_times.append(generation_time)

    return best_individual, best_fitness, character_ga_times, population, avg_gen_dist_list, num_unique_alleles_list


def generate_individual(n):
    return random.sample(range(n), n)


def generate_population(pop_size, n):
    return [generate_individual(n) for _ in range(pop_size)]


def genetic_algorithm_NQueens(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                              K=None, exchange_operator="CX", DIVERSITY_THRESHOLD=10, NAPR_THRESHOLD=0.95,
                              mutation_method="ADAPTIVE", mutation_prob=0.25, threshold=0.8, stagnation_gen=0,
                              decay_rate=0.99,
                              constant_mutation_rate=False):
    # Initialize the population with random individuals
    population = generate_population(pop_size, num_genes)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False

    # Initialize the mutation rate
    mutation_rate = GA_MUTATIONRATE
    avg_gen_dist_list = []
    num_unique_alleles_list = []

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

        # Update mutation rate based on the selected method
        if not constant_mutation_rate:
            mutation_rate = update_mutation_probability(mutation_method, generation, mutation_prob, population,
                                                        fitnesses, fitness_func, threshold, stagnation_gen, decay_rate)

        # Calculate the metrics
        non_attacking_ratio = non_attacking_pairs_ratio(population)
        avg_kendall_tau_distance = average_kendall_tau_distance(population)

        # # Adjust the mutation rate based on the diversity metric and non-attacking pairs ratio
        # if avg_kendall_tau_distance < DIVERSITY_THRESHOLD or non_attacking_ratio > NAPR_THRESHOLD:
        #     mutation_rate *= INCREASE_FACTOR
        # else:
        #     mutation_rate *= DECREASE_FACTOR
        # mutation_rate = min(max(mutation_rate, MIN_MUTATION_RATE), MAX_MUTATION_RATE)

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            if exchange_operator == "CX":
                child1, child2 = cx(parent1, parent2)
            elif exchange_operator == "PMX":
                child1, child2 = pmx(parent1, parent2)

            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    child = inversion_mutation(child)
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        avg_gen_dist_list.append(avg_gen_dist)
        num_unique_alleles_list.append(num_unique_alleles)

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

    # return best_individual, best_fitness
    return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation, avg_gen_dist_list, num_unique_alleles_list


def generate_bin_packing_individual(num_items, num_bins):
    return [random.randint(0, num_bins - 1) for _ in range(num_items)]


def genetic_algorithm_BinPacking(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                                 K=None, item_sizes=None, bin_capacity=None,
                                 mutation_method="ADAPTIVE", mutation_prob=0.25, threshold=0.8, stagnation_gen=0,
                                 decay_rate=0.99,
                                 constant_mutation_rate=False):
    # Initialize the population with random individuals
    population = [generate_bin_packing_individual(num_genes, max(item_sizes)) for _ in range(pop_size)]

    # Calculate the lower bound of bins needed
    lower_bound = math.ceil(sum(item_sizes) / bin_capacity)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False

    # Initialize the mutation rate
    mutation_rate = GA_MUTATIONRATE
    avg_gen_dist_list = []
    num_unique_alleles_list = []

    # Calculate the lower bound of bins needed
    tolerance = 1.1
    lower_bound = math.ceil(sum(item_sizes) / bin_capacity)
    target_fitness = lower_bound * tolerance

    for generation in range(max_generations):
        # Calculate the fitness of each individual
        fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]

        # Check if a solution is found
        if min(fitnesses) <= target_fitness:  # Modified termination condition
            solution_found = True
            break

        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Update mutation rate based on the selected method
        if not constant_mutation_rate:
            mutation_rate = update_mutation_probability(mutation_method, generation, mutation_prob, population,
                                                        fitnesses,
                                                        fitness_func, threshold, stagnation_gen, decay_rate,
                                                        fitness_func_args=
                                                        {'item_sizes': item_sizes, 'bin_capacity': bin_capacity})

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)

            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    child = inversion_mutation(child)
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Calculate genetic diversification metrics
        avg_gen_dist = average_genetic_distance(population)
        num_unique_alleles = unique_alleles(population)

        avg_gen_dist_list.append(avg_gen_dist)
        num_unique_alleles_list.append(num_unique_alleles)

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

    return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation, avg_gen_dist_list, num_unique_alleles_list


def generate_binary_individual(num_genes):
    return [format(random.randint(32, 126), '07b') for _ in range(num_genes)]


def genetic_algorithm_Binary(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                             K=None, age_alpha=0.5, mutation_method="ADAPTIVE", mutation_prob=0.25, threshold=0.8,
                             stagnation_gen=0,
                             decay_rate=0.99, constant_mutation_rate=False):
    binary_ga_times = []
    # Initialize the population with random individuals and their ages
    population = []
    ages = []
    for i in range(pop_size):
        individual = generate_binary_individual(num_genes)
        population.append(individual)
        ages.append(0)

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False

    # Initialize the mutation rate
    mutation_rate = GA_MUTATIONRATE
    avg_gen_dist_list = []
    num_unique_alleles_list = []

    for generation in range(max_generations):
        # fitness of each individual considering age
        fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
                     i, individual in enumerate(population)]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Update mutation rate based on the selected method
        if not constant_mutation_rate:
            mutation_rate = update_mutation_probability(mutation_method, generation, mutation_prob, population,
                                                        fitnesses, fitness_func, threshold, stagnation_gen, decay_rate)

        # Scale fitness values based on the current generation
        scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # genetic_algorithm function snippet
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, scaled_fitnesses)
                parent2 = roulette_wheel_selection(population, scaled_fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover_binary(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < mutation_rate:
                        child[i] = format(random.randint(32, 126), '07b')
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

        avg_gen_dist_list.append(avg_gen_dist)
        num_unique_alleles_list.append(num_unique_alleles)

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                               num_unique_alleles)

        # # Plot the fitness histogram every 10 generations
        # if (generation + 1) % 10 == 0:
        #     plot_fitness_histogram(generation, population, fitness_func)

        # Calculate the best individual in the current generation
        best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
            individual)], fitness_func, age_alpha, max_age=max_generations))

        # Check if the target solution is found
        if ''.join(population[generation]) == GA_TARGET_DECODED:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                                   num_unique_alleles)
            # plot_fitness_histogram(generation, population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            generation_time = time.time() - start_time
            binary_ga_times.append(generation_time)

            return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation, binary_ga_times, population, \
                   avg_gen_dist_list, num_unique_alleles_list

    # Find the individual with the highest fitness considering age
    best_individual = max(population, key=lambda individual: fitness_with_age(individual,
                                                                              ages[population.index(individual)],
                                                                              fitness_func, age_alpha,
                                                                              max_age=max_generations))
    best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
                                    age_alpha, max_age=max_generations)

    generation_time = time.time() - start_time
    binary_ga_times.append(generation_time)

    return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation, binary_ga_times, population, \
           avg_gen_dist_list, num_unique_alleles_list


# GA Binary
# def genetic_algorithm_Binary(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
#                              K=None, age_alpha=0.5):
#     # Initialize the population with random individuals and their ages
#     population = []
#     ages = []
#     for i in range(pop_size):
#         individual = generate_binary_individual(num_genes)
#         population.append(individual)
#         ages.append(0)
#
#     # Evolve the population for a fixed number of generations
#     start_time = time.time()
#     start_clock = time.process_time()
#     solution_found = False
#     for generation in range(max_generations):
#         # fitness of each individual considering age
#         fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
#                      i, individual in enumerate(population)]
#         elite_size = int(pop_size * GA_ELITRATE)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Scale fitness values based on the current generation
#         scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             # genetic_algorithm function snippet
#             if selection_method == "RWS":
#                 parent1 = roulette_wheel_selection(population, scaled_fitnesses)
#                 parent2 = roulette_wheel_selection(population, scaled_fitnesses)
#             elif selection_method == "SUS":
#                 parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
#             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
#                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
#
#             child1, child2 = crossover(parent1, parent2, crossover_type)
#             for child in [child1, child2]:
#                 for i in range(num_genes):
#                     if random.randint(0, 100) < GA_MUTATION:
#                         child[i] = format(random.randint(32, 126), '07b')
#             offspring.append(child1)
#             offspring.append(child2)
#
#         # Update the ages of the population
#         ages = [age + 1 for age in ages]
#
#         # Add new offspring to the population and reset their ages
#         population = elites + offspring
#         ages[:elite_size] = [0] * elite_size
#
#         # Calculate genetic diversification metrics
#         avg_gen_dist = average_genetic_distance(population)
#         num_unique_alleles = unique_alleles(population)
#
#         # Print the statistics and running time for the current generation
#         print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
#                                num_unique_alleles)
#
#         # Plot the fitness histogram every 10 generations
#         if (generation + 1) % 10 == 0:
#             plot_fitness_histogram(generation, population, fitness_func)
#
#         # Calculate the best individual in the current generation
#         best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
#             individual)], fitness_func, age_alpha, max_age=max_generations))
#
#         # Check if the target solution is found
#         if ''.join(population[generation]) == GA_TARGET:
#             print_generation_stats(generation, population, fitness_func, start_time, start_clock)
#             plot_fitness_histogram(generation, population, fitness_func)
#             print("Solution Found!")
#             best_individual = max(population, key=lambda individual: fitness_func(individual))
#             best_fitness = fitness_func(best_individual)
#
#             return best_individual, best_fitness
#
#     # Find the individual with the highest fitness considering age
#     best_individual = max(population, key=lambda individual: fitness_with_age(individual,
#                                                                               ages[population.index(individual)],
#                                                                               fitness_func, age_alpha,
#                                                                               max_age=max_generations))
#     best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
#                                     age_alpha, max_age=max_generations)
#
#     return best_individual, best_fitness
#
#
# def generate_binary_individual(num_genes):
#     return [format(random.randint(32, 126), '07b') for _ in range(num_genes)]

#
# def genetic_algorithm_Binary(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
#                              K=None, age_alpha=0.5,
#                              mutation_method="ADAPTIVE", threshold=0.8, stagnation_gen=0, decay_rate=0.99,
#                              constant_mutation_rate=False):
#     # Initialize the population with random individuals and their ages
#     population = []
#     ages = []
#     for i in range(pop_size):
#         individual = generate_binary_individual(num_genes)
#         population.append(individual)
#         ages.append(0)
#
#     # Evolve the population for a fixed number of generations
#     start_time = time.time()
#     start_clock = time.process_time()
#     solution_found = False
#
#     # Initialize the mutation rate
#     mutation_rate = GA_MUTATIONRATE
#
#     for generation in range(max_generations):
#         # fitness of each individual considering age
#         fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
#                      i, individual in enumerate(population)]
#         elite_size = int(pop_size * GA_ELITRATE)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Scale fitness values based on the current generation
#         scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)
#
#         # Update mutation rate based on the selected method
#         if not constant_mutation_rate:
#             mutation_rate = update_mutation_probability(mutation_method, generation, mutation_rate, population, fitnesses,
#                                                     fitness_func, threshold, stagnation_gen, decay_rate)
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             # genetic_algorithm function snippet
#             if selection_method == "RWS":
#                 parent1 = roulette_wheel_selection(population, scaled_fitnesses)
#                 parent2 = roulette_wheel_selection(population, scaled_fitnesses)
#             elif selection_method == "SUS":
#                 parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
#             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
#                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
#
#             if parent1 is None:
#                 parent1 = generate_binary_individual(num_genes)
#             if parent2 is None:
#                 parent2 = generate_binary_individual(num_genes)
#
#             child1, child2 = crossover_binary(parent1, parent2, crossover_type)
#             for child in [child1, child2]:
#                 for i in range(num_genes):
#                     if random.randint(0, 100) < mutation_rate:
#                         child[i] = format(random.randint(32, 126), '07b')
#             offspring.append(child1)
#             offspring.append(child2)
#
#         # Update the ages of the population
#         ages = [age + 1 for age in ages]
#         # Add new offspring to the population and reset their ages
#         population = elites + offspring
#         ages[:elite_size] = [0] * elite_size
#
#         # Calculate genetic diversification metrics
#         avg_gen_dist = average_genetic_distance(population)
#         num_unique_alleles = unique_alleles(population)
#
#         # Print the statistics and running time for the current generation
#         print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
#                                num_unique_alleles)
#
#         # Calculate the best individual in the current generation
#         best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
#             individual)], fitness_func, age_alpha, max_age=max_generations))
#
#         # Check if the target solution is found
#         if ''.join(best_individual) == GA_TARGET:
#             print_generation_stats(generation, population, fitness_func, start_time, start_clock)
#             plot_fitness_histogram(generation, population, fitness_func)
#             print("Solution Found!")
#             best_fitness = fitness_func(best_individual)
#
#             return best_individual, best_fitness
#
#     # Find the individual with the highest fitness considering age
#     best_individual = max(population, key=lambda individual: fitness_with_age(individual,
#                                                                               ages[population.index(individual)],
#                                                                               fitness_func, age_alpha,
#                                                                               max_age=max_generations))
#     best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
#                                     age_alpha, max_age=max_generations)
#
#     return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation
#

# def initialize_population(pop_size, num_genes):
#     population = []
#     for _ in range(pop_size):
#         individual = [random.randint(0, 1) for _ in range(num_genes)]
#         population.append(individual)
#     return population
#
#
# def genetic_algorithm_Binary(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
#                                       K=None, age_alpha=0.5, mutation_method="ADAPTIVE", threshold=0.8, stagnation_gen=0,
#                                       decay_rate=0.99):
#     # Initialize the population with binary individuals and their ages
#     population = initialize_population(pop_size, num_genes)
#     ages = [0] * pop_size
#
#     # Initialize the mutation rate
#     mutation_rate = GA_MUTATIONRATE
#
#     # Evolve the population for a fixed number of generations
#     start_time = time.time()
#     start_clock = time.process_time()
#
#     for generation in range(max_generations):
#         # fitness of each individual considering age
#         fitnesses = [fitness_with_age(individual, ages[i], fitness_func, age_alpha, max_age=max_generations) for
#                      i, individual in enumerate(population)]
#         elite_size = int(pop_size * GA_ELITRATE)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Update mutation rate based on the selected method
#         mutation_rate = update_mutation_probability(mutation_method, generation, mutation_rate, population, fitnesses,
#                                                     fitness_func, threshold, stagnation_gen, decay_rate)
#
#         # Scale fitness values based on the current generation
#         scaled_fitnesses = scale_fitness(fitnesses, generation, max_generations)
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             # genetic_algorithm function snippet
#             if selection_method == "RWS":
#                 parent1 = roulette_wheel_selection(population, scaled_fitnesses)
#                 parent2 = roulette_wheel_selection(population, scaled_fitnesses)
#             elif selection_method == "SUS":
#                 parent1, parent2 = stochastic_universal_sampling(population, scaled_fitnesses, 2)
#             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
#                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
#
#             child1, child2 = crossover(parent1, parent2, crossover_type)
#             for child in [child1, child2]:
#                 for i in range(len(child)):
#                     if random.random() < mutation_rate:
#                         child = mutate_binary(child, mutation_rate)
#             offspring.append(child1)
#             offspring.append(child2)
#
#         # Update the ages of the population
#         ages = [age + 1 for age in ages]
#
#         # Add new offspring to the population and reset their ages
#         population = elites + offspring
#         ages[:elite_size] = [0] * elite_size
#
#         # Calculate genetic diversification metrics
#         avg_gen_dist = average_genetic_distance(population)
#         num_unique_alleles = unique_alleles(population)
#
#         # Print the statistics and running time for the current generation
#         print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
#                                num_unique_alleles)
#
#         # Plot the fitness histogram every 10 generations
#         if (generation + 1) % 10 == 0:
#             plot_fitness_histogram(generation, population, fitness_func)
#
#         # Calculate the best individual in the current generation
#         best_individual = max(population, key=lambda individual: fitness_with_age(individual, ages[population.index(
#             individual)], fitness_func, age_alpha, max_age=max_generations))
#
#         # Check if the target solution is found
#         if ''.join(str(gene) for gene in population[generation]) == GA_TARGET:
#             print_generation_stats(generation, population, fitness_func, start_time, start_clock)
#             plot_fitness_histogram(generation, population, fitness_func)
#             print("Solution Found!")
#             best_individual = max(population, key=lambda individual: fitness_func(individual))
#             best_fitness = fitness_func(best_individual)
#
#             return best_individual, best_fitness
#
#     # Find the individual with the highest fitness considering age
#     best_individual = max(population, key=lambda individual: fitness_with_age(individual,
#                                                                               ages[population.index(individual)],
#                                                                               fitness_func, age_alpha,
#                                                                               max_age=max_generations))
#     best_fitness = fitness_with_age(best_individual, ages[population.index(best_individual)], fitness_func,
#                                     age_alpha, max_age=max_generations)
#
#     return best_individual, best_fitness

#
# def GA_niching_BinPacking(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
#                           niching_method, sharing_radius=None, num_clusters=None,
#                           K=None, item_sizes=None, bin_capacity=None,
#                           mutation_method="ADAPTIVE", mutation_prob=0.25, threshold=0.8, stagnation_gen=0,
#                           decay_rate=0.99,
#                           constant_mutation_rate=False):
#     # Initialize the population with random individuals
#     population = [generate_bin_packing_individual(num_genes, max(item_sizes)) for _ in range(pop_size)]
#
#     # Calculate the lower bound of bins needed
#     lower_bound = math.ceil(sum(item_sizes) / bin_capacity)
#
#     # Evolve the population for a fixed number of generations
#     start_time = time.time()
#     start_clock = time.process_time()
#     solution_found = False
#
#     # Initialize the mutation rate
#     mutation_rate = GA_MUTATIONRATE
#
#     # Calculate the lower bound of bins needed
#     tolerance = 1.1
#     lower_bound = math.ceil(sum(item_sizes) / bin_capacity)
#     target_fitness = lower_bound * tolerance
#
#     offspring = []
#     offspring.append(child2)
#
#     for generation in range(max_generations):
#         # Calculate the fitness of each individual
#         fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]
#
#         # Apply the niching method
#         if niching_method == "FITNESS_SHARING":
#             fitnesses = [shared_fitness(individual, population, fitness_func, sharing_radius, item_sizes, bin_capacity)
#                          for individual in population]
#         elif niching_method == "CROWDING":
#             offspring_fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in offspring]
#             offspring = non_deterministic_crowding(population, fitnesses, offspring, offspring_fitnesses)
#             population = non_deterministic_crowding(population, fitnesses, offspring, offspring_fitnesses)
#         elif niching_method == "SPECIATION":
#             species = speciation_with_clustering(population, num_clusters)
#
#             new_population = []
#             min_gene_value = min(item_sizes)
#             max_gene_value = max(item_sizes)
#
#             for spec in species:
#                 spec_fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in spec]
#
#                 # Perform selection, crossover, and mutation within the species
#                 selected_parents = []
#                 for _ in range(len(spec)):
#                     parent1 = roulette_wheel_selection_species(spec, spec_fitnesses)
#                     parent2 = roulette_wheel_selection_species(spec, spec_fitnesses)
#                     selected_parents.append((parent1, parent2))
#
#                 offspring = [pmx(parent1, parent2) for parent1, parent2 in selected_parents]
#                 mutated_offspring = [mutate_individual_species(child, mutation_prob, min_gene_value, max_gene_value) for child in offspring]
#
#                 new_population.extend(mutated_offspring)
#
#             population = new_population
#
#         print(fitnesses)
#         print(target_fitness)
#
#         # Check if a solution is found
#         if min(fitnesses) <= target_fitness:  # Modified termination condition
#             solution_found = True
#             break
#
#         elite_size = int(pop_size * GA_ELITRATE)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Update mutation rate based on the selected method
#         if not constant_mutation_rate:
#             mutation_rate = update_mutation_probability(mutation_method, generation, mutation_prob, population,
#                                                         fitnesses,
#                                                         fitness_func, threshold, stagnation_gen, decay_rate,
#                                                         fitness_func_args=
#                                                         {'item_sizes': item_sizes, 'bin_capacity': bin_capacity})
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             if selection_method == "RWS":
#                 parent1 = roulette_wheel_selection(population, fitnesses)
#                 parent2 = roulette_wheel_selection(population, fitnesses)
#             elif selection_method == "SUS":
#                 parent1, parent2 = stochastic_universal_sampling(population, fitnesses, 2)
#             elif selection_method == "RANKING_TOURNAMENT" and K is not None:
#                 parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)
#
#             child1, child2 = crossover(parent1, parent2, crossover_type)
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
#         # Print the statistics, genetic diversification metrics, and running time for the current generation
#         print_generation_stats_BinPacking(generation, population, fitness_func, start_time, start_clock, item_sizes,
#                                           bin_capacity, avg_gen_dist, num_unique_alleles)
#
#     # Find the individual with the highest fitness
#     best_individual = min(population, key=lambda x: fitness_func(x, item_sizes, bin_capacity))
#     best_fitness = fitness_func(best_individual, item_sizes, bin_capacity)
#
#     if solution_found:
#         print("Solution found.")
#     else:
#         print("Solution not found within the given generations.")
#
#     return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation


def GA_niching_BinPacking(pop_size, num_genes, fitness_func, max_generations, crossover_type, selection_method,
                          K=None, item_sizes=None, bin_capacity=None,
                          mutation_method="ADAPTIVE", mutation_prob=0.25, threshold=0.8, stagnation_gen=0,
                          decay_rate=0.99, constant_mutation_rate=False, niche_method=None, sigma_share=None,
                          alpha=None, crowding_factor=None, num_species=None, clustering_algorithm=None):
    # Initialize the population with random individuals
    population = [generate_bin_packing_individual(num_genes, max(item_sizes)) for _ in range(pop_size)]

    # Evolve the population for a fixed number of generations
    start_time = time.time()
    start_clock = time.process_time()
    solution_found = False

    # Initialize the mutation rate
    mutation_rate = GA_MUTATIONRATE

    # Calculate the lower bound of bins needed
    tolerance = 1.1
    lower_bound = math.ceil(sum(item_sizes) / bin_capacity)
    target_fitness = lower_bound * tolerance

    for generation in range(max_generations):
        # Calculate the fitness of each individual
        fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]

        if niche_method == 'fitness_sharing':
            fitnesses = fitness_sharing(population, fitnesses, sigma_share, alpha)
        elif niche_method == 'crowding':
            fitnesses = crowding(population, fitnesses, crowding_factor)
        elif niche_method == 'speciation_with_clustering':
            fitnesses = speciation_with_clustering(population, fitnesses, num_species, clustering_algorithm)

        # Check if a solution is found
        if min(fitnesses) <= target_fitness:  # Modified termination condition
            solution_found = True
            break

        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Update mutation rate based on the selected method
        if not constant_mutation_rate:
            mutation_rate = update_mutation_probability(mutation_method, generation, mutation_prob, population,
                                                        fitnesses,
                                                        fitness_func, threshold, stagnation_gen, decay_rate,
                                                        fitness_func_args=
                                                        {'item_sizes': item_sizes, 'bin_capacity': bin_capacity})

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if selection_method == "RWS":
                parent1 = roulette_wheel_selection(population, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses)
            elif selection_method == "SUS":
                parent1, parent2 = stochastic_universal_sampling(population, fitnesses, 2)
            elif selection_method == "RANKING_TOURNAMENT" and K is not None:
                parent1, parent2 = ranking_and_tournament_selection(population, fitnesses, K)

            child1, child2 = crossover(parent1, parent2, crossover_type)

            for child in [child1, child2]:
                if random.random() < mutation_rate:
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

    return best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generation
