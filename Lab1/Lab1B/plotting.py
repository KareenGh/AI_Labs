import math
import statistics
import time

from matplotlib import pyplot as plt

from LoadFiles import read_binpack_input_file
from fitness import bin_packing_fitness
from utils import average_hamming_distance, first_fit, fitness_variance, top_avg_selection_probability_ratio


# Define a function to calculate and print the statistics for each generation
def print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                           num_unique_alleles):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_func(individual) for individual in population]
    avg_fitness = statistics.mean(fitnesses)
    std_fitness = statistics.stdev(fitnesses, avg_fitness)

    # Calculate the running time for the current generation
    elapsed_time = time.time() - start_time
    clock_ticks = time.process_time() - start_clock

    # Print the statistics and running time for the current generation
    print(f"Generation {generation + 1}:")
    print("Individual:", ''.join(population[generation]))
    print("    Average fitness =", avg_fitness)
    print(f"    Standard deviation =", std_fitness)
    print(f"    Elapsed time = {elapsed_time:.4f}")
    print(f"    Clock ticks = {clock_ticks}")
    diversity = average_hamming_distance(population)
    print(f"    Diversity = {diversity}")
    print(f"    Average genetic distance: {avg_gen_dist:.2f}")
    print(f"    Number of unique alleles: {num_unique_alleles}")
    print("-" * 50)

    return elapsed_time, clock_ticks


# Define a function to plot a histogram of the fitness distribution
def plot_fitness_histogram(generation, population, fitness_func):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_func(individual) for individual in population]

    # Plot the histogram
    plt.hist(fitnesses, bins=math.ceil(math.sqrt(len(population))))
    plt.title(f"Fitness Distribution (Generation {generation+1})")
    plt.xlabel("Fitness")
    plt.ylabel("Number of Individuals")
    plt.show()


# Define a function to calculate and print the statistics for each generation
def print_generation_stats_NQueens(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
                                   num_unique_alleles):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_func(individual) for individual in population]
    avg_fitness = statistics.mean(fitnesses)
    std_fitness = statistics.stdev(fitnesses, avg_fitness)

    # Calculate the running time for the current generation
    elapsed_time = time.time() - start_time
    clock_ticks = time.process_time() - start_clock

    # Find the best individual in the current generation
    best_individual = min(population, key=fitness_func)
    best_fitness = fitness_func(best_individual)

    # Print the statistics and running time for the current generation
    print(f"Generation {generation + 1}:")
    print("Best Individual:", best_individual)
    print("    Best fitness =", best_fitness)
    print("    Average fitness =", avg_fitness)
    print(f"    Standard deviation =", std_fitness)
    print(f"    Elapsed time = {elapsed_time:.4f}")
    print(f"    Clock ticks = {clock_ticks}")
    diversity = average_hamming_distance(population)
    print(f"    Diversity = {diversity}")
    print(f"    Average genetic distance: {avg_gen_dist:.2f}")
    print(f"    Number of unique alleles: {num_unique_alleles}")
    print("-" * 50)

    return elapsed_time, clock_ticks


def print_generation_stats_BinPacking(generation, population, fitness_func, start_time, start_clock, item_sizes,
                                      bin_capacity, avg_gen_dist, num_unique_alleles):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]
    avg_fitness = statistics.mean(fitnesses)
    std_fitness = statistics.stdev(fitnesses, avg_fitness)

    # Calculate the running time for the current generation
    elapsed_time = time.time() - start_time
    clock_ticks = time.process_time() - start_clock

    # Find the best individual in the current generation
    best_individual = min(population, key=lambda x: fitness_func(x, item_sizes, bin_capacity))
    best_fitness = fitness_func(best_individual, item_sizes, bin_capacity)

    # Print the statistics and running time for the current generation
    print(f"Generation {generation + 1}:")
    print("Best Individual:", best_individual)
    print("    Best fitness =", best_fitness)
    print("    Average fitness =", avg_fitness)
    print(f"    Standard deviation =", std_fitness)
    print(f"    Elapsed time = {elapsed_time:.4f}")
    print(f"    Clock ticks = {clock_ticks}")
    diversity = average_hamming_distance(population)
    print(f"    Diversity = {diversity}")
    print(f"    Average genetic distance: {avg_gen_dist:.2f}")
    print(f"    Number of unique alleles: {num_unique_alleles}")
    print("-" * 50)

    return elapsed_time, clock_ticks


def compare_binpack_algorithms(file_path, pop_size=100, max_generations=100, crossover_type="single_point",
                               selection_method="RWS", k=None):
    import Genetic_Algorithms  # import the module here instead
    ga_runtimes = []
    first_fit_runtimes = []
    ga_solutions = []
    first_fit_solutions = []

    # Run the genetic algorithm and First Fit algorithm on each problem instance in the input file
    for bin_capacity, item_sizes in read_binpack_input_file(file_path):
        # Run the genetic algorithm
        start_time = time.time()
        best_individual, best_fitness = Genetic_Algorithms.genetic_algorithm_BinPacking(pop_size=pop_size, num_genes=len(item_sizes),
                                                                     fitness_func=bin_packing_fitness,
                                                                     max_generations=max_generations,
                                                                     crossover_type=crossover_type,
                                                                     selection_method=selection_method, K=k,
                                                                     item_sizes=item_sizes,
                                                                     bin_capacity=bin_capacity)
        ga_runtime = time.time() - start_time
        ga_solution = best_fitness

        # Run the First Fit algorithm
        start_time = time.time()
        first_fit_solution = first_fit(item_sizes, bin_capacity)
        first_fit_runtime = time.time() - start_time

        # Add the results to the lists
        ga_runtimes.append(ga_runtime)
        first_fit_runtimes.append(first_fit_runtime)
        ga_solutions.append(ga_solution)
        first_fit_solutions.append(first_fit_solution)

    # Plot the runtime comparison
    plt.figure(figsize=(10, 6))
    plt.hist(ga_runtimes, alpha=0.5, label="GA")
    plt.hist(first_fit_runtimes, alpha=0.5, label="First Fit")
    plt.title("Runtime Comparison")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot the solution comparison
    plt.figure(figsize=(10, 6))
    plt.hist(ga_solutions, alpha=0.5, label="GA")
    plt.hist(first_fit_solutions, alpha=0.5, label="First Fit")
    plt.title("Solution Comparison")
    plt.xlabel("Number of Bins")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Print the average runtime and solution quality of each algorithm
    print("GA Average Runtime:", sum(ga_runtimes) / len(ga_runtimes))
    print("First Fit Average Runtime:", sum(first_fit_runtimes) / len(first_fit_runtimes))
    print("GA Average Solution Quality:", sum(ga_solutions) / len(ga_solutions))
    print("First Fit Average Solution Quality:", sum(first_fit_solutions) / len(first_fit_solutions))


# # Define the function to compare the GA_binPack and First Fit algorithms on the Bin Packing problem
# def compare_binpack_algorithms(file_path, pop_size=100, max_generations=100, crossover_type="single_point",
#                                selection_method="RWS", k=None):
#     ga_runtimes = []
#     first_fit_runtimes = []
#     ga_solutions = []
#     first_fit_solutions = []
#
#     # Run the genetic algorithm and First Fit algorithm on each problem instance in the input file
#     for bin_capacity, item_sizes in read_binpack_input_file(file_path):
#         # Run the genetic algorithm
#         start_time = time.time()
#         best_individual, best_fitness = genetic_algorithm_BinPacking(pop_size=pop_size, num_genes=len(item_sizes),
#                                                                      fitness_func=bin_packing_fitness,
#                                                                      max_generations=max_generations,
#                                                                      crossover_type=crossover_type,
#                                                                      selection_method=selection_method, K=k,
#                                                                      item_sizes=item_sizes,
#                                                                      bin_capacity=bin_capacity)
#         ga_runtime = time.time() - start_time
#         ga_solution = best_fitness
#
#         # Run the First Fit algorithm
#         start_time = time.time()
#         first_fit_solution = first_fit(item_sizes, bin_capacity)
#         first_fit_runtime = time.time() - start_time
#
#         # Add the results to the lists
#         ga_runtimes.append(ga_runtime)
#         first_fit_runtimes.append(first_fit_runtime)
#         ga_solutions.append(ga_solution)
#         first_fit_solutions.append(first_fit_solution)
#
#     # Plot the runtime comparison
#     plt.figure(figsize=(10, 6))
#     plt.hist(ga_runtimes, alpha=0.5, label="GA")
#     plt.hist(first_fit_runtimes, alpha=0.5, label="First Fit")
#     plt.title("Runtime Comparison")
#     plt.xlabel("Runtime (s)")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()
#
#     # Plot the solution comparison
#     plt.figure(figsize=(10, 6))
#     plt.hist(ga_solutions, alpha=0.5, label="GA")
#     plt.hist(first_fit_solutions, alpha=0.5, label="First Fit")
#     plt.title("Solution Comparison")
#     plt.xlabel("Number of Bins")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()
#
#     # Print the average runtime and solution quality of each algorithm
#     print("GA Average Runtime:", sum(ga_runtimes) / len(ga_runtimes))
#     print("First Fit Average Runtime:", sum(first_fit_runtimes) / len(first_fit_runtimes))
#     print("GA Average Solution Quality:", sum(ga_solutions) / len(ga_solutions))
#     print("First Fit Average Solution Quality:", sum(first_fit_solutions) / len(first_fit_solutions))


# def print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist, num_unique_alleles):
#     fitnesses = [fitness_func(individual) for individual in population]
#     best_fitness = max(fitnesses)
#     average_fitness = statistics.mean(fitnesses)
#     stdev_fitness = statistics.stdev(fitnesses) if len(population) > 1 else 0
#     best_individual = max(population, key=fitness_func)
#
#     print(f"Generation {generation}:")
#     print(f"  Elapsed time: {time.time() - start_time:.2f} seconds")
#     print(f"  Elapsed process time: {time.process_time() - start_clock:.2f} seconds")
#     print(f"  Best fitness: {best_fitness}")
#     print(f"  Average fitness: {average_fitness:.2f}")
#     print(f"  Standard deviation of fitness: {stdev_fitness:.2f}")
#     print(f"  Best individual: {''.join(best_individual)}")
#     print(f"  Average genetic distance: {avg_gen_dist:.2f}")
#     print(f"  Number of unique alleles: {num_unique_alleles}")
#     print()


def print_generation_stats5(generation, population, fitness_func, start_time, start_clock, item_sizes, bin_capacity, selection_method, K, problem="n_queens"):
    # Evaluate the fitness of each individual
    if problem == "n_queens":
        fitnesses = [fitness_func(individual) for individual in population]
    elif problem == "bin_packing":
        fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]

    avg_fitness = statistics.mean(fitnesses)
    std_fitness = statistics.stdev(fitnesses, avg_fitness)

    # Calculate fitness variance
    fit_variance = fitness_variance(population, fitness_func, item_sizes, bin_capacity, problem)

    # Calculate top-average selection probability ratio
    top_avg_ratio = top_avg_selection_probability_ratio(population, selection_method, K)

    # Calculate the running time for the current generation
    elapsed_time = time.time() - start_time
    clock_ticks = time.process_time() - start_clock

    # Find the best individual in the current generation
    if problem == "n_queens":
        best_individual = min(population, key=fitness_func)
        best_fitness = fitness_func(best_individual)
    elif problem == "bin_packing":
        best_individual = min(population, key=lambda x: fitness_func(x, item_sizes, bin_capacity))
        best_fitness = fitness_func(best_individual, item_sizes, bin_capacity)

    # Print the statistics and running time for the current generation
    print(f"Generation {generation + 1}:")
    print("Best Individual:", best_individual)
    print("    Best fitness =", best_fitness)
    print("    Average fitness =", avg_fitness)
    print(f"    Standard deviation =", std_fitness)
    print(f"    Elapsed time = {elapsed_time:.4f}")
    print(f"    Clock ticks = {clock_ticks:.4f}")
    diversity = average_hamming_distance(population)
    print(f"    Diversity = {diversity:.4f}")
    # Print fitness variance and top-average selection probability ratio
    print(f"    Fitness variance = {fit_variance:.4f}")
    if top_avg_ratio is not None:
        print(f"    Top-Average Selection Probability Ratio = {top_avg_ratio}")
    print("-" * 50)

    return elapsed_time, clock_ticks
