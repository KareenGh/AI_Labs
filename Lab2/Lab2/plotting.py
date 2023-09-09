import math
import statistics
import time
import pandas as pd

from matplotlib import pyplot as plt

from LoadFiles import read_binpack_input_file
from Metrics import kendall_tau_distance, non_attacking_pairs_ratio, average_kendall_tau_distance
from fitness import bin_packing_fitness
from genetic_diversity import compute_population_diversity
from utils import average_hamming_distance, first_fit, fitness_variance, top_avg_selection_probability_ratio

#
# # Define a function to calculate and print the statistics for each generation
# def print_generation_stats(generation, population, fitness_func, start_time, start_clock, avg_gen_dist,
#                            num_unique_alleles):
#     # Evaluate the fitness of each individual
#     fitnesses = [fitness_func(individual) for individual in population]
#     avg_fitness = statistics.mean(fitnesses)
#     std_fitness = statistics.stdev(fitnesses, avg_fitness)
#
#     # Calculate the running time for the current generation
#     elapsed_time = time.time() - start_time
#     clock_ticks = time.process_time() - start_clock
#
#     # Print the statistics and running time for the current generation
#     print(f"Generation {generation + 1}:")
#     print("Individual:", ''.join(population[generation]))
#     print("    Average fitness =", avg_fitness)
#     print(f"    Standard deviation =", std_fitness)
#     print(f"    Elapsed time = {elapsed_time:.4f}")
#     print(f"    Clock ticks = {clock_ticks}")
#     diversity = average_hamming_distance(population)
#     print(f"    Diversity = {diversity}")
#     print(f"    Average genetic distance: {avg_gen_dist:.2f}")
#     print(f"    Number of unique alleles: {num_unique_alleles}")
#     print("-" * 50)
#
#     return elapsed_time, clock_ticks


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
    print("Individual:", ''.join(str(gene) for gene in population[generation]))
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

    # Calculate and print metrics for the current generation
    non_attacking_ratio = non_attacking_pairs_ratio(population)
    avg_kendall_tau_dist = average_kendall_tau_distance(population)

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
    print(f"    Non-Attacking Pairs Ratio = {non_attacking_ratio:.2f}")
    print(f"    Average Kendall Tau Distance = {avg_kendall_tau_dist:.2f}")

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

    # Calculate the average Kendall's Tau distance for the population
    total_distance = 0
    num_pairs = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = kendall_tau_distance(population[i], population[j])
            total_distance += distance
            num_pairs += 1
    avg_kendall_tau_distance = total_distance / num_pairs if num_pairs > 0 else 0

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
    print(f"    Average Kendall's Tau distance: {avg_kendall_tau_distance}")
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


def Compare_Mutations(problem, avg_gen_dist_adaptive, num_unique_alleles_adaptive, avg_gen_dist_constant,
                      num_unique_alleles_constant, best_fitness_adaptive, generations_adaptive, best_fitness_constant,
                      generations_constant, best_individual_adaptive, best_individual_constant,
                      avg_gen_dist_list, num_unique_alleles_list, mutation):

    # Compare the genetic tone
    print("Genetic tone comparison for problem", problem)
    print("Adaptive mutation rate: Average genetic distance =", avg_gen_dist_adaptive,
          ", Number of unique alleles =", num_unique_alleles_adaptive)
    print("Constant mutation rate: Average genetic distance =", avg_gen_dist_constant,
          ", Number of unique alleles =", num_unique_alleles_constant)

    # Compare the quality and speed of convergence
    print("Quality and speed of convergence comparison for problem", problem)
    print("Adaptive mutation rate: Best fitness =", best_fitness_adaptive, ", Number of generations =",
          generations_adaptive + 1)
    print("Constant mutation rate: Best fitness =", best_fitness_constant, ", Number of generations =",
          generations_constant + 1)

    # Print the best individual for each mutation rate strategy
    print("Best individual for adaptive mutation rate:", best_individual_adaptive)
    print("Best individual for constant mutation rate:", best_individual_constant)

    generations = list(range(len(avg_gen_dist_list)))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Average Genetic Distance', color=color)
    ax1.plot(generations, avg_gen_dist_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Number of Unique Alleles', color=color)
    ax2.plot(generations, num_unique_alleles_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    X = problem
    Y = mutation
    fig.tight_layout()
    plt.title('Problem ' + X + ' with Mutation Mechanism ' + Y)
    plt.show()


def compare_ga_results(best_individual_binary, best_individual, best_fitness_binary, best_fitness, binary_ga_times,
                       character_ga_times, binary_ga_population, character_ga_population, binary_avg_gen_dist_list,
                       character_avg_gen_dist_list):
    print("\nComparison of GA Results:\n")
    print("GA with Binary Representation:")
    print(f"Best Individual: {best_individual_binary}")
    print(f"Best Fitness: {best_fitness_binary}")

    print("\nGA with Character-Based Representation:")
    best_individual_str = ''.join(best_individual)
    print(f"Best Individual: {best_individual_str}")
    print(f"Best Fitness: {best_fitness}")

    # # Compare solution quality
    # if best_fitness_binary > best_fitness:
    #     print("\nBinary GA found a better solution.")
    # elif best_fitness_binary < best_fitness:
    #     print("\nCharacter-Based GA found a better solution.")
    # else:
    #     print("\nBoth GAs found equally good solutions.")

    # Compare convergence speed (assuming both GAs ran for the same number of generations)
    binary_ga_time = sum(generation_time for generation_time in binary_ga_times)
    character_ga_time = sum(generation_time for generation_time in character_ga_times)

    print("\nConvergence Speed Comparison:")
    print(f"Binary GA Time: {binary_ga_time:.2f} seconds")
    print(f"Character-Based GA Time: {character_ga_time:.2f} seconds")

    if binary_ga_time < character_ga_time:
        print("\nBinary GA converged faster.")
    elif binary_ga_time > character_ga_time:
        print("\nCharacter-Based GA converged faster.")
    else:
        print("\nBoth GAs converged at the same speed.")

    # Compare population diversity (average genetic distance and unique alleles)
    binary_avg_gen_dist, binary_num_unique_alleles = compute_population_diversity(binary_ga_population)
    character_avg_gen_dist, character_num_unique_alleles = compute_population_diversity(character_ga_population)

    print("\nPopulation Diversity Comparison:")
    print(
        f"Binary GA - Average Genetic Distance: {binary_avg_gen_dist:.2f}, Unique Alleles: {binary_num_unique_alleles}")
    print(
        f"Character-Based GA - Average Genetic Distance: {character_avg_gen_dist:.2f}, Unique Alleles: {character_num_unique_alleles}")

    if binary_avg_gen_dist > character_avg_gen_dist and binary_num_unique_alleles > character_num_unique_alleles:
        print("\nBinary GA has higher population diversity.")
    elif binary_avg_gen_dist < character_avg_gen_dist and binary_num_unique_alleles < character_num_unique_alleles:
        print("\nCharacter-Based GA has higher population diversity.")
    else:
        print("\nBoth GAs have similar population diversity.")

    # Compare population diversity over time
    plt.plot(binary_avg_gen_dist_list, label='Binary GA')
    plt.plot(character_avg_gen_dist_list, label='Character-Based GA')
    plt.title('Average Genetic Distance Comparison')
    plt.xlabel('Generation')
    plt.ylabel('Average Genetic Distance')
    plt.legend()
    plt.show()

    # Create a bar chart comparing the convergence speed of both GA algorithms
    labels = ['Binary GA', 'Character-Based GA']
    times = [binary_ga_time, character_ga_time]
    plt.bar(labels, times)
    plt.ylabel('Time (seconds)')
    plt.title('Convergence Speed Comparison')
    plt.show()


# def compare_nishing(results):
#     summary_stats = []
#
#     for algorithm, best_solutions in results.items():
#         fitnesses = [fitness for _, fitness in best_solutions]
#         avg_fitness = sum(fitnesses) / len(fitnesses)
#         min_fitness = min(fitnesses)
#         max_fitness = max(fitnesses)
#         summary_stats.append((algorithm, avg_fitness, min_fitness, max_fitness))
#
#     columns = ['Algorithm', 'Average Fitness', 'Min Fitness', 'Max Fitness']
#     df_summary_stats = pd.DataFrame(summary_stats, columns=columns)
#     print(df_summary_stats)
#
#     fig, ax = plt.subplots()
#
#     for algorithm, best_solutions in results.items():
#         fitnesses = [fitness for _, fitness in best_solutions]
#         ax.plot(fitnesses, label=algorithm)
#
#     ax.set_xlabel('Run')
#     ax.set_ylabel('Best Fitness')
#     ax.set_title('Best Fitness per Run for Each Algorithm')
#     ax.legend()
#
#     plt.show()
#
#     avg_genetic_distances = []
#
#     for algorithm, best_solutions in results.items():
#         individuals = [individual for individual, _ in best_solutions]
#         avg_gen_dist = sum(kendall_tau_distance(ind1, ind2) for ind1 in individuals for ind2 in individuals if ind1 != ind2) / ((len(individuals) * (len(individuals) - 1)) / 2)
#         avg_genetic_distances.append((algorithm, avg_gen_dist))
#
#     df_avg_genetic_distances = pd.DataFrame(avg_genetic_distances, columns=['Algorithm', 'Average Genetic Distance'])
#
#     print(df_avg_genetic_distances)
#     fig, ax = plt.subplots()
#
#     algorithms = [row[0] for row in avg_genetic_distances]
#     avg_gen_dists = [row[1] for row in avg_genetic_distances]
#
#     ax.bar(algorithms, avg_gen_dists)
#
#     ax.set_xlabel('Algorithm')
#     ax.set_ylabel('Average Genetic Distance')
#     ax.set_title('Average Genetic Distance for Each Algorithm')
#
#     plt.show()


def compare_nishing(results):
    summary_stats = []

    for algorithm, best_solution in results.items():
        if len(best_solution) == 1:
            fitness = best_solution[0][1]
        else:
            fitness = best_solution[1]
        summary_stats.append((algorithm, fitness))

    columns = ['Algorithm', 'Fitness']
    df_summary_stats = pd.DataFrame(summary_stats, columns=columns)
    print(df_summary_stats)

    fig, ax = plt.subplots()

    for algorithm, best_solution in results.items():
        if len(best_solution) == 1:
            fitness = best_solution[0][1]
        else:
            fitness = best_solution[1]
        ax.bar(algorithm, fitness, label=algorithm)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness of Best Solution for Each Algorithm')
    ax.legend()

    plt.show()
