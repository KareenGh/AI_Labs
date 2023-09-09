import random
import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt

from Genetic_Algorithms import genetic_algorithm, genetic_algorithm_with_age, genetic_algorithm_NQueens, \
    genetic_algorithm_BinPacking, genetic_algorithm_Binary, GA_niching_BinPacking
from LoadFiles import read_binpack_input_file
from Metrics import binary_edit_distance_fitness, lcs_distance_fitness, jaccard_similarity_fitness, \
    hamming_distance_fitness
from exaptation import exaptation_main
from fitness import fitness_HitStamp, n_queens_fitness, bin_packing_fitness, binary_hit_stamp_fitness
from plotting import compare_binpack_algorithms, Compare_Mutations, compare_ga_results, compare_nishing
from sensitivity_analysis import sensitivity_analysis_NQueens

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = int(GA_MUTATIONRATE * 100)
GA_TARGET = "Hello, world!"
DIVERSITY_THRESHOLD = 40


def run_ga():
    crossover = crossover_var.get()
    selection = selection_var.get()
    k = k_var.get()
    algorithm = algorithm_var.get()

    # Run the selected genetic algorithm with the selected options

    # if algorithm == "GA":
    #     best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness_HitStamp,
    #                                                       max_generations=100, crossover_type=crossover,
    #                                                       selection_method=selection, K=k)
    # elif algorithm == "GA_WITH_AGE":
    #     age_alpha = age_alpha_var.get()
    #     best_individual, best_fitness, character_ga_time, character_ga_population \
    #         = genetic_algorithm_with_age(pop_size=100, num_genes=13, fitness_func=fitness_HitStamp, max_generations=100,
    #                                      crossover_type=crossover, selection_method=selection, K=k, age_alpha=age_alpha)
    # elif algorithm == "GA_NQueens":
    #     crossover_type1 = crossover_type_var.get()
    #     N = N_var.get()
    #     diversity_threshold = DIVERSITY_THRESHOLD
    #     NAPR_threshold = NAPR_threshold_var.get()
    #     best_individual, best_fitness = genetic_algorithm_NQueens(pop_size=100, num_genes=N,
    #                                                               fitness_func=n_queens_fitness,
    #                                                               max_generations=100, crossover_type=crossover,
    #                                                               selection_method=selection, K=k,
    #                                                               exchange_operator=crossover_type1,
    #                                                               DIVERSITY_THRESHOLD=diversity_threshold,
    #                                                               NAPR_THRESHOLD=NAPR_threshold)
    # elif algorithm == "GA_BinPacking":
    #     bin_capacity = bin_capacity_var.get()
    #     num_items = num_items_var.get()
    #     # item_sizes = item_sizes_var.get()
    #
    #     # Generate the item sizes
    #     item_sizes = [42, 98, 27, 21, 71, 71, 78, 76, 57, 24, 91, 84, 35, 25, 77, 96, 97, 89, 30, 86, 81, 39, 75, 66,
    #                   85, 36, 60, 56, 50, 75, 75, 37, 87, 95, 21, 99, 42, 57, 31, 37, 42, 40, 69, 91, 45, 97, 84, 90,
    #                   52, 43, 68, 53, 37, 65, 79, 73, 92, 87, 20, 20, 73, 42, 52, 20, 24, 76, 71, 72, 21, 21, 82, 92,
    #                   78, 87, 50, 41, 31, 73, 89, 59, 88, 40, 71, 69, 45, 57, 49, 68, 84, 32, 69, 77, 92, 98, 57, 39,
    #                   32, 23, 99, 91, 48, 21, 70, 43, 73, 69, 65, 57, 67, 28, 84, 42, 61, 92, 82, 34, 74, 55, 60, 69]
    #
    #     # [random.randint(1, bin_capacity) for _ in range(num_items)]
    #
    #     # Call the genetic algorithm function with the input variables
    #     best_individual, best_fitness = genetic_algorithm_BinPacking(pop_size=100, num_genes=num_items,
    #                                                                  fitness_func=bin_packing_fitness,
    #                                                                  max_generations=100,
    #                                                                  crossover_type=crossover,
    #                                                                  selection_method=selection, K=k,
    #                                                                  item_sizes=item_sizes,
    #                                                                  bin_capacity=bin_capacity)
    #
    # elif algorithm == "NQueens_analysis":
    #     sensitivity_analysis_NQueens(pop_sizes=[10, 20, 30], mutation_rates=[0.01, 0.05, 0.1],
    #                                  selection_methods=["RWS"],
    #                                  survival_strategies=["ELITISM", "AGING"], exchange_operators=["CX", "PMX"])
    #     return

    if algorithm == "GA_Binary":
        age_alpha = age_alpha_var.get()
        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()

        # Run the algorithm with a constant mutation rate
        best_individual, best_fitness, avg_gen_dist_constant, num_unique_alleles_constant, \
        generations_constant, binary_ga_time, binary_ga_population, avg_gen_dist_list, num_unique_alleles_list\
            = genetic_algorithm_Binary(pop_size=200, num_genes=13, fitness_func=binary_hit_stamp_fitness,
                                       max_generations=200, crossover_type=crossover, selection_method=selection, K=k,
                                       age_alpha=age_alpha, mutation_method=mutation, threshold=threshold,
                                       stagnation_gen=stagnation, decay_rate=decay_rate, constant_mutation_rate=True)
        best_individual = ''.join([chr(int(binary_str, 2)) for binary_str in best_individual])

    elif algorithm == "GA_Binary_vs_GA":
        age_alpha = age_alpha_var.get()
        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()

        best_individual_binary, best_fitness_binary, avg_gen_dist_constant, num_unique_alleles_constant, \
        generations_constant, binary_ga_time, binary_ga_population, avg_gen_dist_list, num_unique_alleles_list\
            = genetic_algorithm_Binary(pop_size=200, num_genes=13, fitness_func=binary_hit_stamp_fitness,
                                       max_generations=200, crossover_type=crossover, selection_method=selection, K=k,
                                       age_alpha=age_alpha, mutation_method=mutation, threshold=threshold,
                                       stagnation_gen=stagnation, decay_rate=decay_rate, constant_mutation_rate=True)
        best_individual_binary = ''.join([chr(int(binary_str, 2)) for binary_str in best_individual_binary])

        best_individual, best_fitness, character_ga_time, character_ga_population, avg_gen_dist_list2, \
        num_unique_alleles_list = genetic_algorithm_with_age(pop_size=200, num_genes=13, fitness_func=fitness_HitStamp,
                                                             max_generations=200, crossover_type=crossover,
                                                             selection_method=selection, K=k, age_alpha=age_alpha)

        compare_ga_results(best_individual_binary, best_individual, best_fitness_binary, best_fitness, binary_ga_time,
                           character_ga_time, binary_ga_population, character_ga_population, avg_gen_dist_list,
                           avg_gen_dist_list2)

    elif algorithm == "GA_NQueens_Mutation":
        crossover_type1 = crossover_type_var.get()
        N = N_var.get()
        diversity_threshold = DIVERSITY_THRESHOLD
        NAPR_threshold = NAPR_threshold_var.get()
        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()
        mutation_probability = mutation_probability_var.get()

        # Run the algorithm with an adaptive mutation rate
        best_individual_adaptive, best_fitness_adaptive, avg_gen_dist_adaptive, num_unique_alleles_adaptive, \
        generations_adaptive, avg_gen_dist_list, num_unique_alleles_list \
            = genetic_algorithm_NQueens(pop_size=100, num_genes=N,
                                        fitness_func=n_queens_fitness,
                                        max_generations=100, crossover_type=crossover,
                                        selection_method=selection, K=k,
                                        exchange_operator=crossover_type1,
                                        DIVERSITY_THRESHOLD=diversity_threshold,
                                        NAPR_THRESHOLD=NAPR_threshold,
                                        mutation_method=mutation, mutation_prob=mutation_probability,
                                        threshold=threshold,
                                        stagnation_gen=stagnation, decay_rate=decay_rate,
                                        constant_mutation_rate=False)

        # Run the algorithm with a constant mutation rate
        best_individual_constant, best_fitness_constant, avg_gen_dist_constant, num_unique_alleles_constant, \
        generations_constant, avg_gen_dist_list2, num_unique_alleles_list2 = genetic_algorithm_NQueens(
            pop_size=100, num_genes=N,
            fitness_func=n_queens_fitness,
            max_generations=100, crossover_type=crossover,
            selection_method=selection, K=k,
            exchange_operator=crossover_type1,
            DIVERSITY_THRESHOLD=diversity_threshold,
            NAPR_THRESHOLD=NAPR_threshold,
            mutation_method=mutation, threshold=threshold,
            stagnation_gen=stagnation, decay_rate=decay_rate, constant_mutation_rate=True)

        best_individual = best_individual_adaptive
        best_fitness = best_fitness_adaptive

        Compare_Mutations("N-Queens", avg_gen_dist_adaptive, num_unique_alleles_adaptive,
                          avg_gen_dist_constant,
                          num_unique_alleles_constant, best_fitness_adaptive, generations_adaptive,
                          best_fitness_constant, generations_constant, best_individual_adaptive,
                          best_individual_constant, avg_gen_dist_list, num_unique_alleles_list, mutation)

        return

    elif algorithm == "GA_BinPacking_Mutation":
        bin_capacity = bin_capacity_var.get()
        num_items = num_items_var.get()
        # Generate the item sizes
        # item_sizes = [42, 98, 27, 21, 71, 71, 78, 76, 57, 24, 91, 84, 35, 25, 77, 96, 97, 89, 30, 86, 81, 39, 75, 66,
        #               85, 36, 60, 56, 50, 75, 75, 37, 87, 95, 21, 99, 42, 57, 31, 37, 42, 40, 69, 91, 45, 97, 84, 90,
        #               52, 43, 68, 53, 37, 65, 79, 73, 92, 87, 20, 20, 73, 42, 52, 20, 24, 76, 71, 72, 21, 21, 82, 92,
        #               78, 87, 50, 41, 31, 73, 89, 59, 88, 40, 71, 69, 45, 57, 49, 68, 84, 32, 69, 77, 92, 98, 57, 39,
        #               32, 23, 99, 91, 48, 21, 70, 43, 73, 69, 65, 57, 67, 28, 84, 42, 61, 92, 82, 34, 74, 55, 60, 69]
        item_sizes = [random.randint(1, bin_capacity) for _ in range(num_items)]

        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()
        mutation_probability = mutation_probability_var.get()

        # Run the algorithm with an adaptive mutation rate
        best_individual_adaptive, best_fitness_adaptive, avg_gen_dist_adaptive, num_unique_alleles_adaptive, \
        generations_adaptive, avg_gen_dist_list, num_unique_alleles_list \
            = genetic_algorithm_BinPacking(pop_size=200, num_genes=num_items,
                                           fitness_func=bin_packing_fitness, max_generations=200,
                                           crossover_type=crossover, selection_method=selection, K=k,
                                           item_sizes=item_sizes, bin_capacity=bin_capacity,
                                           mutation_method=mutation,
                                           mutation_prob=mutation_probability,
                                           threshold=threshold,
                                           stagnation_gen=stagnation, decay_rate=decay_rate,
                                           constant_mutation_rate=False)
        # Run the algorithm with a constant mutation rate
        best_individual_constant, best_fitness_constant, avg_gen_dist_constant, num_unique_alleles_constant, \
        generations_constant, avg_gen_dist_list2, num_unique_alleles_list2 \
            = genetic_algorithm_BinPacking(pop_size=200, num_genes=num_items,
                                           fitness_func=bin_packing_fitness, max_generations=200,
                                           crossover_type=crossover, selection_method=selection, K=k,
                                           item_sizes=item_sizes, bin_capacity=bin_capacity,
                                           mutation_method=mutation, mutation_prob=mutation_probability
                                           , threshold=threshold, stagnation_gen=stagnation,
                                           decay_rate=decay_rate, constant_mutation_rate=True)

        best_individual = best_individual_adaptive
        best_fitness = best_fitness_adaptive

        # Compare_Mutations("Bin-Packing", avg_gen_dist_adaptive, num_unique_alleles_adaptive, avg_gen_dist_constant,
        #                   num_unique_alleles_constant, best_fitness_adaptive, generations_adaptive,
        #                   best_fitness_constant, generations_constant, best_individual_adaptive,
        #                   best_individual_constant, avg_gen_dist_adaptive, num_unique_alleles_adaptive)
        Compare_Mutations("Bin-Packing", avg_gen_dist_adaptive, num_unique_alleles_adaptive,
                          avg_gen_dist_constant, num_unique_alleles_constant, best_fitness_adaptive,
                          generations_adaptive, best_fitness_constant, generations_constant, best_individual_adaptive,
                          best_individual_constant, avg_gen_dist_list, num_unique_alleles_list, mutation)

        return

    elif algorithm == "GA_String_Mutation":
        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()
        age_alpha = age_alpha_var.get()
        mutation_probability = mutation_probability_var.get()

        # Run the algorithm with an adaptive mutation rate
        best_individual_adaptive, best_fitness_adaptive, avg_gen_dist_adaptive, num_unique_alleles_adaptive, \
        generations_adaptive, binary_ga_times, population, avg_gen_dist_list, num_unique_alleles_list \
            = genetic_algorithm_Binary(pop_size=200, num_genes=13, fitness_func=binary_edit_distance_fitness,
                                       max_generations=200,
                                       crossover_type=crossover, selection_method=selection, K=k, age_alpha=age_alpha,
                                       mutation_method=mutation,
                                       mutation_prob=mutation_probability, threshold=threshold,
                                       stagnation_gen=stagnation, decay_rate=decay_rate,
                                       constant_mutation_rate=False)

        # Run the algorithm with a constant mutation rate
        best_individual_constant, best_fitness_constant, avg_gen_dist_constant, num_unique_alleles_constant, \
        generations_constant, binary_ga_times, population, avg_gen_dist_list2, num_unique_alleles_list2 \
            = genetic_algorithm_Binary(pop_size=200, num_genes=13, fitness_func=binary_edit_distance_fitness,
                                       max_generations=200,
                                       crossover_type=crossover, selection_method=selection, K=k, age_alpha=age_alpha,
                                       mutation_method=mutation,
                                       mutation_prob=mutation_probability, threshold=threshold,
                                       stagnation_gen=stagnation, decay_rate=decay_rate,
                                       constant_mutation_rate=True)

        best_individual_constant = ''.join([chr(int(binary_str, 2)) for binary_str in best_individual_constant])
        best_individual_adaptive = ''.join([chr(int(binary_str, 2)) for binary_str in best_individual_adaptive])
        best_individual = best_individual_adaptive
        best_fitness = best_fitness_adaptive

        Compare_Mutations("String Matching", avg_gen_dist_adaptive, num_unique_alleles_adaptive,
                          avg_gen_dist_constant, num_unique_alleles_constant, best_fitness_adaptive,
                          generations_adaptive, best_fitness_constant, generations_constant, best_individual_adaptive,
                          best_individual_constant, avg_gen_dist_list, num_unique_alleles_list, mutation)

    elif algorithm == "BinPacking_niching":
        bin_capacity = bin_capacity_var.get()
        num_items = num_items_var.get()
        item_sizes = [random.randint(1, bin_capacity) for _ in range(num_items)]

        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()
        mutation_probability = mutation_probability_var.get()
        niching_method = niching_method_var.get()
        num_clusters = num_clusters_var.get()
        sharing_radius = sharing_radius_var.get()
        alpha = alpha_var.get()
        crowding_factor = crowding_factor_var.get()

        best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
            = GA_niching_BinPacking(pop_size=100, num_genes=num_items, fitness_func=bin_packing_fitness,
                                    max_generations=100, crossover_type=crossover, selection_method=selection,
                                    K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
                                    mutation_method=mutation, mutation_prob=mutation_probability, threshold=threshold,
                                    stagnation_gen=stagnation, decay_rate=decay_rate, constant_mutation_rate=False,
                                    niche_method=niching_method, sigma_share=sharing_radius, alpha=alpha,
                                    crowding_factor=crowding_factor, num_species=num_clusters,
                                    clustering_algorithm='k-means')

        print(f"Method: {niching_method}")
        print(f"Best fitness: {best_fitness}")
        print(f"Average genetic distance: {avg_gen_dist}")
        print(f"Number of unique alleles: {num_unique_alleles}")
        print("\n")
        return

    elif algorithm == "Compare_niching":
        bin_capacity = bin_capacity_var.get()
        num_items = num_items_var.get()
        item_sizes = [random.randint(1, bin_capacity) for _ in range(num_items)]

        mutation = mutation_method_var.get()
        threshold = diversity_threshold_var.get()
        stagnation = stagnation_gen_var.get()
        decay_rate = decay_rate_var.get()
        mutation_probability = mutation_probability_var.get()
        num_clusters = num_clusters_var.get()
        sharing_radius = sharing_radius_var.get()
        alpha = alpha_var.get()
        crowding_factor = crowding_factor_var.get()

        niche_algorithms = ['fitness_sharing', 'crowding', 'speciation_with_clustering']
        results = {}

        for algorithm in niche_algorithms:
            print(f"Running GA with {algorithm} algorithm")
            best_solutions = []

            if algorithm == 'fitness_sharing':
                best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
                    = GA_niching_BinPacking(pop_size=100, num_genes=num_items, fitness_func=bin_packing_fitness,
                                            max_generations=100, crossover_type=crossover,
                                            selection_method=selection,
                                            K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
                                            mutation_method='ADAPTIVE', mutation_prob=mutation_probability,
                                            threshold=threshold,
                                            stagnation_gen=stagnation, decay_rate=decay_rate,
                                            constant_mutation_rate=False,
                                            niche_method='fitness_sharing', sigma_share=sharing_radius, alpha=alpha,
                                            crowding_factor=crowding_factor, num_species=num_clusters,
                                            clustering_algorithm='k-means')

            elif algorithm == 'crowding':
                best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
                    = GA_niching_BinPacking(pop_size=200, num_genes=num_items, fitness_func=bin_packing_fitness,
                                            max_generations=200, crossover_type=crossover,
                                            selection_method=selection,
                                            K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
                                            mutation_method='THM', mutation_prob=mutation_probability,
                                            threshold=threshold,
                                            stagnation_gen=stagnation, decay_rate=decay_rate,
                                            constant_mutation_rate=False,
                                            niche_method='crowding', sigma_share=sharing_radius, alpha=alpha,
                                            crowding_factor=crowding_factor, num_species=num_clusters,
                                            clustering_algorithm='k-means')

            elif algorithm == 'speciation_with_clustering':
                best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
                    = GA_niching_BinPacking(pop_size=100, num_genes=num_items, fitness_func=bin_packing_fitness,
                                            max_generations=100, crossover_type=crossover,
                                            selection_method=selection,
                                            K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
                                            mutation_method='THM', mutation_prob=mutation_probability,
                                            threshold=threshold,
                                            stagnation_gen=stagnation, decay_rate=decay_rate,
                                            constant_mutation_rate=False,
                                            niche_method='speciation_with_clustering', sigma_share=sharing_radius,
                                            alpha=alpha,
                                            crowding_factor=crowding_factor, num_species=num_clusters,
                                            clustering_algorithm='k-means')

            best_solutions.append((best_individual, best_fitness))

            results[algorithm] = best_solutions

        # Compare the results and analyze the performance of each algorithm
        compare_nishing(results)
        return

    elif algorithm == "Exaptation":
        exaptation_main()
        return

    # elif algorithm == "Compare_niching":
    #     bin_capacity = bin_capacity_var.get()
    #     num_items = num_items_var.get()
    #     item_sizes = [random.randint(1, bin_capacity) for _ in range(num_items)]
    #
    #     mutation = mutation_method_var.get()
    #     threshold = diversity_threshold_var.get()
    #     stagnation = stagnation_gen_var.get()
    #     decay_rate = decay_rate_var.get()
    #     mutation_probability = mutation_probability_var.get()
    #     num_clusters = num_clusters_var.get()
    #     sharing_radius = sharing_radius_var.get()
    #     alpha = alpha_var.get()
    #     crowding_factor = crowding_factor_var.get()
    #
    #     niche_algorithms = ['fitness_sharing', 'crowding', 'speciation_with_clustering']
    #     results = {}
    #
    #     for algorithm in niche_algorithms:
    #         print(f"Running GA with {algorithm} algorithm")
    #         best_solutions = []
    #         num_runs = 1
    #         for _ in range(num_runs):
    #             if algorithm == 'fitness_sharing':
    #                 best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
    #                     = GA_niching_BinPacking(pop_size=100, num_genes=num_items, fitness_func=bin_packing_fitness,
    #                                             max_generations=100, crossover_type=crossover,
    #                                             selection_method=selection,
    #                                             K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
    #                                             mutation_method='ADAPTIVE', mutation_prob=mutation_probability,
    #                                             threshold=threshold,
    #                                             stagnation_gen=stagnation, decay_rate=decay_rate,
    #                                             constant_mutation_rate=False,
    #                                             niche_method='fitness_sharing', sigma_share=sharing_radius, alpha=alpha,
    #                                             crowding_factor=crowding_factor, num_species=num_clusters,
    #                                             clustering_algorithm='k-means')
    #
    #             elif algorithm == 'crowding':
    #                 best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
    #                     = GA_niching_BinPacking(pop_size=200, num_genes=num_items, fitness_func=bin_packing_fitness,
    #                                             max_generations=200, crossover_type=crossover,
    #                                             selection_method=selection,
    #                                             K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
    #                                             mutation_method='THM', mutation_prob=mutation_probability,
    #                                             threshold=threshold,
    #                                             stagnation_gen=stagnation, decay_rate=decay_rate,
    #                                             constant_mutation_rate=False,
    #                                             niche_method='crowding', sigma_share=sharing_radius, alpha=alpha,
    #                                             crowding_factor=crowding_factor, num_species=num_clusters,
    #                                             clustering_algorithm='k-means')
    #
    #             elif algorithm == 'speciation_with_clustering':
    #                 best_individual, best_fitness, avg_gen_dist, num_unique_alleles, generations_adaptive \
    #                     = GA_niching_BinPacking(pop_size=100, num_genes=num_items, fitness_func=bin_packing_fitness,
    #                                             max_generations=100, crossover_type=crossover,
    #                                             selection_method=selection,
    #                                             K=k, item_sizes=item_sizes, bin_capacity=bin_capacity,
    #                                             mutation_method='THM', mutation_prob=mutation_probability,
    #                                             threshold=threshold,
    #                                             stagnation_gen=stagnation, decay_rate=decay_rate,
    #                                             constant_mutation_rate=False,
    #                                             niche_method='speciation_with_clustering', sigma_share=sharing_radius, alpha=alpha,
    #                                             crowding_factor=crowding_factor, num_species=num_clusters,
    #                                             clustering_algorithm='k-means')
    #
    #             best_solutions.append((best_individual, best_fitness))
    #
    #         results[algorithm] = best_solutions
    #
    #     # Compare the results and analyze the performance of each algorithm
    #     compare_nishing(results)
    #     return

    # elif algorithm == "GA_Mutation":
    #     mutation = mutation_method_var.get()
    #     threshold = diversity_threshold_var.get()
    #     stagnation = stagnation_gen_var.get()
    #     decay_rate = decay_rate_var.get()
    #     age_alpha = age_alpha_var.get()
    #     mutation_probability = mutation_probability_var.get()
    #
    #     best_individual, best_fitness = genetic_algorithm_with_age(pop_size=100, num_genes=13,
    #                                                                fitness_func=fitness_HitStamp,
    #                                                                max_generations=100,
    #                                                                crossover_type=crossover,
    #                                                                selection_method=selection, K=k,
    #                                                                age_alpha=age_alpha,
    #                                                                mutation_method=mutation,
    #                                                                mutation_prob=mutation_probability,
    #                                                                threshold=threshold,
    #                                                                stagnation_gen=stagnation, decay_rate=decay_rate)

    else:
        raise ValueError("Invalid algorithm type selected.")

    # Create a new window to display the final results
    result_window = tk.Toplevel(window)
    result_window.title("Genetic Algorithm Results")

    # Display the final results in the new window
    result_label = ttk.Label(result_window, text=f"Best individual: {best_individual}"
                                                 f"\nBest fitness: {best_fitness}")
    result_label.pack(padx=50, pady=50)

    # Make sure the new window is displayed on top of the main window
    result_window.lift()

    # Set the focus back to the main window
    window.focus()

    result_label.config(
        text=f"Best individual: {''.join(map(str, best_individual)) if isinstance(best_individual[0], str) else best_individual}\nBest fitness: {best_fitness}")
    print(
        f"Best individual: {''.join(map(str, best_individual)) if isinstance(best_individual[0], str) else best_individual}\nBest fitness: {best_fitness}")


# Create the GUI window
window = tk.Tk()
window.title("Genetic Algorithm")

# Create the option widgets
algorithm_var = tk.StringVar(value='GA')
algorithm_label = ttk.Label(window, text="Algorithm type:")
algorithm_menu = ttk.OptionMenu(window, algorithm_var, 'GA_Binary', 'GA_Binary', 'GA_Binary_vs_GA',
                                'GA_NQueens_Mutation', 'GA_BinPacking_Mutation', 'GA_String_Mutation',
                                'BinPacking_niching', 'Compare_niching', 'Exaptation')

algorithm_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
algorithm_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

crossover_var = tk.StringVar(value='UNIFORM')
crossover_label = ttk.Label(window, text="Crossover type:")
crossover_menu = ttk.OptionMenu(window, crossover_var, 'UNIFORM', 'SINGLE', 'TWO', 'UNIFORM')

crossover_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
crossover_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

selection_var = tk.StringVar(value='RWS')
selection_label = ttk.Label(window, text="Selection method:")
selection_menu = ttk.OptionMenu(window, selection_var, 'RANKING_TOURNAMENT', 'RWS', 'SUS', 'RANKING_TOURNAMENT')

selection_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
selection_menu.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

k_var = tk.IntVar(value=5)
k_label = ttk.Label(window, text="K value:")
k_entry = ttk.Entry(window, textvariable=k_var)

k_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
k_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

age_alpha_var = tk.DoubleVar(value=0.5)
age_alpha_label = ttk.Label(window, text="Age alpha:")
age_alpha_entry = ttk.Entry(window, textvariable=age_alpha_var)

age_alpha_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
age_alpha_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

crossover_type_var = tk.StringVar(value='CX')
crossover_type_label = ttk.Label(window, text="Exchange operators:")
crossover_type_menu = ttk.OptionMenu(window, crossover_type_var, 'CX', 'CX', 'PMX')

crossover_type_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
crossover_type_menu.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

N_var = tk.IntVar(value=8)
N_label = ttk.Label(window, text="Number of Queens")
N_menu = ttk.Entry(window, textvariable=N_var)

N_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
N_menu.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

num_items_var = tk.IntVar(value=20)
num_items_label = ttk.Label(window, text="Items number:")
num_items_entry = ttk.Entry(window, textvariable=num_items_var)

num_items_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
num_items_entry.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

bin_capacity_var = tk.IntVar(value=25)
bin_capacity_label = ttk.Label(window, text="Bin capacity:")
bin_capacity_entry = ttk.Entry(window, textvariable=bin_capacity_var)

bin_capacity_label.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
bin_capacity_entry.grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)

# diversity_threshold_var = tk.DoubleVar(value=25)
# diversity_threshold_label = ttk.Label(window, text="Diversity threshold:")
# diversity_threshold_entry = ttk.Entry(window, textvariable=diversity_threshold_var)
#
# diversity_threshold_label.grid(row=9, column=0, padx=5, pady=5, sticky=tk.W)
# diversity_threshold_entry.grid(row=9, column=1, padx=5, pady=5, sticky=tk.W)

NAPR_threshold_var = tk.DoubleVar(value=0.95)
NAPR_threshold_label = ttk.Label(window, text="NAPR threshold:")
NAPR_threshold_entry = ttk.Entry(window, textvariable=NAPR_threshold_var)

NAPR_threshold_label.grid(row=9, column=0, padx=5, pady=5, sticky=tk.W)
NAPR_threshold_entry.grid(row=9, column=1, padx=5, pady=5, sticky=tk.W)

mutation_method_var = tk.StringVar(value='CONSTANT')
mutation_method_label = ttk.Label(window, text="Mutation method:")
mutation_method_menu = ttk.OptionMenu(window, mutation_method_var, 'CONSTANT', 'CONSTANT', 'NON_UNIFORM', 'ADAPTIVE',
                                      'THM', 'SELF_ADAPTIVE')

mutation_method_label.grid(row=10, column=0, padx=5, pady=5, sticky=tk.W)
mutation_method_menu.grid(row=10, column=1, padx=5, pady=5, sticky=tk.W)

mutation_probability_var = tk.DoubleVar(value=0.25)
mutation_probability_label = ttk.Label(window, text="Mutation probability:")
mutation_probability_entry = ttk.Entry(window, textvariable=mutation_probability_var)

mutation_probability_label.grid(row=11, column=0, padx=5, pady=5, sticky=tk.W)
mutation_probability_entry.grid(row=11, column=1, padx=5, pady=5, sticky=tk.W)

decay_rate_var = tk.DoubleVar(value=0.99)
decay_rate_label = ttk.Label(window, text="Decay rate:")
decay_rate_entry = ttk.Spinbox(window, from_=0, to=1, increment=0.01, textvariable=decay_rate_var)

decay_rate_label.grid(row=12, column=0, padx=5, pady=5, sticky=tk.W)
decay_rate_entry.grid(row=12, column=1, padx=5, pady=5, sticky=tk.W)

diversity_threshold_var = tk.DoubleVar(value=0.8)
diversity_threshold_label = ttk.Label(window, text="Diversity threshold:")
diversity_threshold_spinbox = ttk.Spinbox(window, from_=0, to=1, increment=0.1, textvariable=diversity_threshold_var)

diversity_threshold_label.grid(row=13, column=0, padx=5, pady=5, sticky=tk.W)
diversity_threshold_spinbox.grid(row=13, column=1, padx=5, pady=5, sticky=tk.W)

stagnation_gen_var = tk.IntVar(value=5)
stagnation_gen_label = ttk.Label(window, text="Stagnation generations:")
stagnation_gen_entry = ttk.Spinbox(window, from_=0, to=100, increment=0.5, textvariable=stagnation_gen_var)

stagnation_gen_label.grid(row=14, column=0, padx=5, pady=5, sticky=tk.W)
stagnation_gen_entry.grid(row=14, column=1, padx=5, pady=5, sticky=tk.W)

niching_method_var = tk.StringVar(value='FITNESS_SHARING')
niching_method_label = ttk.Label(window, text="Niching method:")
niching_method_menu = ttk.OptionMenu(window, niching_method_var, 'FITNESS_SHARING', 'FITNESS_SHARING', 'CROWDING',
                                     'SPECIATION')

niching_method_label.grid(row=15, column=0, padx=5, pady=5, sticky=tk.W)
niching_method_menu.grid(row=15, column=1, padx=5, pady=5, sticky=tk.W)

# fitness sharing niching method
sharing_radius_var = tk.DoubleVar(value=1)
sharing_radius_label = ttk.Label(window, text="Sharing radius:")
sharing_radius_entry = ttk.Entry(window, textvariable=sharing_radius_var)

sharing_radius_label.grid(row=16, column=0, padx=5, pady=5, sticky=tk.W)
sharing_radius_entry.grid(row=16, column=1, padx=5, pady=5, sticky=tk.W)

alpha_var = tk.DoubleVar(value=1)
alpha_label = ttk.Label(window, text="Alpha:")
alpha_entry = ttk.Entry(window, textvariable=alpha_var)

alpha_label.grid(row=17, column=0, padx=5, pady=5, sticky=tk.W)
alpha_entry.grid(row=17, column=1, padx=5, pady=5, sticky=tk.W)

# crowding niching method
crowding_factor_var = tk.IntVar(value=1)
crowding_factor_label = ttk.Label(window, text="Crowding factor:")
crowding_factor_entry = ttk.Entry(window, textvariable=crowding_factor_var)

crowding_factor_label.grid(row=18, column=0, padx=5, pady=5, sticky=tk.W)
crowding_factor_entry.grid(row=18, column=1, padx=5, pady=5, sticky=tk.W)

# speciation niching method
num_clusters_var = tk.IntVar(value=10)
num_clusters_label = ttk.Label(window, text="Number of clusters:")
num_clusters_entry = ttk.Entry(window, textvariable=num_clusters_var)

num_clusters_label.grid(row=19, column=0, padx=5, pady=5, sticky=tk.W)
num_clusters_entry.grid(row=19, column=1, padx=5, pady=5, sticky=tk.W)

run_button = ttk.Button(window, text="Run", command=run_ga)

result_label = ttk.Label(window, text="")

run_button.grid(row=20, column=0, columnspan=2, padx=5, pady=5)

result_label.grid(row=21, column=0, columnspan=2, padx=5, pady=5)

window.mainloop()
