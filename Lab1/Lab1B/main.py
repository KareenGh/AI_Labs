import random
import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt

from Genetic_Algorithms import genetic_algorithm, genetic_algorithm_with_age, genetic_algorithm_NQueens, \
    genetic_algorithm_BinPacking
from LoadFiles import read_binpack_input_file
from fitness import fitness_HitStamp, n_queens_fitness, bin_packing_fitness
from plotting import compare_binpack_algorithms
from sensitivity_analysis import sensitivity_analysis_NQueens
from utils import first_fit

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = int(GA_MUTATIONRATE * 100)
GA_TARGET = "Hello, world!"


def run_ga():
    crossover = crossover_var.get()
    selection = selection_var.get()
    k = k_var.get()
    algorithm = algorithm_var.get()

    # Run the selected genetic algorithm with the selected options
    if algorithm == "GA":
        best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness_HitStamp,
                                                          max_generations=100, crossover_type=crossover,
                                                          selection_method=selection, K=k)
    elif algorithm == "GA_WITH_AGE":
        age_alpha = age_alpha_var.get()
        best_individual, best_fitness = genetic_algorithm_with_age(pop_size=100, num_genes=13, fitness_func=fitness_HitStamp,
                                                                   max_generations=100, crossover_type=crossover,
                                                                   selection_method=selection, K=k, age_alpha=age_alpha)
    elif algorithm == "GA_NQueens":
        crossover_type1 = crossover_type_var.get()
        N = N_var.get()
        best_individual, best_fitness = genetic_algorithm_NQueens(pop_size=100, num_genes=N,
                                                                  fitness_func=n_queens_fitness,
                                                                  max_generations=100, crossover_type=crossover,
                                                                  selection_method=selection, K=k,
                                                                  exchange_operator=crossover_type1)
    elif algorithm == "GA_BinPacking":
        bin_capacity = bin_capacity_var.get()
        num_items  = num_items_var.get()
        # item_sizes = item_sizes_var.get()

        # Generate the item sizes
        item_sizes = [random.randint(1, bin_capacity) for _ in range(num_items)]

        # Call the genetic algorithm function with the input variables
        best_individual, best_fitness = genetic_algorithm_BinPacking(pop_size=100, num_genes=num_items,
                                                                     fitness_func=bin_packing_fitness,
                                                                     max_generations=100,
                                                                     crossover_type=crossover,
                                                                     selection_method=selection, K=k,
                                                                     item_sizes=item_sizes,
                                                                     bin_capacity=bin_capacity)

    elif algorithm == "GA_BinPacking_vs_FirstFit":
        # with open("binpack1.txt", "r") as f:
        #     while True:
        #         # Read the first line to get bin capacity, number of items and number of bins in the current solution
        #         line = f.readline()
        #         if not line:
        #             # End of file
        #             break
        #         bin_capacity, num_items, best_known_solution = map(int, line.split())
        #
        #         # Initialize a list to hold the item sizes
        #         item_sizes = []
        #
        #         # Read the remaining lines to get the item sizes
        #         for i in range(num_items):
        #             item_sizes.append(int(f.readline()))
        #

        file_path = "binpack1.txt"
        compare_binpack_algorithms(file_path, pop_size=100, max_generations=100, crossover_type=crossover,
                                   selection_method=selection, k=k)

        # if algorithm == "GA_BinPacking_vs_FirstFit":
        #     for bin_capacity, num_items, item_sizes in read_binpack_input_file(file_path):
        #         # Call the genetic algorithm function with the input variables
        #         best_individual, best_fitness = genetic_algorithm_BinPacking(pop_size=100, num_genes=num_items,
        #                                                                      fitness_func=bin_packing_fitness,
        #                                                                      max_generations=100,
        #                                                                      crossover_type=crossover,
        #                                                                      selection_method=selection, K=k,
        #                                                                      item_sizes=item_sizes,
        #                                                                      bin_capacity=bin_capacity)
        #
        #         # Compare the results to the greedy First Fit algorithm
        #         ga_solution = bin_packing_fitness(best_individual, item_sizes, bin_capacity)
        #         first_fit_solution = first_fit(item_sizes, bin_capacity)
        #
        #         # Compare the results to the greedy First Fit algorithm
        #         print("Bin capacity:", bin_capacity)
        #         print("Item sizes:", item_sizes)
        #         print("Best individual found by GA:", best_individual)
        #         print("Fitness of best individual found by GA:", best_fitness)
        #         print("Solution found by GA:", ga_solution)
        #         print("Solution found by First Fit algorithm:", first_fit_solution)
        #         print("-" * 50)

    elif algorithm == "NQueens_analysis":
        sensitivity_analysis_NQueens(pop_sizes=[10, 20, 30], mutation_rates=[0.01, 0.05, 0.1],
                                     selection_methods=["RWS"],
                                     survival_strategies=["ELITISM", "AGING"], exchange_operators=["CX", "PMX"])
        return

        # pop_sizes = [50, 100, 150, 200]
        # best_fitnesses = []
        # generation_times = []
        #
        # for pop_size in pop_sizes:
        #     best_individual, best_fitness, times = genetic_algorithm_NQueens_analysis(pop_size=pop_size, num_genes=8,
        #                                                                      fitness_func=n_queens_fitness,
        #                                                                      max_generations=100,
        #                                                                      crossover_type="CX",
        #                                                                      selection_method="RWS",
        #                                                                      exchange_operator="CX",
        #                                                                      mutation_rate=0.01,
        #                                                                      elitism_rate=0.1,
        #                                                                      aging_rate=0.1)
        #
        #     best_fitnesses.append(best_fitness)
        #     generation_times.append(times)
        #
        # plt.plot(pop_sizes, best_fitnesses, label="Best Fitness")
        # plt.plot(pop_sizes, generation_times, label="Generation Time")
        # plt.legend()
        # plt.xlabel("Population Size")
        # plt.ylabel("Performance")
        # plt.title("Performance of Genetic Algorithm for N-Queens Problem for Different Population Sizes")
        # plt.show()

        # pop_size = 100
        # num_genes = 8
        # max_generations = 100
        # crossover_type = "CX"
        # selection_method = "RWS"
        # k = None
        # exchange_operator = "CX"
        # mutation_rate = 0.01
        # elitism_rate = 0.1
        # aging_rate = 0.1
        #
        # best_individual, best_fitness = genetic_algorithm_NQueens_analysis(pop_size, num_genes, n_queens_fitness,
        #                                                           max_generations,
        #                                                           crossover_type, selection_method, k,
        #                                                           exchange_operator,
        #                                                           mutation_rate, elitism_rate, aging_rate)
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

    result_label.config(text=f"Best individual: {''.join(map(str, best_individual)) if isinstance(best_individual[0], str) else best_individual}\nBest fitness: {best_fitness}")
    print(
        f"Best individual: {''.join(map(str, best_individual)) if isinstance(best_individual[0], str) else best_individual}\nBest fitness: {best_fitness}")


# Create the GUI window
window = tk.Tk()
window.title("Genetic Algorithm")

# Create the option widgets
crossover_var = tk.StringVar(value='SINGLE')
crossover_label = ttk.Label(window, text="Crossover type:")
crossover_menu = ttk.OptionMenu(window, crossover_var, 'SINGLE', 'SINGLE', 'TWO', 'UNIFORM')

selection_var = tk.StringVar(value='RWS')
selection_label = ttk.Label(window, text="Selection method:")
selection_menu = ttk.OptionMenu(window, selection_var, 'RWS', 'RWS', 'SUS', 'RANKING_TOURNAMENT')

k_var = tk.IntVar(value=5)
k_label = ttk.Label(window, text="K value:")
k_entry = ttk.Entry(window, textvariable=k_var)

algorithm_var = tk.StringVar(value='GA')
algorithm_label = ttk.Label(window, text="Algorithm type:")
algorithm_menu = ttk.OptionMenu(window, algorithm_var, 'GA', 'GA', 'GA_WITH_AGE', 'GA_NQueens', 'GA_BinPacking',
                                'GA_BinPacking_vs_FirstFit', 'NQueens_analysis')

age_alpha_var = tk.DoubleVar(value=0.5)
age_alpha_label = ttk.Label(window, text="Age alpha:")
age_alpha_entry = ttk.Entry(window, textvariable=age_alpha_var)

crossover_type_var = tk.StringVar(value='CX')
crossover_type_label = ttk.Label(window, text="Exchange operators:")
crossover_type_menu = ttk.OptionMenu(window, crossover_type_var, 'CX', 'CX', 'PMX')

N_var = tk.IntVar(value=8)
N_label = ttk.Label(window, text="Number of Queens")
N_menu = ttk.Entry(window, textvariable=N_var)

# item_sizes_var = tk.IntVar()
# item_sizes_label = ttk.Label(window, text="Item sizes:")
# item_sizes_entry = ttk.Entry(window, textvariable=item_sizes_var)

num_items_var = tk.IntVar()
num_items_label = ttk.Label(window, text="Items number:")
num_items_entry = ttk.Entry(window, textvariable=num_items_var)

bin_capacity_var = tk.IntVar()
bin_capacity_label = ttk.Label(window, text="Bin capacity:")
bin_capacity_entry = ttk.Entry(window, textvariable=bin_capacity_var)

run_button = ttk.Button(window, text="Run", command=run_ga)

result_label = ttk.Label(window, text="")

algorithm_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
algorithm_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

crossover_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
crossover_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

selection_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
selection_menu.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

k_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
k_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

age_alpha_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
age_alpha_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

crossover_type_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
crossover_type_menu.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

N_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
N_menu.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

# item_sizes_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
# item_sizes_entry.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

num_items_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
num_items_entry.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

bin_capacity_label.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
bin_capacity_entry.grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)

run_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

result_label.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

window.mainloop()



#
# # Prompt the user for their choice
# choice = input("Enter 1 to run with SINGLE crossover, 2 for TWO crossover, 3 for UNIFORM crossover: ")
#
# # Set the crossover type based on the user's choice
# if choice == "1":
#     crossover_type = "SINGLE"
# elif choice == "2":
#     crossover_type = "TWO"
# elif choice == "3":
#     crossover_type = "UNIFORM"
# else:
#     print("Invalid choice.")
#     exit()
#
# # Prompt the user for their choice
# choice = input("Enter 1 for RWS selection method, 2 for SUS selection method, 3 for ranking and deterministic tournament selection with parameter K: ")
#
# # Set the selection method based on the user's choice
# if choice == "1":
#     selection_method = "RWS"
#     K = None
# elif choice == "2":
#     selection_method = "SUS"
#     K = None
# elif choice == "3":
#     selection_method = "RANKING_TOURNAMENT"
#     K = int(input("Enter K value: "))
# else:
#     print("Invalid choice.")
#     exit()
#
# # Run the genetic algorithm with the user's choices
# best_individual, best_fitness = genetic_algorithm(pop_size=GA_POPSIZE, num_genes=len(GA_TARGET), fitness_func=fitness_scrambler,
#                                                   max_generations=GA_MAXITER, crossover_type=crossover_type,
#                                                   selection_method=selection_method, K=K)
# print("Best individual:", ''.join(best_individual))
# print("Best fitness:", best_fitness)


# Run the genetic algorithm using UNIFORM operator and RWS selection method

# best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness_scrambler,
#                                                   max_generations=100, crossover_type="UNIFORM",
#                                                   selection_method="RANKING_TOURNAMENT", K=5)
# print("Best individual:", ''.join(best_individual))
# print("Best fitness:", best_fitness)
