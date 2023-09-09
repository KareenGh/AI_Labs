import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt

from Genetic_Algorithms import genetic_algorithm, genetic_algorithm_crossover, genetic_algorithm_original_vs_HitStamp, \
    genetic_algorithm_without_mutations, genetic_algorithm_mutation_only
from fitness import fitness, fitness_HitStamp

GA_TARGET = "Hello, world!"


# Wrap the genetic_algorithm function in a function that takes inputs from the GUI
def run_genetic_algorithm():
    algorithm = algorithm_var.get()
    crossover = crossover_var.get()

    if algorithm == "GeneticAlgorithm":
        # Run the genetic algorithm
        best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness,
                                                          max_generations=100)
    elif algorithm == "GeneticAlgorithmCrossover":
        best_individual, best_fitness = genetic_algorithm_crossover(pop_size=100, num_genes=13, fitness_func=fitness,
                                                                    max_generations=100, crossover_type=crossover,
                                                                    target_solution="Hello, world!")
    elif algorithm == "GA_HitStamp":
        best_individual, best_fitness = genetic_algorithm_crossover(pop_size=100, num_genes=13,
                                                                    fitness_func=fitness_HitStamp,
                                                                    max_generations=100, crossover_type=crossover,
                                                                    target_solution="Hello, world!")
    elif algorithm == "Original_vs_HitStamp":
        best_individual, best_fitness = genetic_algorithm_original_vs_HitStamp(pop_size=100, num_genes=13,
                                                                               fitness_func=fitness,
                                                                               max_generations=100,
                                                                               crossover_type=crossover,
                                                                               target_solution="Hello, world!")
        result_text.insert(tk.END, f"Best Individual: {''.join(best_individual)}\nBest Fitness: {best_fitness}\n")
        print("Original")
        print("Best individual:", ''.join(best_individual))
        print("Best fitness:", best_fitness)

        best_individual, best_fitness = genetic_algorithm_original_vs_HitStamp(pop_size=100, num_genes=13,
                                                                               fitness_func=fitness_HitStamp,
                                                                               max_generations=100,
                                                                               crossover_type=crossover,
                                                                               target_solution="Hello, world!")
        result_text.insert(tk.END, f"Best Individual: {''.join(best_individual)}\nBest Fitness: {best_fitness}\n")
        print("Hit Stamp")
        print("Best individual:", ''.join(best_individual))
        print("Best fitness:", best_fitness)
        plt.show()
        return

    elif algorithm == "replacement_without_mutations":
        best_individual, best_fitness = genetic_algorithm_without_mutations(pop_size=100, num_genes=13,
                                                                            fitness_func=fitness_HitStamp,
                                                                            max_generations=100,
                                                                            crossover_type=crossover,
                                                                            target_solution="Hello, world!")
    elif algorithm == "mutations_without_replacement":
        best_individual, best_fitness = genetic_algorithm_mutation_only(pop_size=100, num_genes=13,
                                                                        fitness_func=fitness_HitStamp,
                                                                        max_generations=100,
                                                                        target_solution="Hello, world!")
    elif algorithm == "mutations_and_replacement":
        best_individual, best_fitness = genetic_algorithm_crossover(pop_size=100, num_genes=13,
                                                                    fitness_func=fitness_HitStamp,
                                                                    max_generations=100, crossover_type=crossover,
                                                                    target_solution="Hello, world!")
    elif algorithm == "Original_vs_HitStamp_Simulation":
        GA_TARGET = "Genetic Algorithm"
        GA_ELITRATE = 0.1
        GA_MUTATION = 5
        POP_SIZE = 100
        NUM_GENES = len(GA_TARGET)
        MAX_GENERATIONS = 100

        crossover_types = ["SINGLE", "TWO", "UNIFORM"]
        fitness_functions = {
            "Original GA": fitness,
            "HitStamp Heuristic": fitness_HitStamp
        }

        results = {}

        for algo, fitness_func in fitness_functions.items():
            results[algo] = {}
            for crossover_type in crossover_types:
                start_time = time.time()
                best_individual, best_fitness = genetic_algorithm_crossover(pop_size=100, num_genes=13,
                                                                            fitness_func=fitness_HitStamp,
                                                                            max_generations=100, crossover_type=crossover,
                                                                            target_solution=GA_TARGET)
                end_time = time.time()
                time_elapsed = end_time - start_time
                results[algo][crossover_type] = {
                    "best_individual": best_individual,
                    "best_fitness": best_fitness,
                    "time_elapsed": time_elapsed
                }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for algo, data in results.items():
            fitnesses = [value["best_fitness"] for value in data.values()]
            times = [value["time_elapsed"] for value in data.values()]
            ax1.plot(crossover_types, fitnesses, marker='o', label=algo)
            ax2.plot(crossover_types, times, marker='o', label=algo)

        ax1.set_xlabel("Crossover Types")
        ax1.set_ylabel("Best Fitness")
        ax1.set_title("Comparison of Best Fitness")
        ax1.legend()

        ax2.set_xlabel("Crossover Types")
        ax2.set_ylabel("Time Elapsed (s)")
        ax2.set_title("Comparison of Time Elapsed")
        ax2.legend()

        plt.show()


    # Display the result in the text widget
    # result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Best Individual: {''.join(best_individual)}\nBest Fitness: {best_fitness}\n")
    print("Best individual:", ''.join(best_individual))
    print("Best fitness:", best_fitness)


# Create the main window
window = tk.Tk()
window.title("Genetic Algorithm GUI")

algorithm_var = tk.StringVar(value='GeneticAlgorithm')
algorithm_label = ttk.Label(window, text="Algorithm type:")
algorithm_menu = ttk.OptionMenu(window, algorithm_var, 'GeneticAlgorithm', 'GeneticAlgorithmCrossover', 'GA_HitStamp'
                                , 'Original_vs_HitStamp', 'replacement_without_mutations',
                                'mutations_without_replacement', 'mutations_and_replacement',
                                'Original_vs_HitStamp_Simulation')

# Create the option widgets
crossover_var = tk.StringVar(value='SINGLE')
crossover_label = ttk.Label(window, text="Crossover type:")
crossover_menu = ttk.OptionMenu(window, crossover_var, 'SINGLE', 'SINGLE', 'TWO', 'UNIFORM')

algorithm_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
algorithm_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

crossover_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
crossover_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

# Create the button to start the genetic algorithm
run_button = ttk.Button(window, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Create a text widget to display the results
result_text = tk.Text(window, wrap=tk.WORD, width=40, height=10)
result_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Start the application
window.mainloop()
