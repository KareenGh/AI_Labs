# Define a function to calculate and print the statistics for each generation
import math
import statistics
import time

import matplotlib.pyplot as plt

from fitness import fitness, fitness_HitStamp


def print_generation_stats(generation, population, fitness_func, start_time, start_clock):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_func(individual) for individual in population]
    AvgFitness = statistics.mean(fitnesses)
    StdFitness = statistics.stdev(fitnesses, AvgFitness)

    # Calculate the running time for the current generation
    elapsed_time = time.time() - start_time
    clock_ticks = time.process_time() - start_clock

    # Print the statistics for the current generation
    print(f"Generation", generation + 1)
    print("Individual:", ''.join(population[generation]))
    print(f"    Average fitness =", AvgFitness)
    print(f"    Standard deviation = {StdFitness:.4f}")
    print(f"    Elapsed time = {elapsed_time:.4f}")
    print(f"    Clock ticks =", clock_ticks)
    print("-" * 50)


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


def plot_original_vs_HitStamp(population, fitness_func):
    if fitness_func == fitness:
        # Plot the fitness histogram for the original fitness function
        plt.figure()
        OrigFitnesses = [fitness(individual) for individual in population]
        Normalfitnesses = [f / max(OrigFitnesses) for f in OrigFitnesses]
        # Scale the normalized fitness scores to a range of 0-100
        fitnesses_scaled = [f * 100 for f in Normalfitnesses]
        plt.hist(fitnesses_scaled, bins='auto', alpha=0.5, label='Original GA')
        plt.title("Fitness Distribution (Original Fitness Function)")
        plt.xlabel("Fitness")
        plt.ylabel("Number of Individuals")
        plt.legend(loc='upper right')
    elif fitness_func == fitness_HitStamp:
        # Plot the fitness histogram for the scrambler heuristic
        plt.figure()
        fitnesses_scrambler = [fitness_HitStamp(individual) for individual in population]
        Normalfitnesses = [f / max(fitnesses_scrambler) for f in fitnesses_scrambler]
        # Scale the normalized fitness scores to a range of 0-100
        fitnesses_scaled = [f * 100 for f in Normalfitnesses]
        plt.hist(fitnesses_scaled, bins='auto', alpha=0.5, color='orange', label='HitStamp Heuristic')
        plt.title("Fitness Distribution (Hit Stamp  Heuristic)")
        plt.xlabel("Fitness")
        plt.ylabel("Number of Individuals")
        plt.legend(loc='upper right')


