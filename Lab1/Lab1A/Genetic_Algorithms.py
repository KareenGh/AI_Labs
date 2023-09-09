# Define the genetic algorithm
import random
import time

from matplotlib import pyplot as plt

from crossover import crossover
from fitness import fitness, fitness_HitStamp

from plotting import print_generation_stats, plot_fitness_histogram, plot_original_vs_HitStamp

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = int(GA_MUTATIONRATE * 100)
GA_TARGET = "Hello, world!"


# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations):
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

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes)]
            for i in range(num_genes):
                if random.randint(0, 100) < GA_MUTATION:
                    child[i] = chr(random.randint(32, 126))
            offspring.append(child)

        population = elites + offspring

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock)

        # Plot the fitness histogram every 10 generations
        if (generation + 1) % 10 == 0:
            plot_fitness_histogram(generation, population, fitness_func)

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


# Define the genetic algorithm
def genetic_algorithm_crossover(pop_size, num_genes, fitness_func, max_generations, crossover_type, target_solution):
    # Initialize the population with random individuals
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    # Evolve the population for a fixed number of generations or until the target solution is found
    start_time = time.time()
    start_clock = time.process_time()
    for generation in range(max_generations):
        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child1, child2 = crossover(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < GA_MUTATION:
                        child[i] = chr(random.randint(32, 126))
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Check if the target solution is found
        if ''.join(population[generation]) == target_solution:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_fitness_histogram(generation, population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock)

        # # Plot the fitness histogram every 10 generations
        # if (generation + 1) % 10 == 0:
        #     plot_fitness_histogram(generation, population, fitness_func)

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


# Define the genetic algorithm
def genetic_algorithm_original_vs_HitStamp(pop_size, num_genes, fitness_func, max_generations, crossover_type,
                                           target_solution):
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

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child1, child2 = crossover(parent1, parent2, crossover_type)
            for child in [child1, child2]:
                for i in range(num_genes):
                    if random.randint(0, 100) < GA_MUTATION:
                        child[i] = chr(random.randint(32, 126))
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Check if the target solution is found
        if ''.join(population[generation]) == target_solution:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_original_vs_HitStamp(population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock)

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    plot_original_vs_HitStamp(population, fitness_func)

    return best_individual, best_fitness


# Define the genetic algorithm without mutations
def genetic_algorithm_without_mutations(pop_size, num_genes, fitness_func, max_generations,crossover_type, target_solution):
    # Initialize the population with random individuals
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    start_time = time.time()
    start_clock = time.process_time()

    # Evolve the population for a fixed number of generations
    for generation in range(max_generations):
        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]
        elite_size = int(pop_size * GA_ELITRATE)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Generate new individuals by applying crossover operators only
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child1, child2 = crossover(parent1, parent2, crossover_type)
            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring

        # Check if the target solution is found
        if ''.join(population[generation]) == target_solution:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_original_vs_HitStamp(population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock)

        # Plot the fitness histogram every 10 generations
        if (generation + 1) % 10 == 0:
            plot_fitness_histogram(generation, population, fitness_func)

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


def genetic_algorithm_mutation_only(pop_size, num_genes, fitness_func, max_generations, target_solution):
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

        # Generate new individuals by applying mutation operators only
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent = random.choice(elites)
            child = parent.copy()

            # Apply mutation
            for i in range(num_genes):
                if random.randint(0, 100) < GA_MUTATION:
                    child[i] = chr(random.randint(32, 126))

            offspring.append(child)

        population = elites + offspring

        # Check if the target solution is found
        if ''.join(population[generation]) == target_solution:
            print_generation_stats(generation, population, fitness_func, start_time, start_clock)
            plot_original_vs_HitStamp(population, fitness_func)
            print("Solution Found!")
            best_individual = max(population, key=lambda individual: fitness_func(individual))
            best_fitness = fitness_func(best_individual)

            return best_individual, best_fitness

        # Print the statistics and running time for the current generation
        print_generation_stats(generation, population, fitness_func, start_time, start_clock)

        # Plot the fitness histogram every 10 generations
        if (generation + 1) % 10 == 0:
            plot_fitness_histogram(generation, population, fitness_func)

    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness
