import random
import numpy as np

from utils import winsorize


# Modify the scale_fitness function to use the winsorize function
def scale_fitness(fitnesses, generation, max_generations):
    # Calculate the percentile based on the current generation
    start_percentile = 10
    end_percentile = 5
    percentile = start_percentile - (generation / max_generations) * (start_percentile - end_percentile)

    # Apply the winsorize function with the dynamic percentile
    scaled_fitnesses = winsorize(np.array(fitnesses), percentile)
    return scaled_fitnesses


# Roulette Wheel Selection (RWS) with scaling
def roulette_wheel_selection(population, scaled_fitnesses):
    total_fitness = sum(scaled_fitnesses)
    r = random.uniform(0, total_fitness)
    partial_sum = 0
    for i, individual in enumerate(population):
        partial_sum += scaled_fitnesses[i]
        if partial_sum >= r:
            return individual


# Stochastic Universal Sampling (SUS) with scaling
def stochastic_universal_sampling(population, scaled_fitnesses, num_parents=2):
    selected_parents = []
    total_fitness = sum(scaled_fitnesses)
    pointer_distance = total_fitness / num_parents
    start_pointer = random.uniform(0, pointer_distance)

    for _ in range(num_parents):
        pointer = start_pointer
        partial_sum = 0
        for i, individual in enumerate(population):
            partial_sum += scaled_fitnesses[i]
            if partial_sum >= pointer:
                selected_parents.append(individual)
                break
        start_pointer += pointer_distance

    parent1 = selected_parents.pop(random.randrange(len(selected_parents)))
    parent2 = selected_parents.pop(random.randrange(len(selected_parents)))
    return parent1, parent2


# Ranking and deterministic tournament selection with parameter K
def ranking_and_tournament_selection(population, fitnesses, K):
    selected_parents = []
    num_parents = len(population)

    # Ranking
    ranked_indices = sorted(range(num_parents), key=lambda i: fitnesses[i], reverse=True)
    ranked_population = [population[i] for i in ranked_indices]

    for _ in range(num_parents):
        # Deterministic tournament
        tournament_indices = random.sample(range(num_parents), K)
        best_index = max(tournament_indices, key=lambda i: fitnesses[i])
        selected_parents.append(ranked_population[best_index])

    parent1 = selected_parents.pop(random.randrange(len(selected_parents)))
    parent2 = selected_parents.pop(random.randrange(len(selected_parents)))
    return parent1, parent2


def roulette_wheel_selection_species(species, fitnesses):
    total_fitness = sum(fitnesses)
    r = random.uniform(0, total_fitness)
    partial_sum = 0
    for i, individual in enumerate(species):
        partial_sum += fitnesses[i]
        if partial_sum >= r:
            return individual
