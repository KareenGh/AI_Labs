import random
import statistics

import numpy as np


def average_hamming_distance(population):
    total_distance = 0
    num_individuals = len(population)
    num_pairs = num_individuals * (num_individuals - 1) // 2

    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            distance = sum(x != y for x, y in zip(population[i], population[j]))
            total_distance += distance

    return total_distance / num_pairs


def winsorize(data, percentile):
    lower_bound = np.percentile(data, percentile)
    upper_bound = np.percentile(data, 100 - percentile)
    data = np.where(data < lower_bound, lower_bound, data)
    data = np.where(data > upper_bound, upper_bound, data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data


# N-Queens
# def inversion_mutation(individual):
#     size = len(individual)
#     m1, m2 = random.sample(range(size), 2)
#     if m1 > m2:
#         m1, m2 = m2, m1
#     individual[m1:m2] = reversed(individual[m1:m2])
#     return individual

def inversion_mutation(individual):
    size = len(individual)
    start, end = sorted(random.sample(range(size), 2))
    individual[start:end] = reversed(individual[start:end])
    return individual


def shuffling_mutation(individual):
    size = len(individual)
    m1, m2 = random.sample(range(size), 2)
    if m1 > m2:
        m1, m2 = m2, m1
    individual[m1:m2] = random.sample(individual[m1:m2], m2 - m1)
    return individual


def first_fit(item_sizes, bin_capacity):
    bins = [0] * len(item_sizes)
    num_bins = 0
    for item in item_sizes:
        i = 0
        while i < num_bins:
            if bins[i] + item <= bin_capacity:
                bins[i] += item
                break
            i += 1
        else:
            bins[num_bins] += item
            num_bins += 1
    return num_bins


def fitness_variance(population, fitness_func, item_sizes, bin_capacity, problem):
    if problem == "n_queens":
        fitnesses = [fitness_func(individual) for individual in population]
    elif problem == "bin_packing":
        fitnesses = [fitness_func(individual, item_sizes, bin_capacity) for individual in population]
    else:
        raise ValueError(f"Unsupported problem type: {problem}")
    variance = statistics.variance(fitnesses)
    return variance


def top_avg_selection_probability_ratio(population, selection_method, K):
    if selection_method == "RANKING_TOURNAMENT":
        top_individual_probability = 1 / K
        avg_individual_probability = 1 / len(population)
        ratio = top_individual_probability / avg_individual_probability
        return ratio
    else:
        return None


def adjust_mutation_rate(avg_kendall_tau_distance, min_mutation_rate=0.05, max_mutation_rate=0.3, diversity_threshold=25):
    if avg_kendall_tau_distance < diversity_threshold:
        return max_mutation_rate
    else:
        return min_mutation_rate
