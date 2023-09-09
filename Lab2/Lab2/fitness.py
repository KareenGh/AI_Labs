import Levenshtein

GA_TARGET = "Hello, world!"


# Define the fitness function
def fitness(individual):
    target = list(GA_TARGET)
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


# Define the fitness function with the scrambler heuristic
def fitness_HitStamp(individual):
    target = list(GA_TARGET)
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 15  # big bonus for guessing a letter in the right place
        elif individual[i] in target:
            score += 3  # small bonus for guessing a letter in the wrong place
    return score


# Modify the fitness function to include age
def fitness_with_age(individual, age, fitness_func, alpha, max_age):
    # calculate the fitness score for the candidate solution
    original_score = fitness_func(individual)
    # normalize the age component
    normalized_age = age / max_age  # divide age by the maximum age in the population
    # calculate the age component of the fitness score
    age_score = 1 - normalized_age  # reverse the age score so that younger candidates get higher scores
    # combine the two scores with a weighted sum
    total_score = (1 - alpha) * original_score + alpha * age_score

    return total_score


def n_queens_fitness(individual):
    size = len(individual)
    conflicts = 0

    for i in range(size):
        for j in range(i + 1, size):
            if individual[i] == individual[j]:
                conflicts += 1
            elif abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1

    return conflicts


# def bin_packing_fitness(individual, item_sizes, bin_capacity):
#     bin_space = {}
#     for i, bin_index in enumerate(individual):
#         if bin_index not in bin_space:
#             bin_space[bin_index] = bin_capacity
#         bin_space[bin_index] -= item_sizes[i]
#
#     fill_levels = [bin_capacity - space for space in bin_space.values()]
#     min_fill_level = min(fill_levels)
#     return -min_fill_level

def bin_packing_fitness(individual, item_sizes, bin_capacity):
    num_bins = max(individual) + 1
    bin_space = [bin_capacity] * num_bins

    for i, bin_index in enumerate(individual):
        bin_space[bin_index] -= item_sizes[i]

    unused_space = sum(space for space in bin_space if space >= 0)
    return unused_space


# def binary_distance(a, b):
#     return sum([a[i] != b[i] for i in range(len(a))])
#
#
# def binary_hit_stamp_fitness(individual):
#     target = 'Hello, world!'
#     target_binary = [format(ord(c), '07b') for c in target]
#     fitness = 0
#     for i, gene in enumerate(individual):
#         fitness += 7 - binary_distance(gene, target_binary[i % len(target_binary)])
#     return fitness


'''
This fitness function uses the Levenshtein distance algorithm to calculate the distance between the binary 
representation of the individual and the binary representation of the target string. The Levenshtein distance is a 
metric for measuring the difference between two strings. It is defined as the minimum number of insertions, deletions, 
or substitutions required to transform one string into the other.

In this function, the Levenshtein distance is calculated for each character in the target string by comparing it to the
corresponding character in the binary representation of the individual. The sum of these distances is then subtracted 
from the length of the target string to obtain the fitness value. The fitness value represents how close the individual 
is to the target string, with a higher fitness value indicating a closer match.
'''


def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def binary_hit_stamp_fitness(individual):
    distance = sum(levenshtein_distance(individual[i], format(ord(GA_TARGET[i]), '07b')) for i in range(len(GA_TARGET)))
    return len(GA_TARGET) - distance


#
# def binary_representation(text, encoding='utf-8'):
#     binary = ''.join(format(ord(c), '08b') for c in text)
#     return [int(bit) for bit in binary]


# def fitness_function_binary_distance(individual):
#     target_binary = binary_representation(GA_TARGET)
#     hamming_distance = sum(ind1 != ind2 for ind1, ind2 in zip(individual, target_binary))
#     normalized_distance = hamming_distance / len(target_binary)
#     fitness_score = 1 - normalized_distance
#     return fitness_score


# def fitness_HitStamp_binary(individual):
#     individual = [chr(int(individual[i:i + 8], 2)) for i in range(0, len(individual), 8)]
#     target = list(GA_TARGET)
#     score = 0
#     for i in range(len(individual)):
#         if individual[i] == target[i]:
#             score += 15  # big bonus for guessing a letter in the right place
#         elif individual[i] in target:
#             score += 3
#     return score

# def fitness_HitStamp_binary(individual):
#     target = list(GA_TARGET)
#     score = 0
#     for i in range(len(individual)):
#         binary_indiv = format(ord(individual[i]), '08b') # Convert character to 8-bit binary string
#         binary_target = format(ord(target[i]), '08b') # Convert character to 8-bit binary string
#         for j in range(8):
#             if binary_indiv[j] == binary_target[j]:
#                 score += 1  # increase the score for each matching bit
#     return score


# def hamming_distance(str1, str2):
#     return sum(c1 != c2 for c1, c2 in zip(str1, str2))
#
#
# def hamming_fitness(individual):
#     target = 'Hello, world!'
#     target_binary = [format(ord(c), '07b') for c in GA_TARGET]
#
#     decoded_individual = ''.join([chr(int(gene, 2)) for gene in individual])
#     min_distance = float('inf')
#
#     for i in range(len(decoded_individual) - len(GA_TARGET) + 1):
#         substring = decoded_individual[i:i + len(GA_TARGET)]
#         distance = hamming_distance(substring, GA_TARGET)
#         if distance < min_distance:
#             min_distance = distance
#
#     return len(GA_TARGET) - min_distance


# def binary_distance(a, b):
#     return sum([a[i] != b[i] for i in range(len(a))])
#
#
# def binary_hit_stamp_fitness(individual):
#     target_binary = [format(ord(c), '07b') for c in GA_TARGET]
#     fitness = 0
#     for i, gene in enumerate(individual):
#         fitness += 7 - binary_distance(gene, target_binary[i % len(target_binary)])
#     return fitness


# GA_TARGET_BINARY = [format(ord(c), '07b') for c in GA_TARGET]
#
#
# def hamming_distance(str1, str2):
#     return sum(c1 != c2 for c1, c2 in zip(str1, str2))
#
#
# def binary_hit_stamp_fitness(individual):
#     fitness = sum(hamming_distance(GA_TARGET_BINARY[i % len(GA_TARGET_BINARY)], individual[i]) for i in range(len(individual)))
#
#     return len(individual) - fitness

