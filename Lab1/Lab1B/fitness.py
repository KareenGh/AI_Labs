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
        for j in range(i+1, size):
            if individual[i] == individual[j]:
                conflicts += 1
            elif abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1

    return conflicts


# def bin_packing_fitness(individual, item_sizes, bin_capacity):
#     num_bins = max(individual) + 1
#     bin_space = [bin_capacity] * num_bins
#
#     for i, bin_index in enumerate(individual):
#         bin_space[bin_index] -= item_sizes[i]
#
#     unused_space = sum(space for space in bin_space if space >= 0)
#     return unused_space

# def bin_packing_fitness(individual, item_sizes, bin_capacity):
#     # Initialize the number of bins used to zero
#     num_bins_used = 0
#
#     # Initialize a list to hold the remaining capacity of each bin
#     remaining_capacity = []
#
#     for item_size in item_sizes:
#         # Check if the current item fits in any of the existing bins
#         item_fits = False
#         for i, capacity in enumerate(remaining_capacity):
#             if item_size <= capacity:
#                 remaining_capacity[i] -= item_size
#                 item_fits = True
#                 break
#
#         # If the item doesn't fit in any of the existing bins, create a new bin
#         if not item_fits:
#             remaining_capacity.append(bin_capacity - item_size)
#             num_bins_used += 1
#
#     # Return the number of bins used as the fitness value
#     return num_bins_used

def bin_packing_fitness(individual, item_sizes, bin_capacity):
    num_bins = max(individual) + 1
    bin_space = [bin_capacity] * num_bins

    for i, bin_index in enumerate(individual):
        bin_space[bin_index] -= item_sizes[i]

    unused_space = sum(space for space in bin_space if space >= 0)
    return unused_space


