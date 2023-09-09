from fitness import n_queens_fitness


def binary_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return binary_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def lcs_distance(s1, s2):
    lcs_length = longest_common_subsequence(s1, s2)
    return len(s1) + len(s2) - 2 * lcs_length


def hamming_distance_fitness(individual):
    target = 'Hello, world!'
    target_binary = [format(ord(c), '07b') for c in target]
    if len(target_binary) != len(individual):
        raise ValueError("Target and individual must have the same length")
    return -hamming_distance(target_binary, individual)


def jaccard_similarity_fitness(individual):
    target = 'Hello, world!'
    target_binary = [format(ord(c), '07b') for c in target]
    return jaccard_similarity(target_binary, individual)


def lcs_distance_fitness(individual):
    target = 'Hello, world!'
    target_binary = [format(ord(c), '07b') for c in target]
    return -lcs_distance(target_binary, individual)


def binary_edit_distance_fitness(individual):
    target = 'Hello, world!'
    target_binary = [format(ord(c), '07b') for c in target]
    return -binary_levenshtein_distance(target_binary, individual)


# Bin packing metrics
def kendall_tau_distance(config1, config2):
    n = len(config1)
    inversions = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (config1[i] - config1[j]) * (config2[i] - config2[j]) < 0:
                inversions += 1

    return inversions


# N-Queens metrics
def non_attacking_pairs_ratio(population):
    total_pairs = 0
    non_attacking_pairs = 0

    for individual in population:
        size = len(individual)
        pairs = size * (size - 1) // 2
        total_pairs += pairs
        conflicts = n_queens_fitness(individual)
        non_attacking_pairs += (pairs - conflicts)

    ratio = non_attacking_pairs / total_pairs
    return ratio


def average_kendall_tau_distance(population):
    total_distance = 0
    num_individuals = len(population)
    num_comparisons = 0

    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            distance = kendall_tau_distance(population[i], population[j])
            total_distance += distance
            num_comparisons += 1

    average_distance = total_distance / num_comparisons
    return average_distance

