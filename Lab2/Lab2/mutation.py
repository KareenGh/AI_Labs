import random


def update_mutation_probability(method, generation, mutation_prob, population, fitnesses, fitness_func, threshold=None,
                                stagnation_gen=None, decay_rate=None, fitness_func_args={}):
    if method == "CONSTANT":
        return mutation_prob

    elif method == "NON_UNIFORM":
        if decay_rate is None:
            decay_rate = 0.99
        return mutation_prob * (decay_rate ** generation)

    elif method == "ADAPTIVE":
        avg_fitness = sum(fitnesses) / len(fitnesses)
        if avg_fitness > threshold:
            return mutation_prob * 1.1
        else:
            return mutation_prob * 0.9

    elif method == "THM":
        if generation >= 1:
            prev_best_fitness = max(fitnesses)
            current_best_fitness = max([fitness_func(individual, **fitness_func_args) for individual in population])
            # current_best_fitness = max([fitness_func(individual) for individual in population])
            if abs(prev_best_fitness - current_best_fitness) < threshold:
                stagnation_gen += 1
                if stagnation_gen >= 5:
                    return mutation_prob * 2
            else:
                stagnation_gen = 0
        return mutation_prob

    elif method == "SELF_ADAPTIVE":
        mutation_probs = []
        for individual, fitness in zip(population, fitnesses):
            if fitness < threshold:
                mutation_probs.append(mutation_prob * 1.5)
            else:
                mutation_probs.append(mutation_prob)
        new_mutation_rate = sum(mutation_probs) / len(mutation_probs)
        return new_mutation_rate


def mutate_binary(individual, mutation_rate):
    for i in range(len(individual)):
        if random.randint(0, 100) < mutation_rate:
            individual[i] = format(random.randint(0, 255), '08b')
    return individual


def mutate_individual_species(offspring, mutation_prob, min_gene_value, max_gene_value):
    mutated_offspring = []
    for gene in offspring:
        if random.random() < mutation_prob:
            mutated_gene = random.randint(min_gene_value, max_gene_value)
            mutated_offspring.append(mutated_gene)
        else:
            mutated_offspring.append(gene)
    return mutated_offspring
