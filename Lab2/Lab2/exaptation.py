import numpy as np
import random

# Problem-specific parameters
R1_constraint = 3
R2_constraint = 2

# Hyperparameters
pop_size = 200
mutation_rate = 0.05
crossover_rate = 0.8
migration_rate = 0.05
num_generations = 100
migration_interval = 10


def initialize_population_1(pop_size):
    pop = []
    while len(pop) < pop_size:
        individual = tuple(3 * np.random.rand(2))
        if fitness_function_1(*individual) <= 18:  # Constraint: R <= 3
            pop.append(individual)
    return pop


def initialize_population_2(pop_size):
    pop = []
    while len(pop) < pop_size:
        individual = tuple(5 + 2 * np.random.rand(2))
        if fitness_function_2(*individual) <= 8:  # Constraint: R <= 2
            pop.append(individual)
    return pop


# Fitness functions for each island
def fitness_function_1(x, y):
    return x**2 + y**2


def fitness_function_2(x, y):
    return (x - 5)**2 + (y - 5)**2


def viability_function_1(x, y):
    return x**2 + y**2 <= 18


def viability_function_2(x, y):
    return (x - 5)**2 + (y - 5)**2 <= 8


# Update the 'is_viable' function to call the new viability functions
def is_viable(individual, island):
    x, y = individual
    if island == 1:
        return viability_function_1(x, y)
    elif island == 2:
        return viability_function_2(x, y)


# Selection, crossover, and mutation functions
# def selection(island_pop, fitness_function, island):
#     viable_pop = [ind for ind in island_pop if is_viable(ind, island)]
#     selected_pop = random.choices(viable_pop, weights=[fitness_function(*ind) for ind in viable_pop], k=len(viable_pop))
#     return selected_pop

def selection(island_pop, fitness_function, island):
    k = len(island_pop)
    k = k if k % 2 == 0 else k - 1
    selected_pop = random.choices(island_pop, weights=[fitness_function(*ind) for ind in island_pop], k=k)
    return selected_pop


# def selection(island_pop, fitness_function):
#     selected_pop = random.choices(island_pop, weights=[fitness_function(*ind) for ind in island_pop], k=len(island_pop))
#     return selected_pop


def tournament_selection(island_pop, fitness_function, island, tournament_size=3):
    selected_pop = []
    k = len(island_pop)
    k = k if k % 2 == 0 else k - 1

    for _ in range(k):
        tournament = random.sample(island_pop, tournament_size)
        winner = max(tournament, key=lambda ind: fitness_function(*ind))
        selected_pop.append(winner)

    return selected_pop


def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2


# def mutation(individual):
#     mutated_individual = []
#     for gene in individual:
#         if random.random() < mutation_rate:
#             mutated_individual.append(gene + random.uniform(-0.5, 0.5))
#         else:
#             mutated_individual.append(gene)
#     return tuple(mutated_individual)

def mutation(individual, island):
    mutated_individual = []
    for gene in individual:
        new_gene = gene + random.uniform(-0.5, 0.5) if random.random() < mutation_rate else gene
        mutated_individual.append(new_gene)
    mutated_individual = tuple(mutated_individual)
    return mutated_individual if is_viable(mutated_individual, island) else individual


# def evolve_population(island_pop, fitness_function):
#     new_population = []
#     selected_pop = selection(island_pop, fitness_function)
#
#     for i in range(0, len(selected_pop), 2):
#         parent1, parent2 = selected_pop[i], selected_pop[i+1]
#         child1, child2 = crossover(parent1, parent2)
#         new_population.append(mutation(child1))
#         new_population.append(mutation(child2))
#
#     return new_population

# def evolve_population(island_pop, fitness_function, island):
#     new_population = []
#     selected_pop = selection(island_pop, fitness_function, island)
#
#     for i in range(0, len(selected_pop), 2):
#         parent1, parent2 = selected_pop[i], selected_pop[i + 1]
#         child1, child2 = crossover(parent1, parent2)
#         new_population.append(mutation(child1, island))
#         new_population.append(mutation(child2, island))
#
#     return new_population

def evolve_population(island_pop, fitness_function, island, elitism_rate=0.1):
    new_population = []
    # selected_pop = selection(island_pop, fitness_function, island)
    selected_pop = tournament_selection(island_pop, fitness_function, island)

    # Preserve the top individuals based on elitism rate
    num_elites = int(elitism_rate * len(island_pop))
    sorted_pop = sorted(island_pop, key=lambda ind: fitness_function(*ind), reverse=True)
    elites = sorted_pop[:num_elites]
    new_population.extend(elites)

    # Perform crossover and mutation for the remaining individuals
    for i in range(num_elites, len(selected_pop) - 1, 2):
        parent1, parent2 = selected_pop[i], selected_pop[i + 1]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutation(child1, island))
        new_population.append(mutation(child2, island))

    return new_population


def migrate(island1_pop, island2_pop):
    num_migrants = int(migration_rate * pop_size)

    migrants_1to2 = random.sample(island1_pop, num_migrants)
    migrants_2to1 = random.sample(island2_pop, num_migrants)

    for i in range(num_migrants):
        if is_viable(migrants_1to2[i], 2):
            weakest_individual_index = np.argmin([fitness_function_2(*ind) for ind in island2_pop])
            island2_pop[weakest_individual_index] = migrants_1to2[i]

        if is_viable(migrants_2to1[i], 1):
            weakest_individual_index = np.argmin([fitness_function_1(*ind) for ind in island1_pop])
            island1_pop[weakest_individual_index] = migrants_2to1[i]


def get_best_individual(island_pop, fitness_function):
    best_individual_index = np.argmax([fitness_function(*ind) for ind in island_pop])
    return island_pop[best_individual_index]


def exaptation_main():

    # Initialize the populations for the two islands
    island1_pop = initialize_population_1(pop_size)
    island2_pop = initialize_population_2(pop_size)

    # Main loop
    for generation in range(num_generations):

        # Perform selection, crossover, and mutation for both islands
        island1_pop = evolve_population(island1_pop, fitness_function_1, 1)
        island2_pop = evolve_population(island2_pop, fitness_function_2, 2)

        # island1_pop = evolve_population(island1_pop, fitness_function_1)
        # island2_pop = evolve_population(island2_pop, fitness_function_2)

        # Perform migration at specified intervals
        if generation % migration_interval == 0:
            migrate(island1_pop, island2_pop)

        # Get the best individuals and fitness for the current generation
        best_individual_1 = get_best_individual(island1_pop, fitness_function_1)
        best_individual_2 = get_best_individual(island2_pop, fitness_function_2)
        best_fitness_1 = fitness_function_1(*best_individual_1)
        best_fitness_2 = fitness_function_2(*best_individual_2)

        print(f"Generation {generation + 1}:")
        print(f"  Best solution for function f: {best_individual_1}, Fitness: {best_fitness_1}")
        print(f"  Best solution for function g: {best_individual_2}, Fitness: {best_fitness_2}")

    print("\nFinal best solutions:")
    print("Best solution for function f: ", best_individual_1)
    print("Best solution for function g: ", best_individual_2)

