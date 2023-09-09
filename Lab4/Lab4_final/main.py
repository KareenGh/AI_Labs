import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


N = 6
POPULATION_SIZE = 5000
GENERATIONS = 150
SWAP_MAX = N * (N - 1) / 2   # Maximum possible swaps in a network
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 4  # The number of individuals participating in each tournament selection.
MIN_SWAPS = 12  # Optimal number of swaps
SAMPLE_SIZE = 100  # Number of sequences to sample
ELITISM_RATE = 0.2  # The percentage of the population that automatically passes to the next generation.
# Parameters that control the introduction of new random individuals to maintain diversity in the population.
DIVERSITY_INJECTION_RATE = 0.2  # Introduce 20% new random individuals every few generations
DIVERSITY_INJECTION_INTERVAL = 15  # Introduce new random individuals every 15 generations
binary_sequences = {}


def apply_network(network, sequence):
    sequence = sequence.copy()
    for i, j in network:
        if sequence[i] > sequence[j]:
            sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence


def create_network_with_min_swaps():
    network = []
    while len(network) < MIN_SWAPS:
        swap = sorted(random.sample(range(N), 2))
        if swap not in network:
            network.append(swap)
    return network


def fitness_test_sequences(test_seq, network_population):
    sampled_networks = random.sample(network_population, min(SAMPLE_SIZE, len(network_population)))
    return sum(1 for network in sampled_networks if apply_network(network, test_seq) == sorted(test_seq))


def create_network():
    return [sorted(random.sample(range(N), 2)) for _ in range(random.randint(N, SWAP_MAX))]


def crossover_network(network1, network2):
    if len(network1) < len(network2):
        network1, network2 = network2, network1
    idx = random.randint(1, len(network1) - 2)
    child1 = network1[:idx] + network2[idx:]
    child2 = network2[:idx] + network1[idx:]

    # Check if the child networks have less than MIN_SWAPS,
    # and if so, add swaps until they reach MIN_SWAPS.
    while len(child1) < MIN_SWAPS:
        swap = sorted(random.sample(range(N), 2))
        if swap not in child1:
            child1.append(swap)
    while len(child2) < MIN_SWAPS:
        swap = sorted(random.sample(range(N), 2))
        if swap not in child2:
            child2.append(swap)

    return child1, child2


def mutate_network(network):
    if random.random() < MUTATION_RATE:
        operation = random.choice(['add', 'remove', 'swap'])

        if operation == 'remove' and len(network) > MIN_SWAPS:
            # Remove a random comparator
            network.pop(random.randint(0, len(network) - 1))
        elif operation == 'add':
            # Add a random comparator
            swap = sorted(random.sample(range(N), 2))
            if swap not in network:
                network.append(swap)
        elif operation == 'swap' and len(network) > 1:
            # Swap two existing comparators
            idx1, idx2 = random.sample(range(len(network)), 2)
            network[idx1], network[idx2] = network[idx2], network[idx1]

    return network


def inject_diversity(network_population, generation):
    if generation % DIVERSITY_INJECTION_INTERVAL == 0:
        # Introduce new random individuals
        num_new_individuals = int(DIVERSITY_INJECTION_RATE * len(network_population))
        network_population[-num_new_individuals:] = [create_network() for _ in range(num_new_individuals)]
    return network_population


def tournament_selection(population, fitnesses):
    tournament = [random.choice(list(range(len(population)))) for _ in range(TOURNAMENT_SIZE)]
    return population[max(tournament, key=lambda x: fitnesses[x])]


def visualize_network(network):
    num_elements = len(network)
    num_lines = num_elements // 2

    fig, ax = plt.subplots()

    # Draw horizontal lines
    for i in range(num_lines):
        ax.plot([0, num_elements+1], [i, i], color='grey', linestyle='-', linewidth=1)

    # Draw comparators (swaps)
    colors = plt.cm.get_cmap('tab10', num_elements)
    for i, comparator in enumerate(network):
        x = [i, i]
        y = comparator
        color = colors(i)
        ax.plot(x, y, color=color, linestyle='-', linewidth=3)

    ax.set_xlim(0, num_elements)

    plt.xlabel("Comparators (Swaps)")
    plt.ylabel("Elements")
    plt.title("Sorting Network Visualization")
    plt.show()


def plot_fitness_convergence(network_fitnesses_history):
    generations = range(len(network_fitnesses_history))

    fig, ax1 = plt.subplots()
    ax1.plot(generations, network_fitnesses_history, label='Network Fitness')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title(f"Fitness Convergence for {N} Nodes")
    ax1.legend()
    plt.show()


def generate_binary_sequences(N):
    """Generates all binary sequences of length N."""
    return [[int(digit) for digit in bin(i)[2:].zfill(N)] for i in range(2 ** N)]


# binary_sequences = generate_binary_sequences(N)


def valid_network(network):
    """Checks if network is valid based on zero-one principle."""
    for sequence in binary_sequences:
        sorted_sequence = apply_network(network, sequence)
        if sorted_sequence != sorted(sequence):
            return False
    return True


fitness_cache = {}


def fitness_network(network):
    # we need to convert the network into a hashable type to use it as a key in the dictionary
    network_tuple = tuple(tuple(x) for x in network)

    if network_tuple in fitness_cache:
        return fitness_cache[network_tuple]
    else:
        score = sum(1 for sequence in binary_sequences if apply_network(network, sequence) == sorted(sequence))
        result = score - len(network) / SWAP_MAX
        fitness_cache[network_tuple] = result
        return result


def create_offspring(parents):
    """Create two offspring from two parents."""
    if random.random() < CROSSOVER_RATE:
        child1, child2 = crossover_network(*parents)
    else:
        child1, child2 = parents[0].copy(), parents[1].copy()

    return mutate_network(child1), mutate_network(child2)


def mutate_and_crossover(parents):
    """Apply mutation and crossover to a population."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        offspring = list(executor.map(create_offspring, parents))
    return offspring


def select_and_create_offspring(args):
    network_population, network_fitnesses = args
    parent1 = tournament_selection(network_population, network_fitnesses)
    parent2 = tournament_selection(network_population, network_fitnesses)
    return create_offspring((parent1, parent2))


def reset_globals_based_on_N():
    global SWAP_MAX, binary_sequences, N, MIN_SWAPS

    while True:
        print("Choose a value for N:")
        print("1. N = 6")
        print("2. N = 16")

        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            N = 6
            MIN_SWAPS = 12
            SWAP_MAX = N * (N - 1) / 2
            break
        elif choice == "2":
            N = 16
            MIN_SWAPS = 60
            SWAP_MAX = N * (N - 1) / 2
            break
        else:
            print("Invalid choice. Please try again.")

    binary_sequences = generate_binary_sequences(N)


def main():
    reset_globals_based_on_N()
    network_population = [create_network() for _ in range(POPULATION_SIZE)]
    test_seq_population = [random.sample(range(N), N) for _ in range(POPULATION_SIZE)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        network_fitnesses = list(executor.map(fitness_network, network_population))
    test_seq_fitnesses = [fitness_test_sequences(test_seq, network_population) for test_seq in test_seq_population]
    network_fitnesses_history = []

    stagnant_generations = 0
    stagnant_threshold = 35  # Threshold for stagnant generations

    for generation in range(GENERATIONS):

        network_fitnesses_history.append(max(network_fitnesses))
        print(f'Generation {generation} Fitness: {network_fitnesses_history[-1]}')

        # Inject diversity into the population
        network_population = inject_diversity(network_population, generation)

        # Check for stagnant generations
        if len(network_fitnesses_history) > 1 and network_fitnesses_history[-1] <= network_fitnesses_history[-2]:
            stagnant_generations += 1
            if stagnant_generations >= stagnant_threshold:
                print(f"Fitness didn't improve for {stagnant_generations} generations. Stopping the algorithm...")
                break
        else:
            stagnant_generations = 0

        offspring_network_population = []
        offspring_test_seq_population = []

        while len(offspring_network_population) < POPULATION_SIZE // 2:
            parent1 = tournament_selection(network_population, network_fitnesses)
            parent2 = tournament_selection(network_population, network_fitnesses)

            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover_network(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspring_network_population.append(mutate_network(child1))
            offspring_network_population.append(mutate_network(child2))

        # Generate offspring test sequences
        while len(offspring_test_seq_population) < POPULATION_SIZE // 2:
            offspring_test_seq_population.append(random.sample(range(N), N))

        # Calculate fitness for offspring test sequences using combined network population
        offspring_test_seq_fitnesses = [fitness_test_sequences(test_seq, network_population + offspring_network_population) for test_seq in offspring_test_seq_population]

        combined_network_population = network_population + offspring_network_population
        combined_test_seq_population = test_seq_population + offspring_test_seq_population

        combined_network_fitnesses = network_fitnesses + [fitness_network(network) for network in
                                                          offspring_network_population]

        combined_test_seq_fitnesses = test_seq_fitnesses + offspring_test_seq_fitnesses

        sorted_indices = sorted(range(len(combined_network_fitnesses)), key=lambda x: combined_network_fitnesses[x], reverse=True)
        combined_network_population = [combined_network_population[i] for i in sorted_indices]
        combined_test_seq_population = [combined_test_seq_population[i] for i in sorted_indices]
        combined_network_fitnesses = [combined_network_fitnesses[i] for i in sorted_indices]
        combined_test_seq_fitnesses = [combined_test_seq_fitnesses[i] for i in sorted_indices]

        # Elitism: Keep a certain number of best individuals
        elite_population_size = int(ELITISM_RATE * POPULATION_SIZE)
        elite_network_population = combined_network_population[:elite_population_size]
        elite_test_seq_population = combined_test_seq_population[:elite_population_size]

        # Include the elite population in the next generation
        network_population = elite_network_population + combined_network_population[elite_population_size:POPULATION_SIZE]
        test_seq_population = elite_test_seq_population + combined_test_seq_population[elite_population_size:POPULATION_SIZE]
        network_fitnesses = combined_network_fitnesses[:POPULATION_SIZE]
        test_seq_fitnesses = combined_test_seq_fitnesses[:POPULATION_SIZE]

    best_network = network_population[network_fitnesses.index(max(network_fitnesses))]

    if len(best_network) < MIN_SWAPS:
        best_network = create_network_with_min_swaps()

    print(f"Best Network:\n{best_network}")
    print(f"Fitness: {max(network_fitnesses)}")
    print(f"Number of Swaps: {len(best_network)}")
    print(f"Max Number of Swaps: {SWAP_MAX}")
    print(f"Is Valid Sorting Network: {valid_network(best_network)}")
    visualize_network(best_network)
    plot_fitness_convergence(network_fitnesses_history)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    main()

