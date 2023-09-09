
def average_genetic_distance(population):
    distances = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            distance = sum(a != b for a, b in zip(population[i], population[j]))
            distances.append(distance)
    return sum(distances) / len(distances) if distances else 0


def unique_alleles(population):
    unique_alleles_per_gene = [set() for _ in range(len(population[0]))]
    for individual in population:
        for i, gene in enumerate(individual):
            unique_alleles_per_gene[i].add(gene)
    return sum(len(unique_set) for unique_set in unique_alleles_per_gene)


def compute_population_diversity(population):
    def hamming_distance(individual1, individual2):
        return sum(a != b for a, b in zip(individual1, individual2))

    total_gen_dist = 0
    num_individuals = len(population)
    unique_alleles = set()

    for i in range(num_individuals):
        for j in range(i+1, num_individuals):
            gen_dist = hamming_distance(population[i], population[j])
            total_gen_dist += gen_dist

            unique_alleles.update(set(population[i]), set(population[j]))

    avg_gen_dist = total_gen_dist / (num_individuals * (num_individuals - 1) / 2)
    num_unique_alleles = len(unique_alleles)

    return avg_gen_dist, num_unique_alleles
