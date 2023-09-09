
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
