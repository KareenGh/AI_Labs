# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
#
# def jaccard_similarity(a, b):
#     intersection = sum([1 for i in range(len(a)) if a[i] == b[i]])
#     union = len(a) - intersection
#
#     # Check for an empty union
#     if union == 0:
#         return 1
#
#     return intersection / union
#
#
# def shared_fitness(individual, population, fitness_func, sharing_radius, item_sizes, bin_capacity):
#     niche_count = sum(1 for other in population if jaccard_similarity(individual, other) < sharing_radius)
#     original_fitness = fitness_func(individual, item_sizes, bin_capacity)
#     return original_fitness / niche_count
#
#
# def non_deterministic_crowding(population, fitnesses, offspring, offspring_fitnesses):
#     new_population = []
#
#     for parent1, parent2, child1, child2 in zip(population, population[1:] + [population[0]], offspring, offspring[1:] + [offspring[0]]):
#         parent1_fitness = fitnesses[population.index(parent1)]
#         parent2_fitness = fitnesses[population.index(parent2)]
#         child1_fitness = offspring_fitnesses[offspring.index(child1)]
#         child2_fitness = offspring_fitnesses[offspring.index(child2)]
#
#         if jaccard_similarity(parent1, child1) + jaccard_similarity(parent2, child2) < jaccard_similarity(parent1, child2) + jaccard_similarity(parent2, child1):
#             if parent1_fitness < child1_fitness:
#                 new_population.append(parent1)
#             else:
#                 new_population.append(child1)
#
#             if parent2_fitness < child2_fitness:
#                 new_population.append(parent2)
#             else:
#                 new_population.append(child2)
#         else:
#             if parent1_fitness < child2_fitness:
#                 new_population.append(parent1)
#             else:
#                 new_population.append(child2)
#
#             if parent2_fitness < child1_fitness:
#                 new_population.append(parent2)
#             else:
#                 new_population.append(child1)
#
#     return new_population[:len(population)]
#
#
# def speciation_with_clustering(population, num_clusters=None):
#     # Determine the optimal number of clusters if not provided
#     if num_clusters is None:
#         num_clusters = find_optimal_clusters(population)
#
#     # Perform clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(population)
#     labels = kmeans.labels_
#
#     # Divide the population into species
#     species = [[] for _ in range(num_clusters)]
#     for i, individual in enumerate(population):
#         species[labels[i]].append(individual)
#
#     return species
#
#
# def find_optimal_clusters(population):
#     max_clusters = 10  # This is an example value;
#     # Standardize the data
#     scaler = StandardScaler()
#     standardized_population = scaler.fit_transform(population)
#
#     # Calculate the inertia for different numbers of clusters
#     inertias = []
#     for k in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(standardized_population)
#         inertias.append(kmeans.inertia_)
#
#     # Find the elbow point
#     elbow_point = find_elbow_point(inertias)
#
#     return elbow_point
#
#
# def find_elbow_point(inertias):
#     x = list(range(1, len(inertias) + 1))
#     y = inertias
#
#     # Fit a line through the inertia points
#     coefficients = np.polyfit(x, y, 1)
#     line = np.poly1d(coefficients)
#
#     # Calculate the distance from each point to the fitted line
#     distances = []
#     for i in range(len(x)):
#         point = np.array([x[i], y[i]])
#         line_point = np.array([x[i], line(x[i])])
#         distance = np.linalg.norm(point - line_point)
#         distances.append(distance)
#
#     # Find the point with the maximum distance to the line
#     elbow_point = np.argmax(distances) + 1
#
#     return elbow_point
import numpy as np
from sklearn.cluster import KMeans


def fitness_sharing(population, fitnesses, sigma_share, alpha):
    shared_fitnesses = []
    epsilon = 1e-6
    for i, fit_i in enumerate(fitnesses):
        niche_count = sum([sharing_function(fit_i, fit_j, sigma_share, alpha) for j, fit_j in enumerate(fitnesses) if i != j])
        shared_fitness = fit_i / (niche_count + epsilon)
        shared_fitnesses.append(shared_fitness)
    return shared_fitnesses


def sharing_function(fit_i, fit_j, sigma_share, alpha):
    d = abs(fit_i - fit_j)
    if d < sigma_share:
        return 1 - (d / sigma_share) ** alpha
    else:
        return 0


def crowding(population, fitnesses, crowding_factor):
    population_size = len(population)
    crowding_distances = [0] * population_size

    for i in range(population_size):
        distances = [abs(fitnesses[i] - fitnesses[j]) for j in range(population_size) if i != j]
        distances.sort()
        crowding_distances[i] = sum(distances[:crowding_factor])

    return [fitness / (1 + crowding_distance) for fitness, crowding_distance in zip(fitnesses, crowding_distances)]


def speciation_with_clustering(population, fitnesses, num_species, clustering_algorithm='k-means'):
    if clustering_algorithm == 'k-means':
        kmeans = KMeans(n_clusters=num_species, random_state=0, n_init=10).fit(np.array(fitnesses).reshape(-1, 1))
        species_labels = kmeans.labels_
    else:
        raise ValueError(f"Unknown clustering algorithm: {clustering_algorithm}")

    species_fitnesses = [[] for _ in range(num_species)]

    for i, species_label in enumerate(species_labels):
        species_fitnesses[species_label].append(fitnesses[i])

    shared_fitnesses = []
    for i, species_label in enumerate(species_labels):
        species_fitness = fitnesses[i]
        species_size = len(species_fitnesses[species_label])

        if species_size > 1:
            shared_fitness = species_fitness / species_size
        else:
            shared_fitness = species_fitness

        shared_fitnesses.append(shared_fitness)

    return shared_fitnesses
