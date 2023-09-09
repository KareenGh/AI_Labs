# # from sklearn.cluster import KMeans
# # from scipy.spatial import distance_matrix
# # import numpy as np
# #
# #
# # class ClusteredTSPSolution:
# #     def __init__(self, cvrp_problem, n_clusters):
# #         self.cvrp = cvrp_problem
# #         self.n_clusters = n_clusters
# #
# #     def solve(self):
# #         # Perform k-means clustering
# #         kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.cvrp.node_coords)
# #         labels = kmeans.labels_
# #
# #         # For each cluster, solve the TSP
# #         routes = []
# #         for i in range(self.n_clusters):
# #             cluster_points = [point for point, label in zip(self.cvrp.node_coords, labels) if label == i]
# #             cluster_route = self.solve_tsp(cluster_points)
# #             routes.append(cluster_route)
# #
# #         return routes
# #
# #     def solve_tsp(self, points):
# #         # Calculate the distance matrix
# #         dist_matrix = distance_matrix(points, points)
# #
# #         # Initialize the route with the first point
# #         route = [0]
# #         available_points = list(range(1, len(points)))
# #
# #         # Iteratively add the closest point to the route
# #         while available_points:
# #             last_point = route[-1]
# #             next_point = min(available_points, key=lambda point: dist_matrix[last_point][point])
# #             route.append(next_point)
# #             available_points.remove(next_point)
# #
# #         return route
#
# from sklearn.cluster import KMeans
# from scipy.spatial import distance_matrix
# import numpy as np
#
# #
# # class ClusteredTSPSolution:
# #     def __init__(self, cvrp_problem, n_clusters):
# #         self.cvrp = cvrp_problem
# #         self.n_clusters = n_clusters
# #
# #     def solve(self):
# #         # Perform k-means clustering
# #         kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.cvrp.node_coords)
# #         labels = kmeans.labels_
# #
# #         # For each cluster, solve the TSP
# #         routes = []
# #         for i in range(self.n_clusters):
# #             cluster_indices = [index for index, label in enumerate(labels) if label == i]
# #             cluster_points = [self.cvrp.node_coords[index] for index in cluster_indices]
# #             cluster_route = self.solve_tsp(cluster_points)
# #
# #             # Convert the indices back to the original indices
# #             original_indices_route = [cluster_indices[index] for index in cluster_route]
# #             routes.append(original_indices_route)
# #
# #         return routes
# #
# #     def solve_tsp(self, points):
# #         # Calculate the distance matrix
# #         dist_matrix = distance_matrix(points, points)
# #
# #         # Initialize the route with the first point
# #         route = [0]
# #         available_points = list(range(1, len(points)))
# #
# #         # Iteratively add the closest point to the route
# #         while available_points:
# #             last_point = route[-1]
# #             next_point = min(available_points, key=lambda point: dist_matrix[last_point][point])
# #             route.append(next_point)
# #             available_points.remove(next_point)
# #
# #         return route
#
#
# class ClusteredTSPSolution:
#     def __init__(self, cvrp_problem, n_clusters):
#         self.cvrp = cvrp_problem
#         self.n_clusters = n_clusters
#
#     def solve(self):
#         # Perform k-means clustering
#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.cvrp.node_coords)
#         labels = kmeans.labels_
#
#         # For each cluster, solve the TSP
#         routes = []
#         for i in range(self.n_clusters):
#             cluster_indices = [index for index, label in enumerate(labels) if label == i]
#             cluster_points = [self.cvrp.node_coords[index] for index in cluster_indices]
#
#             # Calculate the distance matrix for the points in this cluster
#             dist_matrix = distance_matrix(cluster_points, cluster_points)
#
#             # Create a list of indices for the points in this cluster
#             cluster_indices_route = list(range(len(cluster_points)))
#
#             # Use 2-opt to find an optimal route for this cluster
#             cluster_route = self.two_opt(cluster_indices_route, dist_matrix)
#
#             # Convert the indices back to the original indices
#             original_indices_route = [cluster_indices[index] for index in cluster_route]
#             routes.append(original_indices_route)
#
#         return routes
#
#     def distance(self, route, dist_matrix):
#         return sum(dist_matrix[route[i - 1]][route[i]] for i in range(len(route)))
#
#     def two_opt(self, route, dist_matrix):
#         best = route
#         improved = True
#         while improved:
#             improved = False
#             for i in range(1, len(route) - 2):
#                 for j in range(i + 1, len(route)):
#                     if j - i == 1: continue  # changes nothing, skip then
#                     new_route = route[:]
#                     new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-optSwap
#                     if self.distance(new_route, dist_matrix) < self.distance(best, dist_matrix):
#                         best = new_route
#                         improved = True
#             route = best
#         return best
import warnings

from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)

# Explicitly set the value of n_init
kmeans = KMeans(n_clusters=4, n_init=10)


class ClusteredTSPSolution:
    def __init__(self, cvrp_problem, n_clusters):
        self.cvrp = cvrp_problem
        self.n_clusters = n_clusters
        self.best_costs = []

    def solve(self):
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.cvrp.node_coords[1:])
        labels = kmeans.labels_

        # For each cluster, solve the TSP
        routes = []
        for i in range(self.n_clusters):
            cluster_indices = [index for index, label in enumerate(labels) if label == i]
            cluster_indices = [index + 1 for index in cluster_indices]
            cluster_indices = [0] + cluster_indices + [0]
            cluster_points = [self.cvrp.node_coords[index] for index in cluster_indices]
            dist_matrix = distance_matrix(cluster_points, cluster_points)
            cluster_indices_route = list(range(len(cluster_points)))
            cluster_route = self.two_opt(cluster_indices_route, dist_matrix)
            original_indices_route = [cluster_indices[index] for index in cluster_route]
            routes.append(original_indices_route)
        return routes

    def distance(self, route, dist_matrix):
        return sum(dist_matrix[route[i - 1]][route[i]] for i in range(1, len(route)))

    def two_opt(self, route, dist_matrix):
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):  # Exclude depot
                for j in range(i + 2, len(route)):  # Exclude depot and next city
                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-optSwap
                    if self.distance(new_route, dist_matrix) < self.distance(best, dist_matrix):
                        best = new_route
                        improved = True
            self.best_costs.append(self.distance(best, dist_matrix))  # Record the best cost at this iteration
            route = best
        return route



