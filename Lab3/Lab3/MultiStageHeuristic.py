# import numpy as np
#
#
# class MultiStageSolution:
#     def __init__(self, cvrp_problem):
#         self.cvrp = cvrp_problem
#
#     # def find_towns_for_route(self): #find town to add to route than don't surpass the capacity
#     #     demands = self.cvrp.DEMAND_SECTION.copy()
#     #     demands.pop(0) #remove the warehouse which is 0
#     #     routes = []
#     #     routes.append(0)
#     #     while max(demands) > -1:
#     #         empty_capacity = self.cvrp.capacity
#     #         route = []
#     #         while max(demands) > -1 and (empty_capacity != 0 or min(demands) > empty_capacity):
#     #             max_index = self.find_max_c_in_array(empty_capacity, demands)
#     #             if max_index > 0:
#     #                 route.append(max_index)
#     #                 empty_capacity -= demands[max_index]
#     #                 demands[max_index] = -1
#     #             else:
#     #                 demands[max_index] = -1
#     #         route = self.find_best_route_for_vehicle(route)
#     #         for k in range(len(route)):
#     #             routes.append(route[k])
#     #     routes.append(0)
#     #     return routes
#
#     def find_towns_for_route(self):
#         demands = self.cvrp.demands.copy()
#         demands.pop(0)  # remove the warehouse which is 0
#         routes = [0]
#
#         while max(demands) > -1:
#             empty_capacity = self.cvrp.capacity
#             route = []
#
#             while max(demands) > -1 and (empty_capacity != 0 or min(demands) > empty_capacity):
#                 # get index of the closest city with acceptable demand
#                 max_index = self.find_closest_acceptable_city(empty_capacity, demands, route)
#                 if max_index > 0:
#                     route.append(max_index)
#                     empty_capacity -= demands[max_index]
#                     demands[max_index] = -1
#                 else:
#                     demands[max_index] = -1
#
#             # find best route for vehicle and append to routes
#             route = self.find_best_route_for_vehicle(route)
#             for k in range(len(route)):
#                 routes.append(route[k])
#         routes.append(0)
#
#         return routes
#
#     def find_closest_acceptable_city(self, c, demands, route):
#         acceptable_cities = [index for index, demand in enumerate(demands) if demand <= c and demand > 0]
#         if not acceptable_cities:
#             return -1
#
#         if not route:  # if the route is empty, find the city closest to the depot
#             distances = [self.cvrp.distance_matrix[0][city] for city in acceptable_cities]
#         else:  # otherwise, find the city closest to the last city in the route
#             last_city = route[-1]
#             distances = [self.cvrp.distance_matrix[last_city][city] for city in acceptable_cities]
#
#         return acceptable_cities[np.argmin(distances)]
#
#     def find_max_c_in_array(self, c, array):
#         max_c = 0
#         max_index = None
#         for j in range(len(array)):
#             if array[j] == c:
#                 return j
#             if max_c < array[j] < c:
#                 max_c = array[j]
#                 max_index = j
#         return max_index
#
#     def find_best_route_for_vehicle(self, route):
#         opt_route = []
#         cities_locations = self.cvrp.node_coords
#         warehouse_cord = cities_locations[0]
#         closest_to_warehouse_index = -1
#         closest_distance = 100000
#         temp_closest = 100000
#         temp_index = -1
#         cnt = 0
#         for r in range(len(route)):  # find closest to the warehouse
#             temp = self.cvrp.compute_dist(warehouse_cord[0], cities_locations[route[r]][0], warehouse_cord[1], cities_locations[route[r]][1])
#             if temp < closest_distance:
#                 closest_distance = temp
#                 closest_to_warehouse_index = route[r]
#         opt_route.append(closest_to_warehouse_index)
#         route.remove(closest_to_warehouse_index)
#
#         while route:
#             for t in range(len(route)):
#                 temp_2 = self.cvrp.compute_dist(cities_locations[opt_route[cnt]][0], cities_locations[route[t]][0],
#                                                 cities_locations[opt_route[cnt]][1], cities_locations[route[t]][1])
#             if temp_2 <= temp_closest:
#                 temp_closest = temp_2
#                 temp_index = route[t]
#             if not route:
#                 break
#             opt_route.append(temp_index)
#             route.remove(temp_index)
#             cnt += 1
#             temp_closest = 100000
#         return opt_route
#
import numpy as np


class MultiStageSolution:
    def __init__(self, cvrp_problem):
        self.cvrp = cvrp_problem

    def find_towns_for_route(self):
        demands = self.cvrp.demands.copy()
        demands.pop(0)  # remove the warehouse which is 0
        routes = []

        while max(demands) > -1:
            empty_capacity = self.cvrp.capacity
            route = []

            while max(demands) > -1 and (empty_capacity != 0 or min(demands) > empty_capacity):
                max_index = self.find_closest_acceptable_city(empty_capacity, demands, route)
                if max_index >= 0:
                    route.append(max_index)
                    empty_capacity -= demands[max_index]
                    demands[max_index] = -1
                else:
                    break

            route = self.find_best_route_for_vehicle(route)

            # ensure each route starts and ends with the depot
            if route:  # only add non-empty routes
                if route[0] != 0:
                    route.insert(0, 0)
                if route[-1] != 0:
                    route.append(0)
                routes.append(route)

        return routes

    def find_closest_acceptable_city(self, c, demands, route):
        acceptable_cities = [index for index, demand in enumerate(demands) if demand <= c and demand > 0]
        if not acceptable_cities:
            return -1

        if not route:  # if the route is empty, find the city closest to the depot
            distances = [self.cvrp.distance_matrix[0][city] for city in acceptable_cities]
        else:  # otherwise, find the city closest to the last city in the route
            last_city = route[-1]
            distances = [self.cvrp.distance_matrix[last_city][city] for city in acceptable_cities]

        return acceptable_cities[np.argmin(distances)]

    def find_best_route_for_vehicle(self, route):
        if not route:
            return []

        opt_route = [route[0]]
        remaining_cities = set(route[1:])

        while remaining_cities:
            last_city = opt_route[-1]
            next_city = min(remaining_cities, key=lambda city: self.cvrp.distance_matrix[last_city][city])
            opt_route.append(next_city)
            remaining_cities.remove(next_city)

        return opt_route
