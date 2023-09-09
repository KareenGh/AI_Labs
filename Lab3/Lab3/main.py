import time
import random

from ALNS import ALNS, worst_removal, greedy_insertion, acceptance_criteria

from AntColonyOptimization import AntColonyOptimization
from CVRP import CVRP
from ClusteredTSPSolution import ClusteredTSPSolution
from GA import GeneticAlgorithm
from Input import read_input
from MultiStageHeuristic import MultiStageSolution
from SimulatedAnnealing import SimulatedAnnealing
from TabuSearch import TabuSearch


def main():
    start_time = time.time()
    random.seed(time.time())

    choose_input = input("Choose input file (1 - 7) \n")
    choose_input = 'input' + choose_input
    dimension, capacity, locations, demands = read_input(choose_input)

    cvrp = CVRP(dimension, capacity, locations, demands)

    print("-" * 50)
    print("Solving the problem using Multi Stage Heuristic:")
    problemName = ' Multi Stage Heuristic'
    multi_stage_solution = MultiStageSolution(cvrp)
    routes1 = multi_stage_solution.find_towns_for_route()
    cvrp.display_results(routes1, start_time,  problemName)

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Clustered TSP:")
    problemName = 'Clustered TSP'
    multi_stage_solution = ClusteredTSPSolution(cvrp, n_clusters=4)
    routes = multi_stage_solution.solve()
    cvrp.display_results(routes, start_time, problemName)
    cvrp.plot_hist(multi_stage_solution,  problemName)

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Tabu Search:")
    problemName = 'Tabu Search'
    initial_solution = routes1 #helpers.initial_solution(cvrp)
    min_tabu_tenure = 2
    max_tabu_tenure = 11
    max_iterations = 100
    tabu_search = TabuSearch(cvrp)
    best_solution = tabu_search.tabu_search(initial_solution, min_tabu_tenure, max_tabu_tenure, max_iterations)
    cvrp.display_results(best_solution, start_time, problemName)
    cvrp.plot_hist(tabu_search,  problemName)

    # print("-" * 50)
    # print("Solving the Ackley function using Tabu Search:")
    # problemName = 'Tabu Search'
    # min_tabu_tenure = 2
    # max_tabu_tenure = 11
    # max_iterations = 1000
    # dimensions = 2  # for example
    # ackley = AckleyFunction(dimensions)
    # tabu_search = TabuSearch(ackley)
    # best_solution = tabu_search.tabu_search(min_tabu_tenure, max_tabu_tenure, max_iterations)
    # print("Best solution found by {}: {}".format(problemName, best_solution))
    # print("Evaluation of the best solution: {}".format(ackley.evaluate(best_solution)))

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Ant Colony Optimization:")
    problemName = 'Ant Colony Optimization'
    aco = AntColonyOptimization(cvrp, n_ants=8, alpha=0.05, beta=7, rho=0.8, q=8, max_iter=100)
    best_solution, total_cost = aco.solve()
    cvrp.display_results(best_solution, start_time, problemName)
    cvrp.plot_hist(aco, problemName)

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Simulated Annealing:")
    problemName = 'Simulated Annealing'
    sa = SimulatedAnnealing(cvrp, T=100, alpha=0.995, max_iter=1000, restarts=7, tabu_tenure=5)
    best_solution, best_cost = sa.solve(routes1)
    cvrp.display_results(best_solution, start_time, problemName)
    cvrp.plot_hist(sa, problemName)

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Genetic Algorithm:")
    problemName = 'Genetic Algorithm'
    # ga = GeneticAlgorithm(cvrp, pop_size=200, elite_size=2, mutation_rate=0.01, generations=500, num_islands=4,
    #                       migration_interval=50, migration_size=1)
    ga = GeneticAlgorithm(cvrp, pop_size=100, elite_size=10, mutation_rate=0.1, generations=500, num_islands=6,
                          migration_interval=50, migration_size=7)
    best_solution, total_cost = ga.solve()
    cvrp.display_results(best_solution, start_time, problemName)
    cvrp.plot_hist(ga, problemName)

    start_time = time.time()

    print("-" * 50)
    print("Solving the problem using Adaptive Large Neighbourhood Search:")
    problemName = 'ALNS'
    destroy_methods = [worst_removal]
    repair_methods = [greedy_insertion]
    weights = [1, 1]
    iterations = 1000
    alns = ALNS(cvrp, destroy_methods, repair_methods, weights, iterations, acceptance_criteria,
                initial_temperature=200, cooling_rate=0.995)
    best_solution, total_cost = alns.solve()
    cvrp.display_results(best_solution, start_time, problemName)
    cvrp.plot_hist(alns, problemName)


if __name__ == "__main__":
    main()
