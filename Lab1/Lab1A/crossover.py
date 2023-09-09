# Define the crossover operator function
import random


def crossover(parent1, parent2, crossover_type):
    if crossover_type == "SINGLE":
        # Perform single point crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    elif crossover_type == "TWO":
        # Perform two point crossover
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    elif crossover_type == "UNIFORM":
        # Perform uniform crossover
        child1 = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
        child2 = [parent2[i] if random.random() < 0.5 else parent1[i] for i in range(len(parent2))]
    else:
        raise ValueError("Invalid crossover type specified.")

    return child1, child2

