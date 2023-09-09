import random


# Define the crossover operator function
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
    # elif crossover_type == "UNIFORM_BINARY":
    #     # Perform uniform crossover with bitwise XOR
    #     child1 = ''.join([bin(int(parent1[i], 2) ^ int(parent2[i], 2))[2:].zfill(8) if random.random() < 0.5
    #                       else parent1[i] for i in range(len(parent1))])
    #     child2 = ''.join([bin(int(parent2[i], 2) ^ int(parent1[i], 2))[2:].zfill(8) if random.random() < 0.5
    #                       else parent2[i] for i in range(len(parent2))])

    return child1, child2


# N-Queens
def pmx(parent1, parent2):
    size = len(parent1)
    p1, p2 = [0] * size, [0] * size

    # Choose crossover points
    cx1 = random.randint(0, size)
    cx2 = random.randint(0, size - 1)
    if cx2 >= cx1:
        cx2 += 1
    else:  # Swap the two crossover points
        cx1, cx2 = cx2, cx1

    # Apply crossover between cx1 and cx2
    for i in range(cx1, cx2):
        p1[i] = parent2[i]
        p2[i] = parent1[i]

    # Map the remaining elements
    for i in range(size):
        if i < cx1 or i >= cx2:
            p1[i] = parent1[i]
            p2[i] = parent2[i]

            while p1[i] in p1[cx1:cx2]:
                p1[i] = parent1[parent2.index(p1[i])]

            while p2[i] in p2[cx1:cx2]:
                p2[i] = parent2[parent1.index(p2[i])]

    return p1, p2


def cx(parent1, parent2):
    size = len(parent1)
    p1, p2 = [-1] * size, [-1] * size
    indices = [0]

    # Find the first cycle
    while indices[-1] != 0 or len(indices) == 1:
        indices.append(parent1.index(parent2[indices[-1]]))

    # Assign the values in the first cycle
    for i in indices[:-1]:
        p1[i] = parent1[i]
        p2[i] = parent2[i]

    # Fill in the remaining values
    for i in range(size):
        if p1[i] == -1:
            p1[i] = parent2[i]
            p2[i] = parent1[i]

    return p1, p2


def crossover_binary(parent1, parent2, crossover_type):
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

    return child1, child2

