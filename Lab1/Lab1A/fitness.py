GA_TARGET = "Hello, world!"


# Define the fitness function
def fitness(individual):
    target = list(GA_TARGET)
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


# Define the fitness function with the scrambler heuristic
def fitness_HitStamp(individual):
    target = list(GA_TARGET)
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 15  # big bonus for guessing a letter in the right place
        elif individual[i] in target:
            score += 3  # small bonus for guessing a letter in the wrong place
    return score

