
def read_input(filename):

    with open(filename) as file:
        lines = file.readlines()

    dimension = None
    capacity = None
    node_coord_section = []
    demand_section = []

    for line in lines:
        if line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            node_coord_section = [list(map(int, line.split()[1:])) for line in lines[lines.index(line)+1:lines.index("DEMAND_SECTION\n")]]
        elif line.startswith("DEMAND_SECTION"):
            demand_section = [int(line.split()[-1]) for line in lines[lines.index(line)+1:lines.index("DEPOT_SECTION\n")]]

    print("Dimension:", dimension)
    print("Capacity:", capacity)
    print("Node coordinates:", node_coord_section)
    print("Demands:", demand_section)

    return dimension, capacity, node_coord_section, demand_section
