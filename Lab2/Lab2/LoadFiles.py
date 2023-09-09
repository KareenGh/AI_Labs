
# def read_binpack_input_file(file_path):
#     with open(file_path, "r") as f:
#         while True:
#             # Read the first line to get bin capacity, number of items and number of bins in the current solution
#             line = f.readline()
#             if not line:
#                 # End of file
#                 break
#             bin_capacity, num_items, best_known_solution = map(int, line.split())
#
#             # Initialize a list to hold the item sizes
#             item_sizes = []
#
#             # Read the remaining lines to get the item sizes
#             for i in range(num_items):
#                 item_sizes.append(int(f.readline()))
#
#             yield bin_capacity, num_items, item_sizes

# Define the function to read the input file for the Bin Packing problem
def read_binpack_input_file(file_path):
    with open(file_path, "r") as f:
        while True:
            # Read the first line to get bin capacity, number of items and number of bins in the current solution
            line = f.readline()
            if not line:
                # End of file
                break
            bin_capacity, num_items, best_known_solution = map(int, line.split())

            # Initialize a list to hold the item sizes
            item_sizes = []

            # Read the remaining lines to get the item sizes
            for i in range(num_items):
                item_sizes.append(int(f.readline()))

            yield bin_capacity, item_sizes
