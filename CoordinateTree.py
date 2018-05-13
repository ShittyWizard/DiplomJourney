import numpy as np


class CoordinateTree:
    def __init__(self, size_max_1):
        self.size_1 = size_max_1
        self.size_2 = pow(size_max_1, 2)
        self.size_3 = pow(size_max_1, 3)
        self.tree = np.empty(self.get_size(), tuple)

    def __str__(self):
        return str(self.tree)

    def __getitem__(self, index):
        return self.tree[index]

    def __setitem__(self, index, coordinates):
        self.tree[index] = coordinates

    def get_index_of_parent(self, index_of_element):
        if index_of_element < self.size_1:
            index_of_parent = index_of_element
            return index_of_parent
        elif self.size_1 <= index_of_element < (self.size_1 + self.size_2):
            index_of_parent = (index_of_element - self.size_1) // self.size_1
            return index_of_parent
        else:
            index_of_parent = self.size_1 + ((index_of_element - self.size_1 - self.size_2) // self.size_1)
            # Return array with index of parent and index of grand-parent
            return [index_of_parent, self.get_index_of_parent(index_of_parent)]

    def get_size(self):
        return self.size_1 + pow(self.size_1, 2) + pow(self.size_1, 3)
