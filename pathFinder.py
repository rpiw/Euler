import numpy as np
from typing import Tuple


def adjacent_vertices(x, y, shape: Tuple):
    u"""x, y - position int,
        shape - maximum value of x and y"""
    results = []
    shape = {x: shape[0], y: shape[1]}
    for k in (x, y):
        if k > 0:
            vertices = (k - 1, k, k + 1)
        elif k == 0:
            vertices = (k, k + 1)
        elif k == shape[k]:
            vertices = (k - 1, k)
        results.append(vertices)

    return ((i, j) for i in results[0] for j in results[1] if not (i == x and j == y))


class PathFinder2D:

    def __init__(self, matrix: np.array):
        self.matrix = matrix

    def find_path(self, start: Tuple[int, int] = (0, 0), end: Tuple[int, int] = (-1, -1), algorithm="Dijkstra"):
        if algorithm == "Dijkstra":
            return self.dijkstra_algorithm(start, end)

    def dijkstra_algorithm(self, start, end):
        shortest_path_tree = []
        not_included_yet = [(x, y) for x in range(len(self.matrix)) for y in range(len(self.matrix))]
        distances = {vertex: np.inf for vertex in not_included_yet}
        distances[start] = 0  # this boyo is the first to choose

        while not_included_yet:
            current = min(distances)
            shortest_path_tree.append(current)
            for i in adjacent_vertices(*current, end):
                print(i)
            break


if __name__ == '__main__':
    matrix = np.loadtxt("p081_matrix.txt", delimiter=",", dtype=int)
    path_finder = PathFinder2D(matrix)
    path_finder.find_path()
