class Sudoku:

    def __init__(self, data: list[list[int]]):
        self.data = data

    def __repr__(self):
        string = ""
        for line in self.data:
            string += '|'
            for number in line:
                string += str(number) + "|"
            string += '\n'
        return string

    def solve(self):
        def find_candidate(i, j):
            if self.data[i][j] != 0:
                return i, j, self.data[i][j]  # Save found
            # Determine square
            square_row = 0 if i < 3 else 1 if i < 6 else 2
            square_col = 0 if j < 3 else 2 if j < 6 else 2
            number = set(range(1, 10)).difference(self.data[i]).difference([self.data[k][j] for k in range(9)])
            print(number)
        find_candidate(1, 0)


if __name__ == '__main__':
    sudoku = Sudoku([[3, 0, 0, 2, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 7, 0, 0, 0],
                     [7, 0, 6, 0, 3, 0, 5, 0, 0],
                     [0, 7, 0, 0, 0, 9, 0, 8, 0],
                     [9, 0, 0, 0, 2, 0, 0, 0, 4],
                     [0, 1, 0, 8, 0, 0, 0, 5, 0],
                     [0, 0, 9, 0, 4, 0, 3, 0, 1],
                     [0, 0, 0, 7, 0, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 8, 0, 0, 6]])
    print(sudoku.solve())
