import numpy as np


class EulerRoutines:

    @staticmethod
    def pandigital_product(number1: int, number2: int) -> int:
        u"""Return product if multiplicand and multiplier and product can be written as a 1 through 9, else 0."""
        product = number1 * number2
        if product > 999999999:
            return 0
        one_to_nine = np.array(range(1, 10), dtype=int)
        results = np.array([False for _ in one_to_nine], dtype=bool)
        digit = str(product) + str(number1) + str(number2)
        for char in digit:
            if int(char) in one_to_nine and not results[int(char) - 1]:
                results[int(char) - 1] = True
            else:
                return 0
        return product


def problem_5():
    # brute force
    number = 2520
    while True:
        if all((number % i == 0 for i in range(1, 21))):
            break
        else:
            number += 2
    return number


def problem_30():
    # brute force approach
    upper_limit = 6 * 9**5
    number = 2  # because why not
    results = np.array([])
    while True:
        if number == np.sum(np.array([int(x)**5 for x in str(number)])):
            results = np.append(results, np.array([number]))
        number += 1
        if number == upper_limit:
            break

    return np.sum(results)


# def problem_31():
#     solutions = 0
#     coins = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=np.int16)
#     factors = np.zeros(8, dtype=np.int16)
#     solution = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=int)
#
#     def func(factors):
#         return np.sum(factors * coins) == 200
#
#     return solutions
#

def problem_32():
    results = np.array([], dtype=int)
    upper_bound = 10 ** 4
    for f in range(1, upper_bound):
        for f2 in range(10 ** (4 - len(str(f))), 10 ** (5 - len(str(f)))):
            l = len(str(f * f2) + str(f) + str(f2))
            if l != 9:
                continue
            d = EulerRoutines.pandigital_product(f, f2)
            if d and d not in results:
                print(f, f2, d)
                results = np.append(results, d)
    return np.sum(results)


if __name__ == '__main__':
    pass
    # number = problem_5()
    # print(number)
    # print(problem_30())
    # print(problem_31())
    print(problem_32())