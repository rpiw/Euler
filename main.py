import numpy as np


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


if __name__ == '__main__':
    # number = problem_5()
    # print(number)
    print(problem_30())
