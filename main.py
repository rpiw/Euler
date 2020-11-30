import numpy as np


class EulerRoutines:

    _primes = np.array([], dtype=int)  # cache primes

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

    @staticmethod
    def is_prime(number: int):
        if number < 2:
            return False
        if EulerRoutines._primes.size > 0 and number < EulerRoutines._primes[-1]:
            return number in EulerRoutines._primes
        for d in range(2, np.int(np.ceil(np.sqrt(number))) + 1):
            if number % d == 0 and number != d:
                return False
        return True

    @staticmethod
    def primes(upper_limit: int, algorithm="brute", cache=True) -> np.array:
        u"""Return array containing primes below upper_limit integer. Algorithm: brute - not implemented yet.
        : cache (bool) - is true store primes in an array as a class member EulerRoutines._primes ."""

        start = 3
        if EulerRoutines._primes.size > 0:
            if EulerRoutines._primes[-1] > upper_limit:
                return np.array([x for x in EulerRoutines._primes if x < upper_limit])
            else:
                start = EulerRoutines._primes[-1]
        results = np.array([2], dtype=int)

        n = start
        while n <= upper_limit:
            if EulerRoutines.is_prime(n):
                results = np.append(results, np.int(n))
            n += 2

        results = np.append(EulerRoutines._primes, results)

        if cache:
            EulerRoutines._primes = results

        return results


def problem_5():
    # brute force
    number = 2520
    while True:
        if all((number % i == 0 for i in range(1, 21))):
            break
        else:
            number += 2
    return number


def problem_27():
    u"""Project Euler problem 27: Considering quadratics of the form:
    n ** 2 + a * n + b , where |b| <= 1000 and |a| < 1000 .
    Find the product of the coefficients, a and b, for the quadratic expression that produces the maximum number of
    primes for consecutive values of n, starting with n = 0."""
    # So, let we think... b must be a prime if result must be a prime for n = 0.
    # Therefore, begin with searching all primes below 1000
    possible_b = EulerRoutines.primes(1000)

    def f(n, a, b): return n ** 2 + a * n + b
    max_n = 0
    A, B = 0, 0
    for b in possible_b:
        for a in np.arange(-10**3, 10**3, 1, dtype=int):
            n = 0
            while True:
                if not EulerRoutines.is_prime(f(n, a, b)):
                    break
                n += 1
            if n > max_n:
                max_n = n
                A, B = a, b
    return A, B, max_n, A * B


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
    # print(problem_32())
    print(problem_27())
