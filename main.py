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

    @staticmethod
    def is_pandigital(number):
        u"""Return true if number is pandigital. Number should be integer or string"""
        number = str(number)
        usage = [0 for _ in range(len(number))]
        for digit in number:
            if int(digit) <= len(number):
                usage[int(digit) - 1] += 1
        if all(usage) == 1:
            return True
        return False

    @staticmethod
    def permute(number):
        from itertools import permutations
        return np.fromiter((int("".join(x)) for x in permutations(str(number))), int)

    @staticmethod
    def pandigital_numbers(n_start=1, n_stop=9):
        u"""Return n-pandigital numbers.
        If first argument is 1, function starts with numbers of length 1.
        If second argument is 9, function ends on all permutation of 123456789.
        Note: there is 9! = 362880 of such numbers.
        """
        starters = []
        results = np.array([], int)
        _range = range(n_start, n_stop + 1) if n_start <= n_stop else range(n_start, n_stop - 1, -1)
        for l in _range:
            digit = ""
            for n in range(1, l + 1):
                digit += str(n)
            starters.append(digit)
        for number in starters:
            permutations = EulerRoutines.permute(number)
            results = np.append(results, permutations)
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


def problem_31():
    solutions = 0
    coins = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=np.int16)
    factors = np.zeros(8, dtype=np.int16)
    solution = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=int)

    solutions = []

    def func(factors):
        return np.sum(factors * coins) == 200

    # Brute force approach, 8 loops: 4 800 000 000 possibilities, no way to go
    # for one in range(0, 200):
    #     for two in range(0, 100):
    #         for five in range(0, 40):
    #             for ten in range(0, 20):
    #                 for twenty in range(0, 25):
    #                     for fifty in range(0, 4):
    #                         for one_hundred in (0, 1, 2):
    #                             for two_hundreds in (0, 1):
    #                                 if func(np.array([one, two, five, ten, twenty, fifty, one_hundred, two_hundreds])):
    #                                     solutions.append(np.array([one, two, five, ten, twenty, fifty,
    #                                                                one_hundred, two_hundreds]))

    return solutions, len(solutions)


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


def problem_41():
    max_n = 0
    for number in EulerRoutines.pandigital_numbers(9, 3):
        if EulerRoutines.is_prime(number) and number > max_n:
            max_n = number
    return max_n


def problem_42():
    from string import ascii_uppercase
    alphabet = {letter: index for (letter, index) in zip(ascii_uppercase, range(1, len(ascii_uppercase) + 1))}

    with open("words.txt") as fi:
        words = fi.readline()

    new_words = words.split(',')
    words = np.array([], dtype=np.str)
    for w in new_words:
        words = np.append(words, w.replace('"', ''))

    def func(n): return int(0.5 * n * (n + 1))
    values = set([func(x) for x in range(1, 2500)])
    results = []
    for word in words:
        word_value = sum(alphabet[letter] for letter in word)
        if word_value in values:
            results.append(word)
    return results


if __name__ == '__main__':
    pass
    # number = problem_5()
    # print(number)
    # print(problem_30())
    # print(problem_31())
    # print(problem_32())
    # print(problem_27())
    # res = problem_42()
    # print(len(res), res)
    print(problem_41())
