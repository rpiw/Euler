from functools import reduce

import numpy as np
from typing import Union


class EulerRoutines:

    # @TODO: use functool.cache decorator
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

    _primes_file_name = "primes.npy"

    @staticmethod
    def primes(upper_limit: int, algorithm="sieve", cache=True, read=True) -> np.array:
        u"""Return array containing primes below upper_limit integer. Algorithm: brute, sieve.
        : cache (bool) - is true store primes in an array as a class member EulerRoutines._primes ."""

        start = 3
        if read:
            read = np.array([])
            try:
                read = np.load(EulerRoutines._primes_file_name)
            except FileNotFoundError:
                pass

            if read.size > 0:
                EulerRoutines._primes = np.unique(np.concatenate((EulerRoutines._primes, read)), 0)

        if EulerRoutines._primes.size > 0:
            if EulerRoutines._primes[-1] > upper_limit:
                return np.array([x for x in EulerRoutines._primes if x < upper_limit])
            else:
                start = EulerRoutines._primes[-1]
        results = np.array([2], dtype=int)

        n = start

        if algorithm == "brute":
            while n <= upper_limit:
                if EulerRoutines.is_prime(n):
                    results = np.append(results, np.int(n))
                n += 2

            results = np.append(EulerRoutines._primes, results)

        elif algorithm == "sieve":
            results = np.arange(3, upper_limit, 2, dtype=int)
            sieve = {x: True for x in results}
            print("Preparing for searching primes with Sieve of this ancient greek guy with funny name")
            print("Space: [{start}, {end}]".format(start=3, end=upper_limit))
            for number in results:
                if sieve[number] and EulerRoutines.is_prime(number):
                    k = number
                    while k <= upper_limit:
                        k += number
                        sieve[k] = False

            results = np.fromiter((x for x in sieve.keys() if sieve[x]), dtype=int)
            results = np.append(np.array([2], dtype=int), results)

        if cache:
            EulerRoutines._primes = results
            np.save(EulerRoutines._primes_file_name, results)

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
        u"""Return all permutation of digits in number."""
        from itertools import permutations
        return np.fromiter((int("".join(x)) for x in permutations(str(number))), int)

    @staticmethod
    def rotate_digits(number):
        u"""Rotate digits: for example input = 197, return: set(197, 971, 719)."""
        number = str(number)
        res = set()
        for j in range(len(number)):
            new_number = {}
            for i in range(len(number)):
                new_number[i] = number[len(number) - i % len(number) - 1 - j]
            res.add(int("".join(reversed(new_number.values()))))
        return res

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

    @staticmethod
    def divisors(number):
        u"""Find all divisors of a given number."""
        results = np.array([1], dtype=int)
        for divisor in range(2, int(np.sqrt(number)) + 1):
            if number % divisor == 0:
                results = np.append(results, divisor)
        for divisor in results:
            res = number % divisor
            if res == 0 and number // divisor != divisor:
                results = np.append(results, number // divisor)
        return sorted(results)

    @staticmethod
    def number_name(number):
        u"""Return number name in British English."""
        if number > 999:
            print("Number is greater than excepted: less than 1000")
            raise ValueError

        return Numeral(number).name


class Numeral:
    _numerals = {}  # cache names of numbers

    @classmethod
    def numerals(cls, file="numerals.txt"):
        u"""Create a dict with unique numerals from file or return dict if has already been created."""
        if Numeral._numerals:
            return Numeral._numerals
        else:
            import csv
            with open(file, "r") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter="\t")
                for row in csv_reader:
                    Numeral._numerals[int(row[0].strip().replace(",", ""))] = row[1].strip()
        return Numeral._numerals

    def __init__(self, number: Union[int, float, str]):
        self.number = number
        self._string = str(self.number)
        self.order = len(self._string) - 1
        self.factors = []
        n = self.number

        for i in reversed(range(len(self._string))):
            factor = n // 10 ** i
            n = n - factor * 10 ** i
            self.factors.append(factor)

        self._name = self.create_name()

    def __repr__(self):
        return "Numeral: {number} ".format(number=self.number) + str(self.factors)

    @property
    def name(self):
        return self._name

    def create_name(self):
        # Create part for tens <= 31
        if self.number <= 31:
            return Numeral.numerals()[self.number]

        name_parts = []
        # Create part for numbers in range(31, 100)
        unit, tens = "", ""
        part = int(str(self.factors[-2]) + str(self.factors[-1]))

        if 0 < part <= 31:
            tens = Numeral.numerals()[part]
        else:
            if self.factors[-1]:
                unit = "-" + Numeral.numerals()[self.factors[-1]]
            if self.factors[-2]:
                tens = Numeral.numerals()[self.factors[-2] * 10]
        name_parts.append(tens + unit)

        # Create a part for hundred
        if self.number >= 100:
            hundred = ""
            if self.factors[-3]:
                hundred = Numeral.numerals()[self.factors[-3]] + " hundred"
                if self.factors[-2] or self.factors[-1]:
                    hundred += " and "
            name_parts.append(hundred)

        # TODO: Create a part for thousand

        return "".join(reversed(name_parts))


def problem_5():
    # brute force
    number = 2520
    while True:
        if all((number % i == 0 for i in range(1, 21))):
            break
        else:
            number += 2
    return number


def problem_12():
    triangle_number = 1
    i = 1
    while len(EulerRoutines.divisors(triangle_number)) <= 500:
        i += 1
        triangle_number += i
    return triangle_number


def problem_17():
    v = 0
    for i in range(1, 1000):
        v += len(Numeral(i).name.replace(" ", "").replace("-", ""))
    v += len("onethousand")
    return v


def problem_19():
    from itertools import cycle
    day = cycle(range(1, 8))
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 30, 31, 31, 31]
    year = 1901
    sundays = 0
    stop = False
    while True:
        if stop:
            break
        if year % 4:
            days_in_months[1] = 29

        months = range(12)
        for m in months:
            days = range(1, days_in_months[m] + 1)
            for d in days:
                if next(day) == 7 and d == 1:
                    sundays += 1

                if year == 2000 and m == 11 and d == 31:
                    stop = True
        year += 1

    return sundays


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


def problem_33():
    from fractions import Fraction
    results = []
    for numerator in np.arange(10, 100, dtype=int):
        for denominator in np.arange(numerator + 1, 100, dtype=int):
            for i in range(2):
                for j in range(2):
                    if str(numerator)[i] == str(denominator)[j]:
                        try:
                            f1 = Fraction(int(str(numerator)[1 - i]), int(str(denominator)[1 - j]))
                            f2 = Fraction(numerator, denominator)
                            if f1 == f2 and numerator % 10 != 0 and denominator % 10 != 0:
                                results.append((numerator, denominator))
                        except ZeroDivisionError:
                            continue
    res = reduce(lambda x, y: x * y, (Fraction(x[0], x[1]) for x in results))
    return res


def problem_34():
    max_n = int(10e6)  # Need to approximate an upper limit: 9! = 362880
    results = np.array([], dtype=int)
    from math import factorial

    # pure brute force - factorials should be precalculated for speed, but I don't care
    for number in range(3, max_n + 1):
        digits = (int(s) for s in str(number))
        suma = sum(map(factorial, digits))
        if number == suma:
            results = np.append(results, number)
    return results


def problem_35():
    primes = EulerRoutines.primes(1000000)
    primes_set = set(primes)
    results = np.array([], dtype=int)
    counter = 0
    print(primes)
    for prime in primes:
        permutations = EulerRoutines.rotate_digits(prime)
        for perm in permutations:
            if perm not in primes_set:
                break
        else:
            counter += 1
            results = np.append(results, prime)
    return counter, results


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
    # print(problem_41())
    # res34 = problem_34()
    # print(res34)
    # print(sum(res34))
    # print(problem_12())
    # print(problem_19())
    # print(problem_35())
    print(problem_33())
