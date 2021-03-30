import functools
import itertools
import math

import numpy
import numpy as np
from typing import Union, Sequence
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

console_formatter = logging.Formatter('(%ascitime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)


class LoggingHelper:
    u"""Class keeping logging simple. Let your class inherit from this one for simple logging."""

    def __init__(self, logger_name: str = __name__, level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.logger.setLevel(self.level)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.level)

        self.console_formatter = logging.Formatter('(%ascitime)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(self.console_formatter)

        self.logger.addHandler(self.console_handler)


class EulerRoutinesSettings:
    u"""Class for keeping different settings."""
    # TODO: track maximum amount of memory for algorithms and memory management


class EulerRoutines(LoggingHelper):
    _primes = np.array([], dtype=int)  # cache primes
    _primes_set = set()

    logger_name = __name__

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
    @functools.cache
    def is_prime(number: int):
        # EulerRoutines().logger.info(f"Checking if {number} is prime.")
        if EulerRoutines._primes.size > 0 and number < EulerRoutines._primes[-1]:
            return number in EulerRoutines._primes_set
        for d in range(2, np.int(np.ceil(np.sqrt(number))) + 1):
            if number % d == 0 and number != d:
                return False
        return True

    _primes_file_name = "primes.npy"

    @staticmethod
    def primes(upper_limit: int, algorithm="sieve", cache=True, read=True) -> np.array:
        u"""Return array containing primes below upper_limit integer.
        :rtype: object
        :algorithm:
            brute - check every integer if is a prime, very slow method!
            sieve - use sieve of eratosthenes, fast and reliable.
        : cache (bool) - is true store primes in an array as a class member EulerRoutines._primes,
        : read (bool) - if true, try to read primes saved on a disk in a file of name in
         variable EulerRoutines._primes_file_name,
         does not effect if upper_limit is higher than the biggest saved prime and algorithm is sieve."""

        if read:
            try:
                loaded = np.load(EulerRoutines._primes_file_name)
            except IOError:
                loaded = np.array([], dtype=int)

            if loaded.size > 0:
                EulerRoutines._primes = np.unique(np.concatenate((EulerRoutines._primes, loaded), 0))
                logger.debug(f"Primes loaded from file {EulerRoutines._primes_file_name}.")

        if EulerRoutines._primes.size > 0 and read:
            return np.array([x for x in EulerRoutines._primes if x < upper_limit])

        # here begins the calculations...
        results = np.array([2], dtype=int)

        if algorithm == "brute":  # not recommended!
            n = EulerRoutines._primes[-1]
            logger.warning("This method works really slowly! Better use one of sieve methods!")
            while n <= upper_limit:
                if EulerRoutines.is_prime(n):
                    results = np.append(results, np.int(n))
                n += 2

            results = np.append(EulerRoutines._primes, results)

        elif algorithm == "sieve":
            logger.info("Preparing for searching primes with Sieve of Eratosthenes.")
            logger.debug(f"Space: 2, {upper_limit}")
            from primes import sieve_of_eratosthenes
            results = sieve_of_eratosthenes(upper_limit)

        if cache:
            logger.debug("Caching...")
            EulerRoutines._primes = results
            EulerRoutines._primes_set = set(results)
            np.save(EulerRoutines._primes_file_name, results)
            logger.info(f"Successfully saved to file: {EulerRoutines._primes_file_name}.")

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
    def gcd(number1, number2, *args):
        u"""Return greatest common divisor."""
        return max(list(functools.reduce(set.intersection, (set(EulerRoutines.divisors(n)) for n in (
            number1, number2, *args)))))

    @staticmethod
    def coprimes(number1, number2, *args):
        u"""Return true if a greatest common divisor of given numbers is 1."""
        if EulerRoutines.gcd(number1, number2, *args) == 1:
            return True
        return False

    @staticmethod
    def number_name(number):
        u"""Return number name in British English."""
        if number > 999:
            print("Number is greater than excepted: less than 1000")
            raise ValueError

        return Numeral(number).name

    @staticmethod
    def is_palindromic(number):
        string = str(number)
        for i in range(int(len(string) / 2)):
            if string[i] != string[-i - 1]:
                return False
        return True

    @staticmethod
    def two_base_palindromic(number):
        string10 = str(number)
        string2 = bin(number)[2:]
        for i in range(int(len(string10) / 2)):
            if string10[i] != string10[-i - 1]:
                return False
        for i in range(int(len(string2) / 2)):
            if string2[i] != string2[-i - 1]:
                return False
        return True

    @staticmethod
    def truncate_numbers(number, from_front=True, from_end=True):
        u"""Returns array of numbers created from truncating digits."""
        results = np.array([number], dtype=int)
        if from_front:
            other = np.fromiter((str(number)[i:] for i in range(1, len(str(number)))), dtype=int)
            results = np.append(results, other)
        if from_end:
            other = np.fromiter((str(number)[:-i] for i in range(1, len(str(number)))), dtype=int)
            results = np.append(results, other)
        return results

    @staticmethod
    def concatenated_product(number, integer):
        if integer <= 1:
            raise ValueError
        return "".join(str(number * x) for x in range(1, integer + 1))

    @staticmethod
    @functools.cache
    def pentagonal_number(n: int):
        u"""Return pentagonal number for given integer n."""
        return int(n * (3 * n - 1) / 2)

    @staticmethod
    def reverse_number(number: int) -> int:
        return int("".join([str(number)[i] for i in range(len(str(number)) - 1, -1, -1)]))

    @staticmethod
    def traverse_matrix(matrix, allowed_moves="r l d u", target="minimum"):
        u"""Traverse through matrix by searching for minimal/maximal sum of elements.
        :matrix - np.array 2x2,
        :allowed_moves - str with letter r l d u for right, left, down, up, delimiter should be a blank space.
        :target - if target is 'minimum' - the sum on path is minimized, if target is 'maximum', sum is maximized"""
        path = []
        allowed_steps = allowed_moves.split(" ")
        steps = {"r": (0, 1), "l": (0, -1), "d": (1, 0), "u": (-1, 0)}
        x0, y0 = 0, 0
        x, y = x0, y0
        value = matrix[x, y]
        while (x, y) != matrix.shape:
            values = {}
            local_minimal = 9999

            for step in allowed_steps:
                try:
                    values[step] = value + matrix[[x + steps[step][0], y + steps[step][1]]]
                except IndexError:
                    continue
                if values[step] <= local_minimal:
                    local_minimal = values[step]
                    next_position = (x + steps[step][0], y + steps[step][1])
                raise NotImplemented

    # @staticmethod
    # def factorize(number: int, x: int = 2):
    #     u"""Factorize. Algorithm: Pollard's rho algorithm"""
    #
    #     for cycle in itertools.count(1):
    #         y = x
    #         for i in range(2 ** cycle):
    #             x = (x * x + 1) % number
    #             factor = math.gcd(x - y, number)
    #             if factor > 1:
    #                 return factor

    @staticmethod
    def factorize_small_numbers(number: int, *args):
        u"""Return prime factors of a number (or tuple of numbers). For large number use factorize"""
        if args:
            numbers = [number] + list(*args)
        else:
            numbers = (number,)
        results = {}
        for n in numbers:
            results[n] = []
            factor = 2
            new_n = n
            while new_n != 1:
                while new_n % factor == 0:
                    new_n //= factor
                    results[n].append(factor)
                else:
                    factor += 1

        return results

    @staticmethod
    def factorize(start: int = 4, stop: int = -1):
        u"""Factorize consecutive integers in range(start, stop + 1). Uses sieve with primes up to sqrt(stop)."""
        results = {}
        if isinstance(start, Sequence):
            start, stop = start[0], start[-1]
        elif isinstance(start, set):
            start = list(start)
            start, stop = start[0], start[-1]

        if stop == -1:
            stop = start

        primes = EulerRoutines.primes(int(math.sqrt(stop)))
        primes_set = set(primes)
        for n in range(start, stop + 1):
            results[n] = []

            if n in primes_set:
                results[n] = [n]  # if is prime, skip it
                continue

            primes_iter = iter(primes)
            factor = next(primes_iter)
            new_n = n

            while new_n != 1:
                while new_n % factor == 0:
                    new_n //= factor
                    results[n].append(factor)
                else:
                    try:
                        factor = next(primes_iter)
                    except StopIteration:
                        if new_n > math.sqrt(n):
                            results[n].append(new_n)  # prime larger than sqrt(end)
                        break

        return results

    @staticmethod
    def euler_totient_function(number: int) -> int:
        u"""Compute Euler's totient function."""
        primes = EulerRoutines.primes(number)
        result = number
        for prime in primes:
            if number % prime == 0:
                result *= 1 - 1 / prime
        return result

    @staticmethod
    def euler_totient_function_array(max_n: int):
        u"""Compute Euler's totient function for all n from 1 to max_n. Return dict."""
        if max_n <= 1 or not isinstance(max_n, int):
            raise TypeError

        primes = EulerRoutines.primes(max_n)
        phi = {n: n for n in range(1, max_n + 1)}
        for i in range(2, max_n + 1):
            if phi[i] == i:
                for j in range(i, max_n + 1, i):
                    phi[j] -= phi[j] / i

        return phi

    @staticmethod
    def is_square(number: int) -> bool:
        return number == math.isqrt(number) ** 2

    @staticmethod
    def solve_pell_equation(n: int, *args):
        u"""Solve equation of form: x ** 2 - N * y ** 2 == 1 for x, y, N integers.
        Algorithm: Chakravala method."""
        numbers = (n,)
        if args:
            for item in args:
                if not isinstance(item, int):
                    raise TypeError
            numbers += args
        max_a, max_b, max_k = 200, 200, 200
        # logger.info(f"Begin to solve Pell's equation by the Chakravala method. There is {len(numbers)} numbers.")

        solutions = {}
        for number in numbers:
            if EulerRoutines.is_square(number):
                # logger.info(f"The number {number} is perfect square, solution does not exists!")
                solutions[number] = None  # if number is square, there is no solution at all!

            # ###
            # a**2 - N * b**2 = k, when k == 1, solution of (a, b, 1) is completed.

            a, b, k = 1, 1, 1
            # Begin with solution for any k found by a brute force.
            while a < max_a:
                k = (a ** 2 - number * b ** 2)
                if k == int(k):  # and k > 0:
                    break
                a += 1
            else:
                raise IndexError("Could not find solution.")

            # Then proceed with trivial solution of (a, b, k) = (m, 1, m**2 - N).
            # After create a new solutions of form (am + NB, a + bm, k(m**2 - N)).
            # if abs(k) == 1, 2, 4, we can use Brahmagupta's identity
            found = False
            while not found:
                acceptable_m = []
                distance = 10 ** 10
                m = 1
                while True:
                    if (a + b * m) % abs(k) == 0:
                        value_to_min = abs(m ** 2 - number)
                        print(m, value_to_min, distance)
                        if value_to_min < distance:

                            acceptable_m.append(m)
                            distance = value_to_min

                        elif value_to_min > distance:
                            minimal_m = acceptable_m[-2]
                            print("Brejk")
                            break
                        else:
                            print("What am I doing here?")
                    m += 1

                m = minimal_m
                print(a, b, k, m)
                a, b, k = tuple(((a * m + number * b) / abs(k), (a + b * m / abs(k)), (m ** 2 - number) / k))
                if abs(k) in (1, 2, 4):
                    found = True
            else:
                solutions[number] = (a, b, k)

        return solutions

    @staticmethod
    def digital_sum(number: int) -> int:
        return sum((int(x) for x in str(number)))

    @staticmethod
    def digital_root(number: int) -> int:
        root = str(number)
        while len(str(root)) > 1:
            root = sum((int(x) for x in str(root)))
        return root


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


def problem_26():
    results = {}
    # Multiple numerator by 10 and check remainder
    for n in range(2, 1001):
        numerator = 1
        remainder = numerator % n
        local_results = [remainder]

        while True:
            numerator = 10 * remainder
            remainder = numerator % n

            if remainder in local_results:
                break
            local_results.append(remainder)

            if len(local_results) >= n:
                break

        results[n] = local_results
    maximum = 0
    current_i = None
    for i in results.keys():
        if len(results[i]) > maximum:
            maximum = len(results[i])
            current_i = i
    return current_i, maximum


def problem_27():
    u"""Project Euler problem 27: Considering quadratics of the form:
    n ** 2 + a * n + b , where |b| <= 1000 and |a| < 1000 .
    Find the product of the coefficients, a and b, for the quadratic expression that produces the maximum number of
    primes for consecutive values of n, starting with n = 0."""
    # So, let we think... b must be a prime if result must be a prime for n = 0.
    # Therefore, begin with searching all primes below 1000
    possible_b = EulerRoutines.primes(1000)

    def f(n, a, b):
        return n ** 2 + a * n + b

    max_n = 0
    A, B = 0, 0
    for b in possible_b:
        for a in np.arange(-10 ** 3, 10 ** 3, 1, dtype=int):
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
    upper_limit = 6 * 9 ** 5
    number = 2  # because why not
    results = np.array([])
    while True:
        if number == np.sum(np.array([int(x) ** 5 for x in str(number)])):
            results = np.append(results, np.array([number]))
        number += 1
        if number == upper_limit:
            break

    return np.sum(results)


def problem_31():
    coins = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=np.int16)

    target = 200
    solutions = [1] + [0] * target
    for coin in coins:
        for i in range(coin, target + 1):
            solutions[i] += solutions[i - coin]

    return solutions[target]


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
    res = functools.reduce(lambda x, y: x * y, (Fraction(x[0], x[1]) for x in results))
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


def problem_36():
    results = []
    for i in range(1, 1000000):
        if EulerRoutines.two_base_palindromic(i):
            results.append(i)

    return results, sum(results)


def problem_37():
    found = []

    # Starts with 3-digit numbers
    def t(n):
        return all(map(EulerRoutines.is_prime, EulerRoutines.truncate_numbers(n)))

    for p in EulerRoutines.primes(1000000):
        if t(p): found.append(p)
    # Remove 2, 3, 5, 7
    found = found[4:]
    return found, sum(found)


def problem_38():
    solutions = []
    for number in range(2, 10000):
        i = 1
        while True:
            i += 1
            product = EulerRoutines.concatenated_product(number, i)
            length = len(str(product))
            if length > 9:
                break
            elif length < 9:
                continue
            if EulerRoutines.is_pandigital(product):
                solutions.append((int(product), number, i))

    return max(solutions), solutions


def problem_40():
    seq = "".join(str(x) for x in range(1, 10 ** 6 + 1))
    return functools.reduce(lambda x, y: x * y, (int(seq[10 ** i - 1]) for i in range(1, 7)))


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

    def func(n):
        return int(0.5 * n * (n + 1))

    values = set([func(x) for x in range(1, 2500)])
    results = []
    for word in words:
        word_value = sum(alphabet[letter] for letter in word)
        if word_value in values:
            results.append(word)
    return results


def problem_43():
    divisors = (2, 3, 5, 7, 11, 13, 17)
    pans = EulerRoutines.permute(1234567890)
    results = []

    def func(number):
        if number <= 987654321:
            return False

        for i in range(7):
            p = str(number)
            if int(p[i + 1] + p[i + 2] + p[i + 3]) % divisors[i] != 0:
                return False
        return True

    for pandigit in pans:
        if func(pandigit):
            results.append(pandigit)

    return sum(results), results


def problem_44():
    u"""Pentagonal numbers are generated by the formula, Pn=n(3n−1)/2. Brute force approach."""
    numbers = dict((n, EulerRoutines.pentagonal_number(n)) for n in range(1, 10 ** 6))
    results = {}
    possible_results = set([(3 * n ** 2 - n) // 2 for n in range(1, 10000)])  # I dont know how to check the upper limit
    for i in range(1, 5000):  # let we assume there is a solution... if not, just change value for higher
        pentagonal_i = (3 * i ** 2 - i) // 2
        for j in range(i, 5000):
            pentagonal_j = (3 * j ** 2 - j) // 2
            sum_pentagonal = pentagonal_j + pentagonal_i
            difference = pentagonal_j - pentagonal_i
            if sum_pentagonal in possible_results and difference in possible_results:
                results[(i, j)] = difference
    return results


def problem_46():
    limit = 10000
    primes = set(EulerRoutines.primes(limit))
    numbers = np.array([x for x in range(3, limit, 2) if x not in primes], dtype=int)

    def goldbach(number):
        for prime in reversed([p for p in primes if p < number]):
            rest, int_part = math.modf(math.sqrt((number - prime) / 2))
            if math.isclose(rest, 0, rel_tol=1e-7):
                return False
        return True

    for number in numbers:
        if goldbach(number):
            return number
    return None


def problem_47():
    u"""

    The first two consecutive numbers to have two distinct prime factors are:

    14 = 2 × 7
    15 = 3 × 5

    The first three consecutive numbers to have three distinct prime factors are:

    644 = 2² × 7 × 23
    645 = 3 × 5 × 43
    646 = 2 × 17 × 19.

    Find the first four consecutive integers to have four distinct prime factors each. What is the first of these
    numbers?

    Well, we dont need to factorize fully numbers - if number has got less or more than 4 distinct prime factors,
    factorization can be stopped.
    """
    limit = int(1.4 * 10 ** 5)
    primes = EulerRoutines.primes(limit, read=True)
    primes_set = set(primes)
    not_primes = set(range(1, limit + 1)) - primes_set
    factors = EulerRoutines.factorize(not_primes)
    local_results = []

    for n in range(4, limit + 1):  # we start with 4 in  factorize!
        if len(local_results) == 4:
            return local_results

        if len(set(factors[n])) == 4:

            if not local_results:
                local_results.append(n)
            else:
                if n - local_results[-1] == 1:
                    local_results.append(n)
                else:
                    local_results.clear()
                    local_results.append(n)
    return


def problem_49():
    primes = [x for x in EulerRoutines.primes(10001, read=False) if len(str(x)) == 4]
    primes_set = set(primes)
    results = []
    for prime_1 in primes:
        for prime_2 in (x for x in primes if x > prime_1):
            number_3 = prime_2 + (prime_2 - prime_1)
            if number_3 in primes_set:
                permutations = set(EulerRoutines.permute(prime_1))
                if prime_2 in permutations and number_3 in permutations:
                    results.append("".join(map(str, (prime_1, prime_2, number_3))))
    return results


def problem_50():
    u"""Find prime below 10^6 that can be written as a sum of consecutive primes."""
    primes = EulerRoutines.primes(1000000)
    primes_set = set(primes)
    results = []

    for i in range(len(primes)):
        number = primes[i]
        for j in range(i + 1, len(primes)):
            number += primes[j]
            if number > primes[-1]:
                break
            if number in primes_set:
                results.append((j - i, number))

    return max(results)


def problem_52():
    number = 123456
    while True:
        number += 1
        if number % 2 == 0:
            continue
        perms = EulerRoutines.permute(number)
        if all(x * number in perms for x in range(2, 7)):
            return number


def problem_53():
    func = lambda n, r: functools.reduce(lambda x, y: x * y, (n - r + i for i in range(1, r + 1)))

    @functools.cache
    def factorial(n):
        return math.factorial(n)

    results = set()
    for n in range(1, 101):
        for r in range(1, n + 1):
            if func(n, r) >= factorial(r) * 10 ** 6:
                results.add((n, r))
    return len(results)


def problem_55():
    results = []
    for i in range(1, 10001):
        iterations = 0
        number = i
        while iterations < 50:
            number = number + EulerRoutines.reverse_number(number)
            if EulerRoutines.is_palindromic(number):
                break
            iterations += 1
        else:
            if iterations == 50:
                results.append(number)
    return len(results)


def problem_56():
    maximum = 0
    for a in range(2, 101):
        for b in range(2, 101):
            power = sum(int(x) for x in str(a ** b))
            maximum = power if power > maximum else maximum
    return maximum


def problem_57():
    from PyFraction import infinite_continued_fraction
    results = infinite_continued_fraction(2, 1000)
    i = 0
    for result in results:
        frac = result[2]
        if len(str(frac.numerator)) > len(str(frac.denominator)):
            i += 1
    return i


def problem_58():
    limit = 10 ** 7
    primes_on_diagonals = 0
    a = 3
    counter = 0
    while counter < limit:
        a1, a2, a3 = a ** 2 - a + 1, a ** 2 - 2 * a + 2, a ** 2 - 3 * a + 3
        if EulerRoutines.is_prime(a1):
            primes_on_diagonals += 1
        if EulerRoutines.is_prime(a2):
            primes_on_diagonals += 1
        if EulerRoutines.is_prime(a3):
            primes_on_diagonals += 1

        diagonals = 4 * (a // 2) + 1

        if primes_on_diagonals / diagonals < 0.1:
            return a, primes_on_diagonals, diagonals, primes_on_diagonals / diagonals, counter
        a += 2
        counter += 1


def problem_59():
    cipher = np.loadtxt("p059_cipher.txt", delimiter=',', dtype=int)
    most_common = np.bincount(cipher).argmax()  # Space symbol ' ' or letter e
    first_occurence = np.where(cipher == most_common)[0][0]  # index 2: last char of the key is ' ' ^ this one
    key = [" ", " ", chr(cipher[2] ^ ord(' '))]
    characters = {" ": cipher[first_occurence]}

    # 2nd most common letter is "i" in English.
    # last character should be '.'. So, let we check if the key's length divides cipher's length.
    if len(cipher) % 3 == 0:
        if chr(cipher[-1] ^ ord(' ')) == ".":
            characters["."] = cipher[-1]

    print(key)

    # so first 2 characters must be: 1 uppercase letter and 1 lowercase letter and make 2 letters long word
    with open("words.txt", "r") as fi:
        dictionary = fi.readline().replace('"', '').split(',')
    two_letters_dict = [x[0] + x[1].lower() for x in dictionary if len(x) == 2]
    print(two_letters_dict)
    from itertools import cycle
    from string import ascii_lowercase
    import re

    def decipher(key: str):
        return "".join(chr(letter ^ ord(k)) for (letter, k) in zip(cipher, cycle(key)))

    pattern = re.compile(r'\s|\w')
    # for char1 in ascii_lowercase:
    #     letter1 = chr(ord(char1) ^ cipher[0])
    #     for char2 in ascii_lowercase:
    #         letter2 = chr(ord(char2) ^ cipher[1])
    #         if letter1 + letter2 in two_letters_dict:
    #             print(letter1 + letter2)
    for word in two_letters_dict:
        letter = chr(ord(word[0]) ^ cipher[0])
        letter2 = chr(ord(word[1]) ^ cipher[1])
        print(letter + letter2)


def problem_61():
    # def p3(n):
    #     return n * (n + 1) // 2
    #
    # def p4(n):
    #     return n ** 2
    #
    # def p5(n):
    #     return n * (3 * n - 1) // 2
    #
    # def p6(n):
    #     return n * (2 * n - 1)
    #
    # def p7(n):
    #     return n * (5 * n - 1)
    #
    # def p8(n):
    #     return n * (3 * n - 2)

    # if (p3[:2] == p4[2:] and p4[:2] == p5[2:] and p5[:2] == p6[2:] and p7[:2] == p8[2:] and p8[:2] == p3[2:]) or \
    #         (p3[:1] == p4[1:] and p4[:1] == p5[1:] and p5[:1] == p6[1:] and p7[:1] == p8[1:] and p8[:1] == p3[1:]):
    #     print(n, p3, p4, p5, p6, p7, p8, sum([p3, p4, p5, p6, p7, p8]))

    numbers = [set(), set(), set(), set(), set(), set()]

    for n in range(19, 144):
        numbers[0].add(str(n * (n + 1) // 2))
        numbers[1].add(str(n ** 2))
        numbers[2].add(str(n * (3 * n - 1) // 2))
        numbers[3].add(str(n * (2 * n - 1)))
        numbers[4].add(str(n * (5 * n - 1)))
        numbers[5].add(str(n * (3 * n - 2)))


    def find_solution(i, j, partial_solution=[]):
        if partial_solution:  # partial solution has a structure: [(index of column, number), ...]
            # begin with the lastest entry
            index = partial_solution[-1][0]
            number = partial_solution[-1][1]
            indices = [x for x in range(6) if x != index]
            for idx in indices:
                for other_number in numbers[idx]:
                    if number[2:] == other_number[:2]:
                        pass

        # if there is no partial solution yet
        for number in numbers[i]:
            for other_number in numbers[j]:
                if number[2:] == other_number[:2]:
                    pass
        if j + 1 <= 5:
            find_solution(i, j + 1)
        elif 5 >= i + 1 != j:
            find_solution(i + 1, j)
        else:
            find_solution(i + 1, 0)
    return


def problem_62():
    number = 100
    _range = range(1, 10 ** 4)
    cubes = np.array([x ** 3 for x in _range], dtype=int)
    cubes_set = set(np.array([x ** 3 for x in _range], dtype=int))  # list of cubes
    for i in _range:
        cube = cubes[i]
        local_sum = 0
        results = []
        permutations = set(EulerRoutines.permute(cube))
        for permutation in permutations:
            if permutation in cubes_set:
                local_sum += 1
                results.append(permutation)
            if local_sum == 5 and functools.reduce(lambda x, y: len(str(x)) == len(str(y)), results):
                return i, results


def problem_63():
    total = 0
    for i in range(1, 1000):
        for j in range(1, 1000):
            word = str(i ** j)
            if len(word) == j:
                total += 1
            elif len(word) > j:
                break
    return total


def problem_64():
    u"""All credits go to:
    https://web.archive.org/web/20151221205104/http://web.math.princeton.edu/mathlab/jr02fall/Periodicity/mariusjp.pdf
    theorem 2.3 from the link above."""

    # first, determine if a fraction is rational
    from PyFraction import continued_fraction_expansion
    expansions = continued_fraction_expansion([x for x in range(2, 10001)])
    total = 0
    for n in range(2, 10000):
        if len(expansions[n][1]) % 2 == 1:
            total += 1
    return total


def problem_65():
    from fractions import Fraction
    from decimal import Decimal
    e = [Decimal(1) for _ in range(100)]

    k = 0
    for i in range(100):
        if i % 3 == 1:
            k += 1
            e[i] = 2 * k
    e = [Decimal(2)] + e
    # test_sample = [Fraction(x[0], x[1]) for x in [(2, 1), (3, 1), (8, 3), (11, 4), (19, 7), (87, 32), (106, 39), (193, 71), (1264, 465), (1457, 536)]]
    limit = 99

    for i in range(limit, limit + 1):
        result = Fraction(2)
        denominator = 0
        for j in range(i):
            n = Fraction(e[i - j])
            denominator = Fraction(numerator=Fraction(1), denominator=(n + denominator))
            # print("d", denominator, "n", n, i - j)
        result += denominator

        # print(result)
        return sum([int(x) for x in str(result.numerator)])


def problem_66():
    print(EulerRoutines.solve_pell_equation(2, 3, 4))


def problem_67():
    matrix = []
    with open("p067_triangle.txt", 'r') as fi:
        for line in fi.readlines():
            line = [int(x) for x in line.replace('\n', '').split()]
            matrix.append(line)

    # Possible transition: next index is equal to previous or previous + 1
    for i in range(len(matrix) - 2, -1, -1):
        for j in range(0, i + 1):
            matrix[i][j] += max(matrix[i + 1][j], matrix[i + 1][j + 1])

    return matrix[0][0]


def problem_68():
    solutions = []

    def check_solution(a, b, c, d, e, f, g, h, i, j):
        if a + b + c == d + c + e == f + e + g == h + g + i == j + i + b:
            return "".join(str(x) for x in (a, b, c, d, c, e, f, e, g, h, g, i, j, i, b))
        else:
            return False

    # TODO: należy uwzględnić powtarzanie permutacji i opis wyniku
    permutations = itertools.permutations(range(1, 11))
    for permutation in permutations:
        solution = check_solution(*permutation)
        if solution:
            solutions.append(solution)
    return max(solutions)


def problem_69():
    euler = EulerRoutines.euler_totient_function_array(10 ** 6)  # dict
    max_value = (1, 1)
    for key, value in euler.items():
        if max_value[0] / max_value[1] < key / value:
            max_value = (key, value)

    return max_value


def problem_74():
    from math import factorial
    factorials = {str(x): factorial(x) for x in range(0, 10)}
    results = []
    for number in range(69, 10 ** 6 + 1):
        chains = {number}
        while True:
            number = sum(factorials[x] for x in str(number))
            if number in chains:
                break
            chains.add(number)
        if len(chains) == 60:
            results.append(chains)

    return len(results)


def problem_79():
    with open("passcode.txt", 'r') as fi:
        passcodes = fi.readlines()
    passcode = [code.replace('\n', '') for code in passcodes]

    numbers = dict((n, []) for n in range(0, 10))

    for seq in passcode:
        for number in range(0, 10):
            for other in range(0, 10):
                if str(number) in seq and str(other) in seq and number != other:
                    if seq.index(str(number)) > seq.index(str(other)):
                        numbers[number].append(other)

    numbers_in_code = set()
    for i in numbers.keys():
        if numbers[i]:
            numbers_in_code.add(i)
            numbers_in_code.update(numbers[i])

    numbers = {k: set(v) for (k, v) in numbers.items() if k in numbers_in_code}
    code = [str(x) for x in sorted(numbers, key=numbers.get)]

    return "".join(code)


def problem_81():
    u"""Graph traversal problem."""
    matrix = np.loadtxt("p081_matrix.txt", delimiter=',', dtype=int)
    print(matrix)
    max_id = (len(matrix) - 1, len(matrix) - 1)

    def adjacent_vertices(x, y):
        u"""In this problem we move only down and right"""
        if 0 <= x < max_id[0] and 0 <= y < max_id[1]:
            vertices = ((x + 1, y), (x, y + 1))
        elif x == max_id[0] and y < max_id[1]:
            vertices = [(x, y + 1)]
        elif x < max_id[0] and y == max_id[1]:
            vertices = [(x + 1, y)]
        else:
            vertices = None
        return vertices

    current = (0, 0)

    total_value = 0
    while current != max_id:
        current_value = matrix[current]
        total_value += current_value
        values = {}
        adjacent = adjacent_vertices(*current)
        if adjacent:
            for vertex in adjacent:
                values[vertex] = total_value + matrix[vertex]
            current = min(values, key=lambda x: values[x])
        else:
            break

    return total_value


def problem_92():
    from SquareDigitChain import SquareDigitChain
    return SquareDigitChain.classify()


def problem96():
    u"""Sudoku solver implemented in my other repo at https://github.com/rpiw/LearnPython.
    Basically the easiest way to implement backtracking. Slow as hell"""
    from Sudoku import Sudoku
    input_file = "sudoku_problem96.txt"
    sudokus = Sudoku.read_from_file(input_file)

    def digit(sudoku: Sudoku):
        return int("".join(map(str, [sudoku.solution[0, i] for i in (0, 1, 2)])))

    for sudoku in sudokus:
        sudoku.solve()

    import pickle
    pickle.dump(sudokus, open("sudokus.obj", 'wb'))
    return sum(digit(sudoku) for sudoku in sudokus)


def problem_97():
    u"""Compute a number of a form: 28433 x 2 ^ 7830457  + 1"""
    factor_a = 28433
    base = 2
    factor_expontial = 7830457
    rest = 1
    l = 2357207
    print(EulerRoutines.is_prime(factor_a), EulerRoutines.is_prime(factor_expontial))
    number = 12345678901
    # number = factor_a * pow(base, factor_expontial) + rest
    return str(number)[:-10]


def problem99():
    from math import log
    numbers = []
    max_number = 0
    with open("problem99.txt", 'r') as fi:
        for line in fi.readlines():
            numbers.append(tuple(int(x) for x in line.replace('\n', '').split(',')))

    max_line = 0
    lenght = 0
    for i in range(len(numbers)):
        p = log(numbers[i][0], 2) * numbers[i][1]
        p_len = len(str(p))
        if p_len > lenght:
            max_number = p
            max_line = i
            lenght = len(str(p))
        elif p_len == lenght:
            if p > max_number:
                max_number = p
                max_line = i
                lenght = p_len
        else:
            continue

    return max_line + 1


def problem_401():
    def mod_of_square(number: int) -> np.array:
        return np.array([a ** 2 % number for a in range(0, number)], dtype=int)

    print(mod_of_square(6))


if __name__ == '__main__':
    from time import time

    start = time()
    print(problem_69())
    end = time() - start
    print(end)


def period(arr):
    u"""Find period in a sequence arr."""
    possible = set(range(1, len(arr) + 1))
    p, i = 1, 0
    while True:
        if np.array_equal(arr[i: i + p], arr[i + p: i + 2 * p]):
            i += 1
        else:
            possible.remove(p)
            p += 1
            i = 0
        if i + 2 * p == len(arr):
            return p
