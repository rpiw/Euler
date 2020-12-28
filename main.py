import functools
import itertools
import math

import numpy as np
from typing import Union


class EulerRoutines:
    _primes = np.array([], dtype=int)  # cache primes
    _primes_set = set()

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
        if EulerRoutines._primes.size > 0 and number < EulerRoutines._primes[-1]:
            return number in EulerRoutines._primes_set
        for d in range(2, np.int(np.ceil(np.sqrt(number))) + 1):
            if number % d == 0 and number != d:
                return False
        return True

    _primes_file_name = "primes.npy"
    _sieve = "sieve.obj"  # TODO: cache also dict for sieve - if not, cannot use concatenation

    @staticmethod
    def primes(upper_limit: int, algorithm="sieve", cache=True, read=False) -> np.array:
        u"""Return array containing primes below upper_limit integer. Algorithm: brute, sieve.
        : cache (bool) - is true store primes in an array as a class member EulerRoutines._primes,
        : read (bool) - DOES NOT WORK."""

        start = 3
        if read:
            loaded = np.array([])
            try:
                loaded = np.load(EulerRoutines._primes_file_name)
            except FileNotFoundError:
                pass

            if loaded.size > 0:
                EulerRoutines._primes = np.unique(np.concatenate((EulerRoutines._primes, loaded), 0))
                print("Primes loaded.")

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
            print("Space: [{start}, {end}]".format(start=start, end=upper_limit))
            for number in results:
                if sieve[number] and EulerRoutines.is_prime(number):
                    k = number
                    while k <= upper_limit:
                        k += number
                        sieve[k] = False

            results = np.fromiter((x for x in sieve.keys() if sieve[x]), dtype=int)
            results = np.append(np.array([2], dtype=int), results)

        if cache:
            print("Caching...")
            EulerRoutines._primes = results
            EulerRoutines._primes_set = set(results)
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
    # def recurring_fraction(p: int, q: int, base: int=10):
    #     digits = [str(x) for x in range(0, base)]
    #
    #     occurs = {}
    #
    #     string_of_digits = ""
    #     position = 0
    #
    #     while not occurs.get(p, None):
    #         occurs[p] = position  # the position of place with reminader p
    #         bp = base * p
    #         z = math.floor(bp / q)  # index z of digit within: 0 ≤ z ≤ b-1
    #         p = bp - z * q


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


def problem_26():
    results = []
    return results


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
    numbers = dict((n, EulerRoutines.pentagonal_number(n)) for n in range(1, 10 ** 6))
    results = {}

    def func():
        for i in range(1, len(numbers)):
            for j in range(i + 1, len(numbers)):
                third = (numbers[i] + numbers[j], numbers[i] - numbers[j])
                if third in numbers.values():
                    results[(i, j)] = third
                if third[0] > numbers[10 ** 6 - 1]:
                    return

    func()
    best = 10 ** 5
    for key in results.keys():
        if results[key] < best:
            best = results[key]

    return best


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


def problem_49():
    primes = np.fromiter(filter(lambda x: len(str(x)) == 4, EulerRoutines.primes(10000)), dtype=int)
    results = set()
    for prime in primes:
        permutations = EulerRoutines.permute(prime)
        local_results = []
        for p in set(permutations):
            if p in primes:
                local_results.append(p)
        local_results.sort()
        if len(local_results) >= 3:
            results.add(tuple(local_results))
        # if len(local_results) >= 3:
        #     increments = []
        #     for i in range(len(local_results) - 1):
        #         if increments.count(local_results[i + 1] - local_results[i]) == 2:
        #             results.append(local_results)
        #             break
        #         else:
        #             increments.append(local_results[i + 1] - local_results[i])

    return sorted(results)


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
    matrix = np.loadtxt("p081_matrix.txt", delimiter=',')
    print(matrix)


def problem_92():
    from SquareDigitChain import SquareDigitChain
    return SquareDigitChain.classify()


def problem96():
    u"""sudoku!"""
    with open("sudoku_problem96.txt", 'r') as fi:
        sudokus = fi.readlines()
    sudokus = list((x.replace("\n", '') for x in sudokus if "Grid" not in x))
    sudokus = list(sudokus[i: i + 9] for i in range(0, len(sudokus), 9))
    new_sudokus = []
    for sudoku in sudokus:
        new_sudoku = []
        for line in sudoku:
            line = list(int(x) for x in line)
            new_sudoku.append(line)
        new_sudokus.append(new_sudoku)
    print(new_sudokus)


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


if __name__ == '__main__':
    print(problem_46())
