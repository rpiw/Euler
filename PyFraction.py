# Class extending python STL Fraction class for purpose in Project Euler
import math
from fractions import Fraction
import numpy as np
from typing import Union, Iterable

from main import period


class PyFraction(Fraction):
    u"""Represent fraction."""
    max_iterations = 1024

    def __init__(self, *args, **kwargs):
        super(PyFraction, self).__init__()
        self._continued_fraction_representation = None

    @property
    def continued_fraction_representation(self):
        if self._continued_fraction_representation is not None:
            return self._continued_fraction_representation
        else:
            return self.continued_fraction()

    def continued_fraction(self, max_iterations=max_iterations):
        u"""Represent a fraction as sequence of integers."""
        self._continued_fraction_representation = []
        approximation = int(self)
        fraction = self
        i = 0
        while True:
            fraction = (fraction - approximation) ** -1
            self._continued_fraction_representation.append(approximation)
            approximation = int(fraction)
            if fraction == int(fraction):
                self._continued_fraction_representation.append(approximation)
                break
            elif i == max_iterations:
                break
            else:
                i += 1

        return self._continued_fraction_representation

    def convergents(self):
        u"""Return convergents of fraction up to i step."""
        # First 3 continued fractions are needed
        fraction = self.continued_fraction_representation
        if len(fraction) < 3:
            raise ValueError

        numerators = [fraction[0], fraction[0] * fraction[1] + 1,
                      fraction[2] * (fraction[0] * fraction[1] + 1) + fraction[0]]
        denominators = [1, fraction[1], fraction[1] * fraction[0] + 1]

        for n in range(3, len(fraction)):
            numerators.append(fraction[n] * numerators[n - 1] + numerators[n - 2])
            denominators.append(fraction[n] * denominators[n - 1] + denominators[n - 2])

        return tuple((h, k) for h, k in zip(numerators, denominators))


def continued_fraction_expansion(number: Union[int, Iterable]):
    u""" Let we denote rational number D with irrational square root. This function compute continued fraction expansion
    of an irrational square root of the given number D. If given number is a square, result is just a root of the number
    with an empty expansion.

    :arg number - integer or iterable of integers, but for sake of efficiency types of items in iterable is not checked!

    Do not use this function to compute continued fraction expansion of irrational numbers D - it will never ends!
    """
    if not isinstance(number, Iterable):
        if isinstance(number, int):
            numbers = (number,)
        else:
            raise TypeError
    else:
        numbers = number

    results = {}

    for number in numbers:
        if math.isqrt(number) == math.sqrt(number):
            results[number] = (number, ())
            continue

        a0 = math.isqrt(number)
        b, c = a0, number - a0 ** 2
        expansions = []
        an = 0
        while an != a0 * 2:
            an = math.floor((a0 + b) / c)
            b = an * c - b
            c = (number - b ** 2) / c
            expansions.append(an)

            results[number] = (a0, expansions)
    return results


def infinite_continued_fraction(number=2, iterations: int = 10):
    u"""Infinite continued fraction expansion.
        Fraction = starter + 1 / (number + ...)"""
    fraction = PyFraction(1, number)
    results = []
    for i in range(iterations):
        results.append((i + 1, fraction, fraction + 1))
        fraction = PyFraction(1, number + fraction)
    return results


if __name__ == '__main__':
    pass
    # p = PyFraction(415, 93)
    # print(p.convergents())
    # infinite_continued_fraction(2, 10)
    print(continued_fraction_expansion(16))
