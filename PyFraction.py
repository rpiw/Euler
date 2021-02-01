# Class extending python STL Fraction class for purpose in Project Euler
from fractions import Fraction


class PyFraction(Fraction):
    u"""Represent fraction."""
    max_iterations = 1024

    def __init__(self, *args, **kwargs):
        super(PyFraction, self).__init__()
        self.continued_fraction_representation = None

    def continued_fraction(self, max_iterations=max_iterations):
        u"""Represent a fraction as sequence of integers."""
        self.continued_fraction_representation = []
        approximation = int(self)
        fraction = self
        i = 0
        while True:
            fraction = (fraction - approximation) ** -1
            self.continued_fraction_representation.append(approximation)
            approximation = int(fraction)
            if fraction == int(fraction):
                self.continued_fraction_representation.append(approximation)
                break
            elif i == max_iterations:
                break
            else:
                i += 1

        return self.continued_fraction_representation
