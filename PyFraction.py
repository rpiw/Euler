# Class extending python STL Fraction class for purpose in Project Euler
from fractions import Fraction


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


if __name__ == '__main__':
    p = PyFraction(415, 93)
    print(p.convergents())
