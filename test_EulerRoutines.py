import unittest
from main import EulerRoutines, Numeral
import numpy as np


class TestEulerRoutines(unittest.TestCase):

    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                       61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
                       131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                       193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                       263, 269, 271], dtype=int)

    def test_primes_debug(self):
        u"""Smaller sample set - for debugging purpose"""
        primes = TestEulerRoutines.primes
        my_primes = EulerRoutines.primes(272, read=False)
        for correct, excepted in zip(primes, my_primes):
            self.assertEqual(correct, excepted)

    def test_primes(self):
        correct_primes = np.loadtxt("primes_correct.txt", dtype=int).flatten()

        new_len = int(1/10 * len(correct_primes))
        my_primes = EulerRoutines.primes(correct_primes[new_len])

        self.assertEqual(len(correct_primes[:new_len]), len(my_primes))
        for p1, p2 in zip(correct_primes, my_primes):
            self.assertEqual(p1, p2)

    def test_is_prime(self):
        for p in TestEulerRoutines.primes:
            self.assertTrue(EulerRoutines.is_prime(p))

    def test_permute(self):
        good = {1: [1], 12: [12, 21], 123: [123, 132, 312, 321, 213, 231]}
        for key in good.keys():
            values = np.array(sorted([str(x) for x in good[key]]), int)
            self.assertTrue(np.array_equal(EulerRoutines.permute(key), values))

    def test_is_pandigital(self):
        numbers_true = np.array([1, 12, 21, 123, 132, 321, 1234, 1243, 3421, 2134, 123456789])
        numbers_false = np.array([11, 122, 13, 14212342114242134123243545])

        for good in numbers_true:
            self.assertTrue(EulerRoutines.is_pandigital(good), msg=str(good))
        for bad_number in numbers_false:
            self.assertFalse(EulerRoutines.is_pandigital(bad_number), msg=str(bad_number))

    def test_divisors(self):
        div28 = np.array([1, 2, 4, 7, 14, 28], dtype=int)
        div100 = np.array([1, 2, 4, 5, 10, 20, 25, 50, 100])
        self.assertTrue(np.array_equal(div28, EulerRoutines.divisors(28)))
        self.assertTrue(np.array_equal(div100, EulerRoutines.divisors(100)))

    def test_number_name(self):
        # Examples from numerals.txt for sure must pass test, so, let we starts with them
        names = Numeral.numerals()
        # Add some more samples
        names[342] = "three hundred and forty-two"
        names[115] = "one hundred and fifteen"
        for number in names.keys():
            if number < 1000:
                self.assertEqual(EulerRoutines.number_name(number), names[number])
            else:
                with self.assertRaises(ValueError):
                    EulerRoutines.number_name(number)

    def test_rotate_digits(self):
        expected = set((197, 971, 719))
        actual = EulerRoutines.rotate_digits(197)
        self.assertEqual(expected, actual)


class TestNumeral(unittest.TestCase):

    def test_numerals_init(self):
        numbers = (1, 12, 34, 64, 109, 345)
        for n in numbers:
            numeral = Numeral(n)
            self.assertEqual(int("".join(map(lambda x: str(x), numeral.factors))),
                             n)

    def test_numerals_name(self):
        numbers = (1, 12, 34, 64, 109, 345)
        for n in numbers:
            numeral = Numeral(n)
            self.assertEqual(str(n), numeral.name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
