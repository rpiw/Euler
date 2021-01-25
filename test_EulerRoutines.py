import unittest
from main import EulerRoutines, Numeral
import numpy as np
from Card import Card, PokerHand


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

        my_primes = EulerRoutines.primes(correct_primes[-1] + 1, read=False)
        self.assertEqual(len(correct_primes), len(my_primes))
        for p1, p2 in zip(correct_primes, my_primes):
            self.assertEqual(p1, p2)

        # check reading from file - if new upper_limit is lower than previous, method should read from file and returns
        # the results
        new_len = len(correct_primes) // 3
        my_primes = EulerRoutines.primes(correct_primes[new_len], read=True)
        correct_primes = np.array([x for x in correct_primes if x <= my_primes[-1]])
        self.assertEqual(len(correct_primes), len(my_primes))
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
        expected = {197, 971, 719}
        actual = EulerRoutines.rotate_digits(197)
        self.assertEqual(expected, actual)

    def test_is_palindromic(self):
        expected_true = (585, 101, 121, 1313131, 12345678987654321)
        expected_false = (112, 121211111, 433434333333)
        for n in expected_true:
            self.assertTrue(EulerRoutines.is_palindromic(n))
        for n in expected_false:
            self.assertFalse(EulerRoutines.is_palindromic(n))

    def test_two_base_palindromic(self):
        expected_true = [585]
        for n in expected_true:
            self.assertTrue(EulerRoutines.two_base_palindromic(n))

    def test_truncate_numbers(self):
        sample = 12345678
        expected_results = np.array([12345678, 2345678, 345678, 45678, 5678, 678, 78, 8,
                                     1234567, 123456, 12345, 1234, 123, 12, 1], dtype=int)
        self.assertTrue(np.array_equal(EulerRoutines.truncate_numbers(sample), expected_results))

    def test_concatenated_product(self):
        sample = (192, 3)
        expected_result = "192384576"
        self.assertEqual(expected_result, EulerRoutines.concatenated_product(*sample))

    def test_reverse_number(self):
        sample = (123, 1, 456, 123456789)
        expected_results = (321, 1, 654, 987654321)
        for s, res in zip(sample, expected_results):
            print(s, res)
            self.assertEqual(res, EulerRoutines.reverse_number(s))

    def test_traverse_matrix(self):
        matrix = np.loadtxt("p081_matrix.txt", delimiter=',')
        EulerRoutines.traverse_matrix(matrix)
        self.fail("Not implemented yet")

    def test_gcd(self):
        expected_output = 6
        self.assertEqual(expected_output, EulerRoutines.gcd(24, 30, 36))

    def test_coprimes(self):
        self.assertTrue(EulerRoutines.coprimes(3, 248))
        self.assertFalse(EulerRoutines.coprimes(3, 6, 9))

    def test_factorize(self):
        self.assertEqual(((3, 1), (5, 1)), EulerRoutines.factorize(15))

    def test_euler_totient_function(self):
        expected_outputs = [1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8, 12, 10, 22, 8, 20, 12, 18,
                            12, 28, 8, 30, 16, 20, 16, 24, 12, 36, 18, 24, 16, 40, 12, 42, 20, 24, 22, 46, 16, 42, 20,
                            32, 24, 52, 18, 40, 24, 36, 28, 58, 16, 60, 30, 36, 32, 48, 20, 66, 32, 44]
        for output, n in zip(expected_outputs, range(2, 101)):
            self.assertAlmostEqual(output, EulerRoutines.euler_totient_function(n), msg=f"n={n}")


class TestNumeral(unittest.TestCase):

    def test_numerals_init(self):
        numbers = (1, 12, 34, 64, 109, 345)
        for n in numbers:
            numeral = Numeral(n)
            self.assertEqual(int("".join(map(lambda x: str(x), numeral.factors))),
                             n)

    def test_numerals_name(self):
        numbers = (1, 12, 34, 64, 109, 345)
        correct = ("one", "twelve", "thirty-four", "sixty-four", "one hundred and nine", "three hundred and forty-five")
        for name, n in zip(correct, numbers):
            numeral = Numeral(n)
            self.assertEqual(name, numeral.name)


class TestEulerRoutinesSettings(unittest.TestCase):
    from main import EulerRoutinesSettings

    def test_init(self):
        settings = TestEulerRoutinesSettings()
        print(settings)


if __name__ == '__main__':
    unittest.main(verbosity=2)
