import unittest
from main import EulerRoutines
import numpy as np


class TestEulerRoutines(unittest.TestCase):

    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                       61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
                       131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                       193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                       263, 269, 271], dtype=int)

    def test_primes(self):
        my_primes = EulerRoutines.primes(272)
        for p1, p2 in zip(TestEulerRoutines.primes, my_primes):
            self.assertEqual(p1, p2)
        self.assertEqual(len(my_primes), len(EulerRoutines._primes))

    def test_is_prime(self):
        for p in TestEulerRoutines.primes:
            self.assertTrue(EulerRoutines.is_prime(p))


if __name__ == '__main__':
    unittest.main()
