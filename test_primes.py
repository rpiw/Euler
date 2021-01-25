from unittest import TestCase, expectedFailure
import numpy as np
import primes


class Test(TestCase):
    primes = np.loadtxt("primes_correct.txt", dtype=int).flatten()

    @expectedFailure
    def test_sieve_of_atkin(self):
        self.fail()

    def test_sieve_of_sundaram(self):
        upper_limit = 15485863 + 1
        for correct, calculated in zip(Test.primes, primes.sieve_of_sundaram(upper_limit)):
            self.assertEqual(correct, calculated, msg=f"{correct}, {calculated}")