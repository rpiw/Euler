# Utilities for dealing with primes
import numpy as np
import logging
import time


logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
logger.addHandler(sh)


def timer(function):
    u"""Measure time and print fancy output to log. Use as decorator."""
    def inner(*args, **kwargs):
        time_start = time.time()
        result = function(*args, **kwargs)
        time_end = time.time()
        logger.info(f"Completed {function.__name__} in {time_end - time_start}")
        return result
    return inner


@timer
def sieve_of_sundaram(upper_limit: int) -> np.array:
    u"""Generate primes from 2 to upper_limit. Algorithm: sieve of Atking."""
    k = (upper_limit - 2) // 2
    numbers = set(range(1, upper_limit))
    for i in range(1, k + 1):
        j = 1
        while i + j + 2 * i * j <= k:
            numbers.discard(i + j + 2 * i * j)
            j += 1
    return np.append([2], np.fromiter((2 * x + 1 for x in numbers), dtype=int))


@timer
def sieve_of_eratosthenes(upper_limit: int) -> np.array:
    u"""Generate primes from 2 to upper_limit. Algorithm: sieve of Eratosthenes."""
    sieve = {x: True for x in range(2, upper_limit + 1)}
    n = 2
    while n * n <= upper_limit:
        if sieve[n]:
            i = 2
            k = n * i
            while k <= upper_limit:
                sieve[k] = False
                i += 1
                k = n * i
        n += 1
    return np.array([x for x in sieve.keys() if sieve[x]], dtype=int)
