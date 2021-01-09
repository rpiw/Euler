# Utilities for dealing with primes
import numpy as np
import logging
import time


logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
logger.addHandler(sh)


def sieve_of_atkin(upper_limit: int) -> np.array:
    u"""Generate primes from 2 to upper_limit. Algorithm: sieve of Atkin."""
    results = np.array([2, 3, 5], dtype=int)
    sieve = np.arange(3, upper_limit, 2)
    remainders = np.array([1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59], dtype=int)
    is_prime = {False: number for number in sieve}
    raise NotImplemented


def sieve_of_sundaram(upper_limit: int) -> np.array:
    u"""Generate primes from 2 to upper_limit. Algorithm: sieve of Atking."""
    time_start = time.time()
    k = (upper_limit - 2) // 2
    numbers = set(range(1, upper_limit))
    for i in range(1, k + 1):
        j = 1
        while i + j + 2 * i * j <= k:
            numbers.discard(i + j + 2 * i * j)
            j += 1
    time_end = time.time()
    logger.info(f"Completed sieve_of_sundaram for limit {upper_limit} in time of: {time_end - time_start}")
    return np.append([2], np.fromiter((2 * x + 1 for x in numbers), dtype=int))


if __name__ == '__main__':
    print(sieve_of_sundaram(10))