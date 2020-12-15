from unittest import TestCase
from SquareDigitChain import SquareDigitChain


class TestSquareDigitChain(TestCase):
    def test_square_digit_chain(self):
        excepted = [85, 89, 145, 42, 20, 4, 16, 37, 58]
        excepted2 = [44, 32, 13, 10, 1]
        self.assertEqual(excepted, SquareDigitChain(85).chain)
        self.assertEqual(89, SquareDigitChain(85).end)
        self.assertEqual(excepted2, SquareDigitChain(44).chain)
        self.assertEqual(1, SquareDigitChain(44).end)
