class SquareDigitChain:

    def __init__(self, number):
        self.number = number
        self.end = None
        self.chain = self.square_digit_chain()

    def __repr__(self):
        return f"SquareDigitChain: start ${self.number}, end ${self.end}."

    def square_digit_chain(self) -> list[int]:
        u"""Create a square digit chain: next number is created as a sum of squares of digits in number.
            Every chain ends when a created number is repeated. The last number always is 1 or 89."""
        results = {1, 89}
        results_list = []
        current = self.number
        while current not in results:
            results.add(current)
            results_list.append(current)
            current = SquareDigitChain.sum_of_powers(current)
        else:
            self.end = current
            results_list.append(current)
        return results_list

    @staticmethod
    def sum_of_powers(number: int) -> int:
        return int(sum(pow(int(digit), 2) for digit in str(number)))

    @classmethod
    def classify(cls, upper_limit=10**7):
        u"""Classify all number from 1 to upper_limit by end of chain: 1 or 89"""
        digits_ends_with_89 = set()
        # Calculate maximum non-repeating number
        maximum = SquareDigitChain.sum_of_powers(int("".join([str(9) for _ in range(len(str(upper_limit)))])))
        # Classify all numbers from 1 to maximum
        for i in range(1, maximum + 1):
            sdc = SquareDigitChain(i)
            if sdc.end == 89:
                digits_ends_with_89.add(i)
        for i in range(maximum, upper_limit):
            powers = SquareDigitChain.sum_of_powers(i)
            if powers in digits_ends_with_89:
                digits_ends_with_89.add(i)
        return len(digits_ends_with_89)
