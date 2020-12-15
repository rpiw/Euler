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
            current = int(sum(pow(int(digit), 2) for digit in str(current)))
        else:
            self.end = current
            results_list.append(current)
        return results_list
