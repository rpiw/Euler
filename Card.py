from functools import total_ordering, reduce


@total_ordering
class Card:
    _values_order = ("2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace".split(", "))
    _suits = ("H", "C", "S", "D")

    def __init__(self, value: str, suit: str = ""):
        if len(value) == 2 and suit == "":
            value, suit = value[0], value[1]
        if value not in Card._values_order or suit.upper() not in Card._suits:
            raise ValueError
        self.value = value
        self.suit = suit

    def __eq__(self, other):
        return self.as_number() == other.as_number()

    def __lt__(self, other):
        return self.as_number() < other.as_number()

    def __repr__(self):
        return "Card: {}, {}".format(self.value, self.suit)

    def as_number(self):
        return Card._values_order.index(self.value)

    @classmethod
    def values(cls):
        return cls._values_order

    @classmethod
    def suits(cls):
        return cls._suits

    @staticmethod
    def same_suits(cards: []):
        return all(x.suit == cards[0].suit for x in cards)


class Deck:

    def __init__(self, name=""):
        self.cards = [Card(val, suit) for val in Card.values() for suit in Card.suits()]
        self.name = name

    def __repr__(self):
        return "Deck: {}".format(self.name)

    def __str__(self):
        return " ".join(self.cards)


@total_ordering
class PokerHand:
    u"""Class representing hand in a card game poker: 5 cards with a rank of:
            High Card: Highest value card.
            One Pair: Two cards of the same value.
            Two Pairs: Two different pairs.
            Three of a Kind: Three cards of the same value.
            Straight: All cards are consecutive values.
            Flush: All cards of the same suit.
            Full House: Three of a kind and a pair.
            Four of a Kind: Four cards of the same value.
            Straight Flush: All cards are consecutive values of same suit.
            Royal Flush: Ten, Jack, Queen, King, Ace, in same suit.

        Initialize with a list of cards.
    """
    ranks = {"High Card": 0, "One Pair": 1, "Two Pairs": 2, "Three of a Kind": 3, "Straight": 4, "Flush": 5,
             "Full House": 6, "Four of a Kind": 7, "Straight Flush": 8, "Royal Flush": 9}

    def __init__(self, cards: [Card]):
        self.cards = cards
        self.rank = self.determine_hand_type()

    def as_number(self):
        return PokerHand.ranks.get(self.rank, int(self.rank) - 10)

    def __eq__(self, other):
        return self.as_number() == other.as_number()

    def __lt__(self, other):
        return self.as_number() < other.as_number()

    def highest_card(self):
        return max(self.cards)

    def __repr__(self):
        return "PokerHand at {}".format(id(self))

    def __str__(self):
        return "PokerHand of rank {rank}, highest card: {card}.".format(rank=self.rank, card=self.highest_card())

    def is_royal_flush(self) -> bool:
        u"""Return true if cards in a hand are of 'Royal Flush' rank."""
        if all(map(lambda c: c.value in ["10", "Jack", "Queen", "King", "Ace"], self.cards)) \
                and Card.same_suits(self.cards):
            return True
        return False

    def is_straight(self) -> bool:
        u"""Return true if cards in a hand are of 'Straight' rank."""
        self.cards.sort()
        if len(self.cards) != 5:
            return False
        for i in range(len(self.cards)):
            if self.cards[i + 1].as_number() - self.cards[i].as_number() != 1:
                return False
        return True

    def is_flush(self) -> bool:
        u"""Return true if cards in a hand are of 'Flush' rank."""
        return Card.same_suits(self.cards)

    def is_straight_flush(self):
        u"""Return true if cards in a hand are of 'Straight Flush' rank."""
        return self.is_straight() and self.is_flush()

    def _detect_repetition_of_kind(self, n, list_of_cards=[]):
        u"""Return true if n cards in a hand are of the same value."""
        results = []

        if list_of_cards:
            cards = list_of_cards
        else:
            cards = self.cards

        for card in cards:
            local_results = []
            for another_card in cards:
                if card != another_card and card.value == another_card.value:
                    local_results.append(True)
            results.append(local_results)

        for result in results:
            if result.count(True) == n:
                return result
        return []

    def is_full_house(self):
        u"""Return true if cards in a hand are of 'Full house' rank."""
        return bool(self._detect_repetition_of_kind(2)) and bool(self._detect_repetition_of_kind(3))

    def is_four_of_kind(self):
        u"""Return true if cards in a hand are of 'Four of a Kind' rank."""
        return bool(self._detect_repetition_of_kind(4))

    def is_three_of_kind(self):
        u"""Return true if cards in a hand are of 'Three of a kind' rank."""
        return bool(self._detect_repetition_of_kind(3))

    def is_pair(self):
        u"""Return true if cards in a hand are of 'Pair' rank."""
        return bool(self._detect_repetition_of_kind(2))

    def is_two_pair(self):
        u"""Return true if cards in a hand are of 'Two Pairs' rank."""
        first = self._detect_repetition_of_kind(2)
        if not first:
            return False
        cards = self.cards.copy()
        cards.remove(first[0])
        cards.remove(first[1])
        return bool(self._detect_repetition_of_kind(2, cards))

    def is_high_card(self):
        u"""Return true if cards in a hand are of 'High Rank' rank."""
        return not self.is_pair() and not self.is_two_pair() and not self.is_three_of_kind() and \
               not self.is_four_of_kind() and not self.is_full_house() and not self.is_flush() and \
               not self.is_straight()

    def determine_hand_type(self):
        if self.is_royal_flush():
            rank = "Royal Flush"
        elif self.is_straight_flush():
            rank = "Straight Flush"
        elif self.is_straight():
            rank = "Straight"
        elif self.is_flush():
            rank = "Flush"
        elif self.is_full_house():
            rank = "Full House"
        elif self.is_four_of_kind():
            rank = "Four of a kind"
        elif self.is_three_of_kind():
            rank = "Three of a kind"
        elif self.is_two_pair():
            rank = "Two pairs"
        elif self.is_pair():
            rank = "Pair"
        else:
            rank = self.highest_card()
        return rank
