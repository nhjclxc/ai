import random

# 我们的扑克只有52张牌（没有大小王），游戏需要将 52 张牌发到 4 个玩家的手上，每个玩家手上有 13 张牌，
# 按照黑桃、红心、草花、方块的顺序和点数从小到大排列，暂时不实现其他的功能。

# 定义所有花色
SUITES = '♠♥♣♦'


# 定义所有牌面
FACES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

suit_order = {'♠': 0, '♥': 1, '♣': 2, '♦': 3}
face_order = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
              '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

class Card:
    """ 定义牌类 """

    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    @staticmethod
    def shuffle():
        """ 生成并打乱所有卡牌(洗牌) """

        cards = [Card(suite, face) for suite in SUITES for face in FACES]
        random.shuffle(cards)
        return cards

    @staticmethod
    def distribution(persons, cards):
        """ 分发所有卡牌 """

        index = 0
        for card in cards:
            # persons[index].add_card(card)
            persons[index].add_card2(card)
            index += 1
            if index == len(persons):
                index = 0

        print("所有卡牌分发完毕！")

    def __str__(self):
        return str(self.suite) + str(self.face)


class Person:
    """ 定义人 """

    def __init__(self, name):
        self.name = name
        self.cards = []

    def add_card(self, card):
        """ 仅接收手牌 """
        self.cards.append(card)

    def add_card2(self, card):
        """ 接收的同时整理手牌 """
        self.cards.append(card)
        self.cards.sort(key=lambda card: (suit_order[card.suite], face_order[card.face]))

    def print_cards(self):
        cards = [card.__str__() for card in self.cards]
        print(f"{self.name} 持有的卡牌为: {len(cards)}, {cards}")


    def print_cards2(self):
        # 排序
        self.cards.sort(key=lambda card: (suit_order[card.suite], face_order[card.face]))

        # 打印
        cards_str = [str(card) for card in self.cards]
        print(f"{self.name} 持有的卡牌({len(cards_str)}张): {cards_str}")

    def arrange(self):
        """整理手上的牌"""

        self.cards.sort(key=lambda card: (suit_order[card.suite], face_order[card.face]))

if __name__ == '__main__':
    p1 = Person('东邪')
    p2 = Person('西毒')
    p3 = Person('南帝')
    p4 = Person('北丐')

    # 生成卡牌
    cards = Card.shuffle()
    Card.distribution([p1, p2, p3, p4], cards)
    p1.print_cards()
    p2.print_cards()
    p3.print_cards()
    p4.print_cards()
    p4.print_cards2()

    p2.arrange()
    p2.print_cards()


