import json

import ujson


def test1():
    # dict <-> json字符串
    data = {
        "name": "骆昊",
        "age": 40,
        "friends": ["王大锤", "白元芳"],
        "cars": [
            {"brand": "BMW", "max_speed": 240},
            {"brand": "Benz", "max_speed": 280},
            {"brand": "Audi", "max_speed": 280}
        ]
    }
    print(data)

    print(json.dumps(data))
    print(json.dumps(data, ensure_ascii=False, indent=4))

    # 写入文件
    json.dump(data, fp=open("pa2.json", "w"), ensure_ascii=False, indent=4)

    print('---------')

    # 加载文件里面的json数据
    data1 = json.load(open("pa2.json", "r"))
    print(data1)
    with open('pa2.json', 'r') as file:
        my_dict = json.load(file)
        print(type(my_dict))  # <class 'dict'>
        print(my_dict)

    pass


class Card:
    def __init__(self, brand, max_speed):
        self.brand = brand
        self.max_speed = max_speed
    # json.dumps() 可以序列化的类型：dict、list、str、int、float、bool、None
    # 自定义对象必须先转换成这些类型。使用以下to_dict方法

    def to_dict(self):
        return {"brand": self.brand, "max_speed": self.max_speed}


class JsonObj:
    def __init__(self, name, age, friends, cars):
        self.name = name
        self.age = age
        self.friends = friends
        self.cars = cars

    def to_dict(self):
        return {
            "name": self.name,
            "age": self.age,
            "friends": self.friends,
            "cars": [c.to_dict() for c in self.cars]
        }

def test2():
    # obj <-> json字符串

    c1 = Card("BMW", 240)
    c2 = Card("Benz", 280)
    c3 = Card("Audi", 280)
    jb = JsonObj("骆昊", 40, ["王大锤", "白元芳"], [c1, c2, c3])

    json_str = json.dumps(jb.to_dict(), ensure_ascii=False, indent=4)
    print(json_str)

    # 写入文件
    with open("pa3.json", "w", encoding="utf-8") as f:
        f.write(json_str)


    pass


def test1_ujson():
    # dump - 将Python对象按照JSON格式序列化到文件中
    # load - 将文件中的JSON数据反序列化成对象

    data = {
        "name": "骆昊",
        "age": 40,
        "friends": ["王大锤", "白元芳"],
        "cars": [
            {"brand": "BMW", "max_speed": 240},
            {"brand": "Benz", "max_speed": 280},
            {"brand": "Audi", "max_speed": 280}
        ]
    }
    print(data)

    file = open("pa12.json", "w", encoding="utf-8")

    ujson.dump(data, file, ensure_ascii=False, indent=4)

    file.close()

    file = open("pa12.json", "r", encoding="utf-8")

    data2 = ujson.load(file)
    print(data2)

    file.close()


def test2_ujson():
    # dumps - 将Python对象处理成JSON格式的字符串
    # loads - 将字符串的内容反序列化成Python对象


    c1 = Card("BMW", 240)
    c2 = Card("Benz", 280)
    c3 = Card("Audi", 280)
    jb = JsonObj("骆昊", 40, ["王大锤", "白元芳"], [c1, c2, c3])

    # s = ujson.dumps(jb, ensure_ascii=False, indent=4)
    s = ujson.dumps(jb.to_dict(), ensure_ascii=False, indent=4)
    print(s)

    jb2_str = ujson.loads(s)

    jb2_cars = [Card(item["brand"], item["max_speed"]) for item in jb2_str["cars"]]
    jb2 = JsonObj(jb2_str["name"], jb2_str["age"], jb2_str["friends"], jb2_cars)

    print(jb2)

    pass


if __name__ == '__main__':
    # 对象的序列化和反序列化 , 需要用到json包， import json
    # json模块有四个比较重要的函数，分别是：
    #
    # dump - 将Python对象按照JSON格式序列化到文件中
    # dumps - 将Python对象处理成JSON格式的字符串
    # load - 将文件中的JSON数据反序列化成对象
    # loads - 将字符串的内容反序列化成Python对象



    # test1()

    # test2()

    # test1_ujson()

    test2_ujson()

    pass
