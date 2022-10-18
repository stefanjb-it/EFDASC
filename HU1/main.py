import math
import pandas as pd


def BSP_1():
    print("---BSP1---")
    inventory = {"Apfel": 3, "Banane": 5, "Obstkorb": ["Banane", "Apfel", "Birne"]}
    print(inventory)
    inventory["Weintraube"] = 4
    print(inventory)
    inventory["Obstkorb"].append("Weintraube")
    print(inventory)
    inventory.update({"Obstkorb": sorted(inventory.get("Obstkorb"))})
    print(inventory)
    inventory.update({"Banane": (inventory.get("Banane") + 10)})
    print(inventory)


def BSP_2():
    print("---BSP2---")
    list_1 = list(range(1, 5))
    list_2 = [list(range(1, 5))]
    print("Länge Liste [1,2,3,4]: ", len(list_1))
    print("Länge Liste [[1,2,3,4]]: ", len(list_2))
    list_3 = [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))]
    print(list_3)
    list_4 = list(range(1, 7))
    list_4[2] = "TEST"
    print(list_4)


def BSP_3():
    print("---BSP3---")
    list_1 = [1, 3, 3, 2, 4, 6, 'd', 'b', 'a']
    print(list_1)
    # print("Summe: ", sum(list_1)) --> ERROR
    # print("Min: ", min(list_1)) --> ERROR
    # print("Max: ", max(list_1)) --> ERROR
    list_2 = [x for x in list_1.copy() if isinstance(x, int)]
    print(list_2)
    print("Summe: ", sum(list_2))
    print("Min: ", min(list_2))
    print("Max: ", max(list_2))
    list_3 = [x for x in list_1.copy() if not isinstance(x, int)]
    print(list_3)
    list_4 = sorted(list_2.copy())
    print(list_4)
    list_5 = list_3.copy()
    list_5.reverse()
    print(list_5)
    list_6 = [x for x in list_4.copy() if x < 5]
    print(list_6)
    list_7 = ['a', 'c', 'e']
    # list_5 = [(x, y) for x in list_5.copy() for y in list_7.copy()]
    for x in list_7:
        for i, y in enumerate(list_5):
            if x < y:
                list_5.insert(i, x)
                break
            elif (x > y) & (i == len(list_5) - 1):
                list_5.append(x)
                break
            elif x > y:
                continue
            elif x == y:
                break
    print(list_5)


def check_prime(num):
    temp = 2
    while temp <= math.sqrt(num):
        if num % temp < 1:
            return "num is not a prime number"
        temp += 1
    return "num is a prime number"


def convert_temperature(x, flag1, flag2):
    if flag1 == flag2:
        return "No conversion necessary"
    if flag1 == "C":
        if flag2 == "K":
            return x + 273.15
        elif flag2 == "F":
            return (x * (9 / 5)) + 32
    elif flag1 == "K":
        if flag2 == "C":
            return x - 273.15
        elif flag2 == "F":
            return ((x - 273.15) * (9 / 5)) + 32
    elif flag1 == "F":
        print()
        if flag2 == "K":
            return ((x - 32) * (5 / 9)) + 273.15
        elif flag2 == "C":
            return (x - 32) * (5 / 9)


def BSP_7():
    print("---BSP7---")
    list_1 = ["HELLO", "WORLD", 1, 2, 3, 4, 2, "TEST", 3, 'a', 4, 5, 5, 9, 75, 4, 2221, 34, 5, 5]
    print(list_1)
    list_2 = [x for x in list_1.copy() if isinstance(x, str)]
    print(list_2)
    list_3 = [x * x for x in range(4, 12) if x % 2 == 0]
    print(list_3)
    list_4 = [list(range(1, 43)), list(range(5, 77)), list(range(9, 45))]
    print(list_4)
    list_5 = [y for i, x in enumerate(list_4) for k, y in enumerate(list_4[i]) if k == 1]
    print(list_5)


def BSP_8():
    print("---BSP8---")
    df_1 = pd.read_csv('./data/menu.csv')
    df_1.dropna()
    df_2 = df_1[['Calories', 'Calories from Fat', 'Total Fat', 'Saturated Fat (% Daily Value)', 'Trans Fat',
                 'Cholesterol', 'Cholesterol (% Daily Value)', 'Sodium', 'Sodium (% Daily Value)', 'Sugars',
                 'Protein', 'Vitamin C (% Daily Value)', 'Calcium (% Daily Value)']]
    df_3 = df_1[['Category', 'Item', 'Serving Size']]
    print(df_2.head())
    print(df_2.describe())
    print(df_2.info())
    print(df_3['Category'].unique())
    print(df_3['Category'].nunique())
    df_4 = df_2.copy()
    df_5 = df_4[(df_4['Saturated Fat (% Daily Value)'] == 100) | (df_4['Cholesterol (% Daily Value)'] == 100) |
                (df_4['Sodium (% Daily Value)'] == 100) | (df_4['Vitamin C (% Daily Value)'] == 100) |
                (df_4['Calcium (% Daily Value)'] == 100)]
    print(df_5)
    df_5 = df_2[(df_4['Saturated Fat (% Daily Value)'] >= 100) | (df_4['Cholesterol (% Daily Value)'] >= 100) |
                (df_4['Sodium (% Daily Value)'] >= 100) | (df_4['Vitamin C (% Daily Value)'] >= 100) |
                (df_4['Calcium (% Daily Value)'] >= 100)]
    print(df_5)
    print(df_3._get_value(df_2['Calories'].idxmax(), 'Item'))
    print(df_3._get_value(df_2['Calories'].idxmin(), 'Item'))
    cat_dict = dict(zip(df_3['Category'].unique(), [[] for _ in range(1, len(df_3['Category'].unique()) + 1)]))
    [cat_dict[y].append(i) for i, y in enumerate(df_3['Category'])]
    print(cat_dict)
    for j in cat_dict:
        for i, k in enumerate(cat_dict.get(j)):
            cat_dict[j][i] = df_2['Total Fat'].iloc[k]
    print(cat_dict)
    result_dict = dict(zip(df_3['Category'].unique(), [0 for _ in range(1, len(df_3['Category'].unique()) + 1)]))
    for index_1 in cat_dict:
        for p in cat_dict[index_1]:
            result_dict.update({index_1: (result_dict.get(index_1) + p)})
        result_dict.update({index_1: (result_dict.get(index_1) / len(cat_dict[index_1]))})
    print(result_dict)
    print(max(result_dict, key=result_dict.get))


def prime_twins(n):
    list_1 = [num for num in range(1, n) if check_prime(num) == "num is a prime number"]
    list_2 = []
    for index, x in enumerate(list_1[1:]):
        if x - list_1[index] == 2:
            list_2.append((list_1[index], x))
    print(list_2)


if __name__ == '__main__':
    BSP_1()
    BSP_2()
    BSP_3()
    BSP_7()
    BSP_8()
    prime_twins(556)
