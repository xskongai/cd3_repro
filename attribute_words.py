#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/13/26
Description: attribute_words
 # 论文 5.1.1 的 gender 敏感词对(10 组)
 敏感属性词表 (Sensitive Attribute Words)
========================================
来源: Li et al. 2024 (CD3), 论文 5.1.1 节, 完全照抄它给的 gender 词对。
论文原文引用了 Sent-Debias 和 FairFil 的词表。

这是反事实数据增强 (CDA) 的"输入字典":它决定了我们"改哪个变量"。
每一对 (male_word, female_word) 语义对立、其余语境保持不变。

注意:这就是 CD3 论文质疑的那个"依赖人工先验"的东西 —— 也正是你 HanFair
在中文上花大力气构建的对应物。复现时亲手碰一下它,你会对这个痛点有体感。
"""

# 论文 5.1.1: (Male, Female) 的 10 组词对
GENDER_PAIRS = [
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("son", "daughter"),
    ("guy", "gal"),
    ("male", "female"),
    ("his", "her"),
    ("himself", "herself"),
    ("john", "mary"),
]

# 建两个方向的查表，替换时 o(1)命中
MALE_TO_FEMALE = {m: f for m, f in GENDER_PAIRS}
FEMALE_TO_MALE = {f: m for m, f in GENDER_PAIRS}

# 所有敏感词的扁平集合，用于 “这句话里到底有没有敏感词"的 快速判断
ALL_GENDER_WORDS = set(MALE_TO_FEMALE) | set(FEMALE_TO_MALE)


def swap_gender_word(word: str):
    """
    把单个词换成它的反事实对立词；不是敏感词则返回None
    大寂写策略，先按小写切尔西，命中后尽量保留原词的首字母大小写
    """
    lower = word.lower()
    if lower in MALE_TO_FEMALE:
        swapped = MALE_TO_FEMALE[lower]
    elif lower in FEMALE_TO_MALE:
        swapped = FEMALE_TO_MALE[lower]
    else:
        return None
    # 保留首字母大小写 (句首的 He -> She)
    if word[:1].isupper():
        swapped = swapped.capitalize()
    return swapped


def main():
    print(f"gender 词对数量: {len(GENDER_PAIRS)}")
    print(f"敏感词总数(双向): {len(ALL_GENDER_WORDS)}")
    for w in ["boy", "She", "his", "dog"]:
        print(f"  {w!r:10} -> {swap_gender_word(w)!r}")


if __name__ == "__main__":
    main()
