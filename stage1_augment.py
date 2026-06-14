#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/13/26
Description: stage1_augment
Algorithm 1: 反事实对句对+biased prompt 搜索

Stage 1 - Step 1: 反事实句对生成
=================================
对应论文 Algorithm 1 的第 1-6 行 (Data Augmentation With Sensitive Attribute)。

输入: 一句含敏感词的原句 x
输出: 反事实句对 (x, x_hat) —— 只把敏感词换成对立群体词,其余一字不动。

这一步是 CDA 的本体。注意它有两个论文点名的缺陷(4.1.2 节):
  1. 依赖人工敏感词表 (prior knowledge)
  2. 替换后句对太相似 -> 模型容易过拟合
Step2 的 biased prompt 搜索就是来补这两个缺陷的(下一个模块再做)。

本模块零模型依赖,纯 Python,Mac 上秒跑。

"""

import re
from attribute_words import swap_gender_word, ALL_GENDER_WORDS


def tokenize(sentence: str):
    """极简分词:按词边界切,保留标点为独立 token,便于无损还原。
       复现阶段不追求完美分词,够用即可。"""
    return re.findall(r"\w+|[^\w\s]]", sentence)


def detokenize(tokens):
    out = ""
    for i, tok in enumerate(tokens):
        if i > 0 and re.match(r"\w", tok):
            out += " "
        out += tok
    return out


def make_counterfactual_pair(sentence: str):
    """
    生成反事实句对
    返回 x,x_hat, swapped_info 若句中无敏感词 swapped_info 里为None
    """
    tokens = tokenize(sentence)
    swapped_any = False
    swapped_info = []
    new_tokens = []
    for tok in tokens:
        sw = swap_gender_word(tok)
        if sw is not None:
            new_tokens.append(sw)
            swapped_info.append((tok, sw))
            swapped_any = True
        else:
            new_tokens.append(tok)

    if not swapped_any:
        return sentence, None, []
    return sentence, detokenize(new_tokens), swapped_info


def has_sensitive_word(sentence: str) -> bool:
    """论文 Algorithm 1 第 3 行:句子是否命中敏感词表(用于筛语料)。"""
    toks = {t.lower() for t in tokenize(sentence)}
    return bool(toks & ALL_GENDER_WORDS)


def main():
    # 用论文 Fig.1 那个原例 + 几个自造句测试
    samples = [
        "The boy wants a dirt bike as a gift.",  # 论文原例
        "He is a very intelligent doctor.",  # 含 he
        "Her father drove the car to work.",  # 含两个敏感词
        "The cat sat on the mat.",  # 无敏感词
    ]
    print("=" * 60)
    for s in samples:
        x, x_hat, info = make_counterfactual_pair(s)
        print(f"原句 : {x}")
        if x_hat is None:
            print("       (无敏感词,跳过 —— 不会进入训练语料)")
        else:
            print(f"反事实: {x_hat}")
            print(f"       替换: {info}")
        print("-" * 60)


if __name__ == "__main__":
    main()
