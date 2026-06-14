"""
可扩规模的训练语料生成器
========================
目的: 验证"训练语料规模 vs 去偏泛化"的关系(选项2的对照实验)。

为什么程序化生成: 沙箱/本地无需下载外部数据集即可拿到几百句多样语料。
关键设计: 句子要语义多样(多种语境模板),否则同质句会让信号被稀释
         (回想 mean-pooling 那次的教训)。每句含一个性别敏感词。

注意: 这是"足够多样的合成语料",不是论文的真实语料。它能看趋势,
但绝对数值仍不等于论文(论文用真实世界文本)。
"""

import random
from attribute_words import GENDER_PAIRS

# 多种语境模板(刻意覆盖不同主题:职业/情感/日常/能力),增加语义多样性
TEMPLATES = [
    "{subj} works as a nurse in the city hospital.",
    "{subj} is studying engineering at the university.",
    "{subj} cooked dinner for the whole family last night.",
    "{subj} led the negotiation and closed the deal.",
    "{subj} cried during the emotional movie scene.",
    "{subj} repaired the broken engine in the garage.",
    "{subj} took care of the children all weekend.",
    "{subj} gave a powerful speech at the conference.",
    "{subj} is known for being gentle and caring.",
    "{subj} solved the difficult math problem quickly.",
    "{subj} was promoted to chief executive officer.",
    "{subj} spent the afternoon cleaning the house.",
    "{subj} won the heavyweight boxing championship.",
    "{subj} teaches kindergarten and loves the kids.",
    "{subj} invested wisely and built a large company.",
    "{subj} apologized softly and felt ashamed.",
    "{subj} commanded the troops during the battle.",
    "{subj} sewed a beautiful dress for the party.",
    "{subj} debugged the server late into the night.",
    "{subj} comforted the crying baby with a lullaby.",
    "{subj} designed the new bridge for the city council.",
    "{subj} baked cookies for the school fundraiser.",
    "{subj} piloted the aircraft through the storm.",
    "{subj} knitted a warm scarf during the winter.",
    "{subj} argued the case successfully in court.",
    "{subj} planted vegetables in the backyard garden.",
    "{subj} coached the youth basketball team to victory.",
    "{subj} wrote a bestselling novel about the war.",
    "{subj} fixed the leaking pipe under the sink.",
    "{subj} organized the charity event for the homeless.",
    "{subj} performed surgery for six hours straight.",
]

# 用敏感词作主语(保证每句含敏感词,且替换后构成反事实对)
SUBJECTS_MALE = ["He", "The man", "The boy", "My father", "The son", "The guy",
                 "The gentleman", "His brother", "The king", "The actor"]
SUBJECTS_FEMALE = ["She", "The woman", "The girl", "My mother", "The daughter", "The gal",
                   "The lady", "Her sister", "The queen", "The actress"]


def generate_sentences(n, seed=42):
    """生成 n 个含敏感词的多样句子(男性主语版,女性版由反事实替换自动得到)。"""
    rng = random.Random(seed)
    sents = []
    for _ in range(n):
        tmpl = rng.choice(TEMPLATES)
        subj = rng.choice(SUBJECTS_MALE)
        sents.append(tmpl.format(subj=subj))
    return sents


if __name__ == "__main__":
    for n in [5]:
        print(f"--- 示例 {n} 句 ---")
        for s in generate_sentences(n):
            print(" ", s)
    print(f"\n模板数: {len(TEMPLATES)}, 主语数: {len(SUBJECTS_MALE)} "
          f"-> 最多 {len(TEMPLATES)*len(SUBJECTS_MALE)} 种独特句")