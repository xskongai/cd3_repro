"""
Stage 1 - Step 2: biased prompt 自动搜索 (CD3 核心创新)
=====================================================
对应论文 Algorithm 1 的第 8-15 行。

思路:
  给反事实句对 (x, x_hat) 各拼上一个候选 prompt P,得到 (x', x_hat')。
  用 BERT 编码两句、算 cos 相似度。相似度越低 = prompt 越能放大群体差异 = 越"偏"。
  beam search 每轮留 top-K 最低相似度的 prompt,下轮在其后继续拼词,迭代 ε 轮。

这一步第一次加载 BERT(只前向、不训练)。设备走 config.DEVICE (Mac=MPS)。

注: transformers 5.x 下只用最稳的 AutoTokenizer/AutoModel。
若报 API 错,把报错贴回来现场改。
"""

import torch
from transformers import AutoTokenizer, AutoModel

import config
from stage1_augment import make_counterfactual_pair


_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"[加载] {config.MODEL_NAME} 到 {config.DEVICE} ...")
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        _model = AutoModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
        _model.eval()
        print("[加载] 完成")
    return _tokenizer, _model


@torch.no_grad()
def encode(sentences):
    """句向量池化,由 config.POOLING 控制:
      mean: 所有 token 平均(论文用法,会稀释单敏感词信号)
      cls : [CLS] 向量(第 0 个 token)
    """
    tok, model = _load_model()
    enc = tok(sentences, padding=True, truncation=True,
              max_length=64, return_tensors="pt").to(config.DEVICE)
    out = model(**enc).last_hidden_state          # [B, T, H]

    if config.POOLING == "cls":
        return out[:, 0, :]                       # [CLS] 向量

    # 默认 mean-pooling
    mask = enc["attention_mask"].unsqueeze(-1)
    summed = (out * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# 英文高频词表(模拟论文的 Wikipedia top-5000,这里给一份精简高频表)。
# 写死可复现、不依赖 tokenizer 内部结构(避开 transformers 5.x 的 get_vocab 行为差异)。
# 故意混入一些可能带性别/职业色彩的词,便于观察 biased prompt 是否能被搜出来。
_BASE_VOCAB = """
the of and to in is was for that with as his he it her she they on at by an be
this from but not are or had which have one all were their has would there been
when who will more no out up into time than its over only some could other these
man woman men women boy girl father mother son daughter brother sister husband wife
king queen actor actress doctor nurse engineer teacher lawyer soldier scientist
strong weak smart beautiful handsome emotional logical aggressive gentle leader
football cooking nursing fighting driving fixing caring teaching building cleaning
heavyweight pitcher coach captain chairman secretary assistant manager worker
company office home kitchen school hospital court army team family business money
played worked drove fixed solved lifted taught cooked cleaned helped managed led
initially exactly typical strange experienced approximately whether either allows
""".split()


def build_vocab(tokenizer, size):
    """候选词表:用写死的高频词表(论文用 Wikipedia top-5000 的等价替身)。
    去重保序后截断到 size。tokenizer 参数保留接口一致性,此版本未用到。"""
    seen = set()
    words = []
    for w in _BASE_VOCAB:
        if w not in seen:
            seen.add(w)
            words.append(w)
        if len(words) >= size:
            break
    print(f"vocab:{words}")
    return words


def search_biased_prompts(pairs):
    """beam search 搜 biased prompts。返回每轮的 top-K。"""
    tok, _ = _load_model()
    vocab = build_vocab(tok, config.VOCAB_SIZE)
    print(f"[搜索] 候选词表 {len(vocab)} 词, beam K={config.BEAM_K}, "
          f"迭代 {config.NUM_ITERS} 轮, 句对 {len(pairs)} 组")

    beam = [""]
    all_results = []

    for it in range(config.NUM_ITERS):
        scored = []
        for prefix in beam:
            for w in vocab:
                prompt = (prefix + " " + w).strip()
                xs = [x + " " + prompt for x, _ in pairs]
                print(f"xs:{xs}")
                xhs = [xh + " " + prompt for _, xh in pairs]
                print(f"xhs:{xhs}")
                z = encode(xs)
                zh = encode(xhs)
                avg = cos_sim(z, zh).mean().item()
                scored.append((avg, prompt))
        scored.sort(key=lambda t: t[0])
        beam = [p for _, p in scored[:config.BEAM_K]]
        all_results.append(scored[:config.BEAM_K])

        print(f"\n--- 第 {it+1} 轮:最偏的 {config.MAX_PROMPT_PER_ITER_PRINT} 个 prompt "
              f"(cos 越低越偏) ---")
        for avg, p in scored[:config.MAX_PROMPT_PER_ITER_PRINT]:
            print(f"  cos={avg:.6f}  {p!r}")

    return all_results


if __name__ == "__main__":
    torch.manual_seed(config.SEED)

    raw_sentences = [
        "The boy wants a dirt bike as a gift.",
        "He is a very intelligent person.",
        "The man worked as a doctor for years.",
        "His father drove to the office.",
        "That guy is good at math.",
        "The son inherited the family business.",
        "He always solved problems quickly.",
        "The male candidate gave a speech.",
        "John is the leader of the team.",
        "He fixed the engine himself.",
        "The boy played football after school.",
        "His brother is a famous scientist.",
        "The man lifted the heavy box.",
        "He is the chairman of the board.",
        "The father taught his son to drive.",
    ]

    pairs = []
    print("test begin...")
    for s in raw_sentences:
        x, x_hat, _ = make_counterfactual_pair(s)
        if x_hat is not None:
            pairs.append((x, x_hat))
            print(f"pair x:{x},x_hat:{x_hat}")
    print(f"[准备] 从 {len(raw_sentences)} 句筛出 {len(pairs)} 个反事实句对\n")

    search_biased_prompts(pairs)