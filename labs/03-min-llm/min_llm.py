# -*- coding: utf-8 -*-
"""手搓最小 LLM：使用 CPU 训练字符级 GPT。

本文件采用便于阅读和复现的单文件框架：
1. 内置语料
2. 字符级分词器
3. 最小 GPT 模型
4. 随机批数据生成
5. 训练、绘图、保存和采样主流程

可直接使用命令行运行，也可按照 experiment_record.ipynb 复现实验。
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 1. 内置语料：公共领域古诗选段 ============
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    "远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。",
    "巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。",
    "沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
    "湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。",
    "杨柳青青江水平，闻郎江上踏歌声。东边日出西边雨，道是无晴却有晴。",
    "自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧霄。",
    "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
    "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
    "月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
    "晓看红湿处，花重锦官城。野径云俱黑，江船火独明。",
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
    "烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。",
    "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
    "花近高楼伤客心，万方多难此登临。锦江春色来天地，玉垒浮云变古今。",
    "独怜幽草涧边生，上有黄鹂深树鸣。春潮带雨晚来急，野渡无人舟自横。",
    "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
    "昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。",
    "晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。",
    "客舍青青柳色新，渭城朝雨浥轻尘。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "山光忽西落，池月渐东上。散发乘夕凉，开轩卧闲敞。",
    "荷笠带斜阳，青山独归远。苍苍竹林寺，杳杳钟声晚。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
    "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。",
    "花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "长安一片月，万户捣衣声。秋风吹不尽，总是玉关情。",
    "弃我去者，昨日之日不可留。乱我心者，今日之日多烦忧。",
    "抽刀断水水更流，举杯消愁愁更愁。人生在世不称意，明朝散发弄扁舟。",
    "千里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
    "山重水复疑无路，柳暗花明又一村。莫笑农家腊酒浑，丰年留客足鸡豚。",
    "纸上得来终觉浅，绝知此事要躬行。古人学问无遗力，少壮功夫老始成。",
    "死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。",
    "小荷才露尖尖角，早有蜻蜓立上头。泉眼无声惜细流，树阴照水爱晴柔。",
    "接天莲叶无穷碧，映日荷花别样红。毕竟西湖六月中，风光不与四时同。",
    "胜日寻芳泗水滨，无边光景一时新。等闲识得东风面，万紫千红总是春。",
    "半亩方塘一鉴开，天光云影共徘徊。问渠那得清如许，为有源头活水来。",
    "郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。",
    "人生自古谁无死，留取丹心照汗青。辛苦遭逢起一经，干戈寥落四周星。",
    "咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。",
    "千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。",
    "浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。",
    "九州生气恃风雷，万马齐喑究可哀。我劝天公重抖擞，不拘一格降人才。",
    "力微任重久神疲，再竭衰庸定不支。苟利国家生死以，岂因祸福避趋之。",
]


def build_corpus(extra_file=None):
    """拼接内置语料；若指定的补充语料存在，则追加其内容。"""
    text = "\n".join(POEMS)
    if extra_file:
        extra_path = Path(extra_file)
        if not extra_path.is_absolute():
            extra_path = Path(__file__).resolve().parent / extra_path
        if extra_path.exists():
            text += "\n" + extra_path.read_text(encoding="utf-8")
            print(f"已追加语料：{extra_path}")
    return text


# ============ 2. 字符级分词器 ============
class CharTokenizer:
    def __init__(self, text=None, chars=None):
        if chars is None:
            if text is None:
                raise ValueError("text 和 chars 至少提供一个")
            chars = sorted(set(text))
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """将字符串编码为 token ID；提示词中的未知字符会被忽略。"""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ============ 3. 最小 GPT 模型 ============
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力。"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)

        # (B, T, C) -> (B, H, T, D)，其中 D=C/H。
        q = q.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        y = att @ v

        # (B, H, T, D) -> (B, T, C)。
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.proj(y)


class Block(nn.Module):
    """Pre-Norm Transformer 块。"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=128,
        use_pos=True,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = (
            nn.Embedding(block_size, n_embd) if use_pos else None
        )
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # 输入词嵌入和输出分类层共享同一份权重。
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """小方差初始化，使初始预测接近均匀分布。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("输入序列长度超过 block_size")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @staticmethod
    def _banned_ngram_tokens(sequence, ngram_size):
        """返回会重复已有 n-gram 的候选 token，用于选做实验。"""
        if ngram_size <= 0 or len(sequence) < ngram_size - 1:
            return set()
        prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned = set()
        for start in range(len(sequence) - ngram_size + 1):
            old_prefix = tuple(sequence[start : start + ngram_size - 1])
            if old_prefix == prefix:
                banned.add(sequence[start + ngram_size - 1])
        return banned

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=1.0,
        top_k=None,
        no_repeat_ngram_size=0,
    ):
        """按 temperature 和 top-k 自回归采样。"""
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if no_repeat_ngram_size > 0:
                for row in range(idx.size(0)):
                    sequence = idx[row].tolist()
                    banned = self._banned_ngram_tokens(
                        sequence, no_repeat_ngram_size
                    )
                    if banned:
                        logits[row, list(banned)] = float("-inf")

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k)[0][:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # 极端限制导致所有候选都被屏蔽时，退回未限制的模型分布。
            if not torch.isfinite(logits).any(dim=-1).all():
                fallback, _ = self(idx_cond)
                logits = fallback[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============ 4. 数据批生成 ============
def get_batch(data, block_size, batch_size, device):
    """随机截取 block_size+1 个 token，并把标签错开一位。"""
    if len(data) <= block_size:
        raise ValueError("语料长度必须大于 block_size")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ============ 5. 训练、保存和采样 ============
def build_parser():
    parser = argparse.ArgumentParser(description="CPU 字符级 MiniGPT 实验")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prompts", type=str, default="春,月")
    parser.add_argument("--samples_per_prompt", type=int, default=2)
    parser.add_argument("--no_pos", action="store_true")
    parser.add_argument("--extra_corpus", type=str, default="corpus_extra.txt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    return parser


def make_run_tag(args):
    if args.run_name:
        return args.run_name
    if args.no_pos:
        return "no_pos"
    block_part = "" if args.block_size == 128 else f"_T{args.block_size}"
    return f"L{args.n_layer}_E{args.n_embd}{block_part}_lr{args.lr:g}"


def model_config_from_args(args, vocab_size):
    return {
        "vocab_size": vocab_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "block_size": args.block_size,
        "use_pos": not args.no_pos,
    }


def save_checkpoint(path, model, tokenizer, model_config, run_tag):
    payload = {
        "model_state_dict": model.state_dict(),
        "chars": tokenizer.chars,
        "model_config": model_config,
        "run_tag": run_tag,
    }
    torch.save(payload, path)


def load_checkpoint(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    tokenizer = CharTokenizer(chars=payload["chars"])
    config = payload["model_config"]
    model = MiniGPT(**config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, tokenizer, config, payload.get("run_tag", "checkpoint")


def save_losses_csv(path, losses):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "loss"])
        writer.writerows(enumerate(losses, start=1))


def save_loss_curve(path, losses, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "Training Loss "
        f"(L={args.n_layer}, E={args.n_embd}, T={args.block_size}, "
        f"lr={args.lr:g}, pos={'off' if args.no_pos else 'on'})"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_samples(model, tokenizer, args, model_tag, output_dir, device):
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]
    records = []
    sections = []

    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer.encode(prompt)
        if not encoded:
            raise ValueError(f"提示词不含词表内字符：{prompt!r}")
        for sample_index in range(args.samples_per_prompt):
            sample_seed = args.seed + 1000 + prompt_index * 100 + sample_index
            torch.manual_seed(sample_seed)
            start = torch.tensor([encoded], dtype=torch.long, device=device)
            output = model.generate(
                start,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            text = tokenizer.decode(output[0].tolist())
            records.append(
                {
                    "prompt": prompt,
                    "sample_index": sample_index + 1,
                    "sample_seed": sample_seed,
                    "text": text,
                }
            )
            sections.append(
                f"提示词：{prompt}｜样本：{sample_index + 1}｜seed：{sample_seed}\n{text}"
            )

    top_k_text = "all" if args.top_k <= 0 else str(args.top_k)
    sample_tag = (
        f"{model_tag}_temp{args.temperature:g}_topk{top_k_text}"
        f"_ngram{args.no_repeat_ngram_size}"
    )
    text_path = output_dir / f"generated_{sample_tag}.txt"
    text_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    # 为基线生成结果保留一个稳定的文件名。
    if (
        model_tag == "L2_E128_lr0.001"
        and args.temperature == 1.0
        and args.top_k == 20
        and args.no_repeat_ngram_size == 0
    ):
        (output_dir / "generated_base.txt").write_text(
            "\n\n".join(sections) + "\n", encoding="utf-8"
        )

    print(f"已保存生成文本：{text_path}")
    return records, text_path


def main():
    args = build_parser().parse_args()
    if args.iters <= 0 and not args.sample_only:
        raise ValueError("iters 必须大于 0")
    if args.samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt 必须大于 0")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cpu"

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PyTorch {torch.__version__} | 设备：{device} | "
        f"线程数：{torch.get_num_threads()}"
    )

    if args.sample_only:
        if not args.checkpoint:
            raise ValueError("--sample_only 必须同时提供 --checkpoint")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = script_dir / checkpoint_path
        model, tokenizer, model_config, model_tag = load_checkpoint(
            checkpoint_path, device
        )
        n_params = sum(parameter.numel() for parameter in model.parameters())
        print(f"已加载检查点：{checkpoint_path}")
        print(f"模型参数量：{n_params:,} | 配置：{model_config}")
        records, text_path = generate_samples(
            model, tokenizer, args, model_tag, output_dir, device
        )
        summary = {
            "mode": "sample_only",
            "checkpoint": str(checkpoint_path),
            "model_tag": model_tag,
            "model_config": model_config,
            "n_params": n_params,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "generated_file": str(text_path),
            "samples": records,
        }
        summary_tag = (
            f"sample_{model_tag}_temp{args.temperature:g}_"
            f"topk{args.top_k}_ngram{args.no_repeat_ngram_size}"
        )
        summary_path = output_dir / f"summary_{summary_tag}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已保存摘要：{summary_path}")
        return

    text = build_corpus(args.extra_corpus)
    tokenizer = CharTokenizer(text=text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"语料：{len(text)} 字符 | 词表大小 V = {tokenizer.vocab_size}")

    model_config = model_config_from_args(args, tokenizer.vocab_size)
    model = MiniGPT(**model_config).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    run_tag = make_run_tag(args)
    print(
        f"模型参数量：{n_params:,} ({n_params / 1e6:.2f}M) | "
        f"配置：L={args.n_layer} H={args.n_head} E={args.n_embd} "
        f"T={args.block_size} pos={'有' if not args.no_pos else '无'}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    start_time = time.time()

    model.train()
    for step in range(1, args.iters + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % 100 == 0 or step == args.iters:
            elapsed = time.time() - start_time
            speed = step / max(elapsed, 1e-8)
            eta = (args.iters - step) / max(speed, 1e-8)
            print(
                f"step {step:5d}/{args.iters} | loss {loss.item():.4f} | "
                f"{speed:.2f} it/s | 预计剩余 {eta:.0f}s"
            )

    train_time = time.time() - start_time
    first_window = losses[: min(100, len(losses))]
    print(
        f"训练完成，耗时 {train_time / 60:.2f} 分钟 | "
        f"初始 loss {losses[0]:.4f} | 最终 loss {losses[-1]:.4f} | "
        f"前 {len(first_window)} 步均值 {sum(first_window) / len(first_window):.4f}"
    )

    losses_path = output_dir / f"losses_{run_tag}.csv"
    curve_path = output_dir / f"loss_curve_{run_tag}.png"
    save_losses_csv(losses_path, losses)
    save_loss_curve(curve_path, losses, args)
    print(f"已保存 loss 数据：{losses_path}")
    print(f"已保存 loss 曲线：{curve_path}")

    checkpoint_path = None
    if args.save_checkpoint:
        checkpoint_path = output_dir / f"checkpoint_{run_tag}.pt"
        save_checkpoint(
            checkpoint_path, model, tokenizer, model_config, run_tag
        )
        print(f"已保存检查点：{checkpoint_path}")

    samples = []
    generated_path = None
    if not args.skip_generation:
        samples, generated_path = generate_samples(
            model, tokenizer, args, run_tag, output_dir, device
        )

    summary = {
        "mode": "train",
        "run_tag": run_tag,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "corpus_chars": len(text),
        "vocab_size": tokenizer.vocab_size,
        "model_config": model_config,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "lr": args.lr,
        "n_params": n_params,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_seconds": train_time,
        "losses_file": str(losses_path),
        "curve_file": str(curve_path),
        "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
        "generated_file": str(generated_path) if generated_path else None,
        "samples": samples,
    }
    summary_path = output_dir / f"summary_{run_tag}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存实验摘要：{summary_path}")


if __name__ == "__main__":
    main()
