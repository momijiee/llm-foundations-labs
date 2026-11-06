"""CPU 字符级语言模型蒸馏；主要入口为同目录 Notebook。"""
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import time
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    seed: int = 0
    block_size: int = 96
    batch_size: int = 24
    teacher_iters: int = 200
    student_iters: int = 120
    abl_iters: int = 80
    lr: float = 3e-3
    temperature: float = 4.0
    alpha: float = 0.5
    eval_iters: int = 20
    threads: int = 4

# 与指南的槽位和文法一致。古诗为公版文本。
SUBJ = '小明 小红 阿远 书生 渔夫 牧童 旅人 诗人 少年 故人 归客 山僧 孤客 游子'.split()
PLACE = '江边 山寺 长亭 渡口 竹林 荒村 城楼 野渡 溪桥 松间 篱下 窗前 陌上 舟中'.split()
VERB = '望见 听见 想起 送去 寻到 梦见 独对 遥看 静听 又逢 恰遇 空忆'.split()
OBJ = '明月 落花 孤舟 归雁 残灯 细雨 青山 晚钟 白雪 寒江 疏星 暮云 秋草 远帆'.split()
MOOD = '心中惆怅 不觉天明 欣然归去 泪湿衣襟 长叹一声 默然无语 忽忆旧游 欲语还休 此意难平 一笑而别'.split()
POEMS = [
'床前明月光，疑是地上霜。举头望明月，低头思故乡。',
'春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。',
'白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
'千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。',
'空山不见人，但闻人语响。返景入深林，复照青苔上。',
'红豆生南国，春来发几枝。愿君多采撷，此物最相思。',
'独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。',
'月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。',
'两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。',
'朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。',
'日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。',
'渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。',
'寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。',
'独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
'清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。',
'朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。']

def build_corpus(n_lines=3000, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append(f'{s}{p}{v}{o}，{m}。')
        elif i % 3 == 1:
            out.append(f'{s}在{p}{o}了一夜{rng.choice(["看", "听", "守", "忆"])}，{m}。')
        else:
            out.append(f'{s}{v}道：“{o}{rng.choice(["正好", "可怜", "可惜", "依旧"])}。”{m}。')
        out.append('\n')
    return ''.join(out + POEMS)

class GPT(nn.Module):
    def __init__(self, vocab, width, block=96):
        super().__init__()
        self.width, self.block = width, block
        self.tok_emb = nn.Embedding(vocab, width)
        self.pos_emb = nn.Embedding(block, width)
        self.blocks = nn.ModuleList([Block(width, block) for _ in range(2)])
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, hidden=False):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1)))
        for layer in self.blocks:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return (logits, x) if hidden else logits

class Block(nn.Module):
    def __init__(self, width, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.register_buffer('mask', torch.tril(torch.ones(block, block)).bool().view(1, 1, block, block))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [z.view(b, t, 4, c // 4).transpose(1, 2) for z in self.qkv(self.ln1(x)).chunk(3, dim=-1)]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(c // 4)
        a = a.masked_fill(~self.mask[:, :, :t, :t], float('-inf')).softmax(-1)
        x = x + self.proj((a @ v).transpose(1, 2).contiguous().view(b, t, c))
        return x + self.mlp(self.ln2(x))

def kd_loss(student, teacher, temperature):
    # KL(p_teacher || p_student)，按 token 平均，避免序列长度放大 KL。
    v = student.size(-1)
    return F.kl_div(F.log_softmax(student.reshape(-1, v) / temperature, -1),
                    F.softmax(teacher.reshape(-1, v) / temperature, -1),
                    reduction='batchmean') * temperature ** 2

def batch(data, cfg, generator):
    offsets = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,), generator=generator)
    x = torch.stack([data[i:i + cfg.block_size] for i in offsets])
    y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in offsets])
    return x, y

@torch.no_grad()
def evaluate(model, data, cfg, teacher=None):
    model.eval()
    if teacher is not None:
        teacher.eval()
    gen = torch.Generator().manual_seed(1234)
    losses, hit, count = [], 0, 0
    for _ in range(cfg.eval_iters):
        x, y = batch(data, cfg, gen)
        z = model(x)
        losses.append(F.cross_entropy(z.reshape(-1, z.size(-1)), y.reshape(-1)).item())
        if teacher is not None:
            hit += (z.argmax(-1) == teacher(x).argmax(-1)).sum().item()
            count += y.numel()
    loss = sum(losses) / len(losses)
    return {'val_loss': loss, 'ppl': math.exp(loss), 'agree': hit / count if count else None}

def train(model, data, cfg, steps, teacher=None, temperature=4., alpha=.5, feat=0., cache=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    projection = nn.Linear(model.width, teacher.width) if feat else None
    params = list(model.parameters()) + (list(projection.parameters()) if projection is not None else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=.01)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    history = {'objective': [], 'ce': [], 'kd': [], 'feature': []}
    start = time.perf_counter()
    for i in range(steps):
        if cache is None:
            x, y = batch(data, cfg, gen)
            tl = None
        else:
            x, y, tl = cache[i % len(cache)]
        z, hs = model(x, hidden=True)
        ce = F.cross_entropy(z.reshape(-1, z.size(-1)), y.reshape(-1))
        kd, fm = z.new_zeros(()), z.new_zeros(())
        if teacher is not None or tl is not None:
            if tl is None:
                with torch.no_grad():
                    tl, ht = teacher(x, hidden=True)
            kd = kd_loss(z, tl, temperature)
            loss = (1 - alpha) * ce + alpha * kd
            if projection is not None:
                fm = F.mse_loss(projection(hs), ht)
                loss = loss + feat * fm
        else:
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        for key, value in zip(history, (loss, ce, kd, fm)):
            history[key].append(value.item())
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == steps:
            print(f'  {i+1}/{steps}: objective={loss.item():.4f}, CE={ce.item():.4f}', flush=True)
    return history, time.perf_counter() - start

class Experiment:
    def __init__(self, cfg=None, output='runs_distill'):
        self.cfg = cfg or Config()
        torch.set_num_threads(self.cfg.threads)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        text, val = build_corpus(), build_corpus(300, 999)
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        assert set(val) <= set(text), '留出集存在未知字符'
        self.train_ids = torch.tensor([self.stoi[c] for c in text])
        self.val_ids = torch.tensor([self.stoi[c] for c in val])
        self.signature = {'config': asdict(self.cfg), 'chars': self.chars,
                          'data_sha256': hashlib.sha256((text + val).encode()).hexdigest(), 'implementation': 1}
        self.environment = {'os': platform.system() + ' ' + platform.release(),
                            'python': platform.python_version(), 'torch': torch.__version__,
                            'cpu': platform.processor() or '请在系统信息中补充 CPU 型号',
                            'logical_cores': os.cpu_count(), 'threads': self.cfg.threads,
                            'device': 'CPU', 'train_chars': len(text), 'val_chars': len(val),
                            'vocab': len(self.chars)}
        self.write_json('environment.json', self.environment)
        self.write_json('configuration.json', self.signature)

    def write_json(self, name, data):
        target = self.output / name
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(target)

    def new_model(self, width, seed):
        torch.manual_seed(seed)
        return GPT(len(self.chars), width, self.cfg.block_size)

    def fit(self, name, width, steps, teacher=None, feat=0., force=False):
        path = self.output / f'{name}.pt'
        record = self.output / f'{name}.json'
        spec = {'signature': self.signature, 'width': width, 'steps': steps, 'feat': feat,
                'teacher_sha256': hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in teacher.parameters())).hexdigest() if teacher is not None else None}
        model = self.new_model(width, self.cfg.seed if name.startswith('teacher') else self.cfg.seed + 7)
        if path.exists() and record.exists() and not force:
            row = json.loads(record.read_text(encoding='utf-8'))
            if row['spec'] != spec:
                raise ValueError(f'{name} 缓存配置不同：请改用新的输出目录或 force=True。')
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            print(f'{name}: 复用已完成记录，原训练耗时 {row["time_s"]:.2f}s')
            return model, row
        print(f'开始 {name}', flush=True)
        hist, seconds = train(model, self.train_ids, self.cfg, steps, teacher,
                              self.cfg.temperature, self.cfg.alpha, feat)
        row = {'name': name, 'spec': spec, 'params': sum(p.numel() for p in model.parameters()),
               'steps': steps, 'time_s': seconds, 'final_train_loss': hist['objective'][-1],
               'final_train_ce': hist['ce'][-1], 'history': hist,
               **evaluate(model, self.val_ids, self.cfg, teacher)}
        torch.save(model.state_dict(), path)
        self.write_json(f'{name}.json', row)
        return model, row

    def baseline(self):
        t, tr = self.fit('teacher', 160, self.cfg.teacher_iters)
        s, sr = self.fit('scratch', 64, self.cfg.student_iters)
        k, kr = self.fit('kd', 64, self.cfg.student_iters, t)
        # 对照组也必须测与同一教师的一致率。
        sr.update(evaluate(s, self.val_ids, self.cfg, t))
        self.write_json('scratch.json', sr)
        rows = [tr, sr, kr]
        self.write_json('results_all.json', rows)
        return (t, s, k), rows

    @torch.no_grad()
    def samples(self, models):
        rows = []
        for name, model in zip(('teacher', 'scratch', 'kd'), models):
            model.eval()
            torch.manual_seed(self.cfg.seed + 2026)
            ids = torch.tensor([[self.stoi['月']]])
            for _ in range(60):
                z = model(ids[:, -self.cfg.block_size:])[:, -1] / .9
                cutoff = torch.topk(z, min(20, z.size(-1))).values[:, -1:]
                z = z.masked_fill(z < cutoff, float('-inf'))
                ids = torch.cat((ids, torch.multinomial(z.softmax(-1), 1)), dim=1)
            continuation = ''.join(self.chars[i] for i in ids[0, 1:].tolist())
            rows.append({'model': name, 'prompt': '月', 'new_characters': 60,
                         'temperature': .9, 'top_k': 20, 'seed': self.cfg.seed + 2026,
                         'continuation': continuation})
        self.write_json('samples.json', rows)
        return rows

    def ablate(self, teacher):
        teacher.eval()
        gen = torch.Generator().manual_seed(2024)
        cache = []
        with torch.no_grad():
            for _ in range(32):
                x, y = batch(self.train_ids, self.cfg, gen)
                cache.append((x, y, teacher(x)))
        rows = []
        for temp in (1, 2, 4, 8):
            for alpha in (.3, .5, .7, .9):
                name = f'ablate_T{temp}_a{alpha}'
                path = self.output / f'{name}.json'
                if path.exists():
                    row = json.loads(path.read_text(encoding='utf-8'))
                    if row['signature'] != self.signature:
                        raise ValueError('消融缓存配置不同，请使用新输出目录。')
                else:
                    print(name, flush=True)
                    model = self.new_model(64, self.cfg.seed + 7)
                    hist, seconds = train(model, self.train_ids, self.cfg, self.cfg.abl_iters,
                                          temperature=temp, alpha=alpha, cache=cache)
                    row = {'T': temp, 'alpha': alpha, 'steps': self.cfg.abl_iters,
                           'time_s': seconds, 'signature': self.signature,
                           **evaluate(model, self.val_ids, self.cfg), 'history': hist}
                    self.write_json(f'{name}.json', row)
                rows.append(row)
        self.write_json('results_ablate.json', rows)
        return rows

    def scan(self, base_models, base_rows):
        rows = []
        for width in (96, 160, 256):
            if width == 160:
                tr, kr = base_rows[0], base_rows[2]
            else:
                teacher, tr = self.fit(f'teacher_C{width}', width, self.cfg.teacher_iters)
                _, kr = self.fit(f'kd_teacher_C{width}', 64, self.cfg.student_iters, teacher)
            scratch = base_rows[1]['val_loss']
            rows.append({'teacher_width': width, 'teacher_params': tr['params'],
                         'teacher_loss': tr['val_loss'], 'teacher_time_s': tr['time_s'],
                         'scratch_loss': scratch, 'kd_loss': kr['val_loss'],
                         'kd_time_s': kr['time_s'], 'gain_percent': 100 * (scratch - kr['val_loss']) / scratch})
            self.write_json('teacher_scan.json', rows)
        return rows

    def quantize(self, model):
        import warnings
        import torch.ao.quantization as tq
        engines = torch.backends.quantized.supported_engines
        engine = next((e for e in ('x86', 'fbgemm', 'qnnpack') if e in engines), None)
        if engine is None:
            raise RuntimeError(f'当前 PyTorch 不支持 CPU 动态量化：{engines}')
        torch.backends.quantized.engine = engine
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            quantized = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        rows = []
        x, _ = batch(self.val_ids, self.cfg, torch.Generator().manual_seed(77))
        for label, m in (('fp32', model), ('int8', quantized)):
            path = self.output / f'student_{label}.pt'
            torch.save(m.state_dict(), path)
            with torch.inference_mode():
                for _ in range(5):
                    m(x)
                timings = []
                for _ in range(30):
                    start = time.perf_counter()
                    m(x)
                    timings.append((time.perf_counter() - start) * 1000)
            rows.append({'format': label, 'bytes': path.stat().st_size,
                         'median_forward_ms': sorted(timings)[len(timings)//2],
                         'batch_size': self.cfg.batch_size, 'sequence_length': self.cfg.block_size,
                         'warmup': 5, 'repeats': 30, 'engine': engine,
                         **evaluate(m, self.val_ids, self.cfg)})
        self.write_json('quantization.json', rows)
        return rows

    def parameter_table(self):
        rows = []
        for width in (160, 64):
            model = self.new_model(width, self.cfg.seed)
            exact = sum(p.numel() for p in model.parameters())
            approx = 2 * len(self.chars) * width + self.cfg.block_size * width + 2 * 12 * width ** 2
            rows.append({'model': 'teacher' if width == 160 else 'student', 'n_layer': 2,
                         'n_head': 4, 'n_embd': width, 'block_size': self.cfg.block_size,
                         'vocab': len(self.chars), 'params': exact, 'approx_params': approx,
                         'relative_error_percent': abs(exact - approx) / exact * 100})
        rows[1]['compression_ratio'] = rows[0]['params'] / rows[1]['params']
        self.write_json('parameters.json', rows)
        return rows

    def export_tables(self, tables):
        for name, rows in tables.items():
            clean = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
            keys = list(dict.fromkeys(k for row in clean for k in row))
            with (self.output / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(clean)
        self.write_json('report_data.json', tables)

def plot_results(rows, ablation, output):
    import matplotlib.pyplot as plt
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in rows:
        axes[0].plot(range(1, row['steps'] + 1), row['history']['objective'], label=row['name'])
        axes[1].plot(range(1, row['steps'] + 1), row['history']['ce'], label=row['name'])
    for ax, title, ylabel in zip(axes, ('Training objectives (different definitions)', 'Hard-label CE (same definition)'), ('Objective', 'Cross entropy')):
        ax.set(title=title, xlabel='Training step', ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=160)
    if ablation:
        fig2, ax = plt.subplots(figsize=(6, 4))
        for t in (1, 2, 4, 8):
            subset = [r for r in ablation if r['T'] == t]
            ax.plot([r['alpha'] for r in subset], [r['val_loss'] for r in subset], 'o-', label=f'T={t}')
        ax.set(xlabel='alpha', ylabel='Held-out CE', title='Temperature / weight ablation')
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(output / 'ablation_curves.png', dpi=160)
        fig3, ax = plt.subplots(figsize=(6, 4))
        grid = [[next(r['val_loss'] for r in ablation if r['T'] == t and r['alpha'] == a)
                 for a in (.3, .5, .7, .9)] for t in (1, 2, 4, 8)]
        im = ax.imshow(grid, cmap='viridis_r')
        ax.set(xticks=range(4), xticklabels=(.3, .5, .7, .9), yticks=range(4),
               yticklabels=(1, 2, 4, 8), xlabel='alpha', ylabel='T', title='Held-out CE (lower is better)')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{grid[i][j]:.4f}', ha='center', va='center', color='white' if im.norm(grid[i][j]) < .5 else 'black')
        fig3.colorbar(im, ax=ax)
        fig3.tight_layout()
        fig3.savefig(output / 'ablation_heatmap.png', dpi=160)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'teacher', 'all', 'ablate'], default='quick')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    cfg = Config()
    if args.mode == 'quick':
        cfg = Config(teacher_iters=2, student_iters=2, abl_iters=2, eval_iters=1, batch_size=2)
    e = Experiment(cfg, Path(__file__).parent / 'runs_distill' / ('quick' if args.mode == 'quick' else 'formal'))
    if args.mode == 'teacher':
        e.fit('teacher', 160, cfg.teacher_iters, force=args.retrain)
    elif args.mode == 'ablate':
        teacher, _ = e.fit('teacher', 160, cfg.teacher_iters)
        e.ablate(teacher)
    else:
        models, rows = e.baseline()
        print(json.dumps([{k: v for k, v in r.items() if k not in ('history', 'spec')} for r in rows], indent=2))
        e.samples(models)

if __name__ == '__main__':
    main()
