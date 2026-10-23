"""实验六：SmolVLM-256M 的网格定位、工具纠错与两轮 RSI。

运行数据保存在本目录的 results/；训练图、适配器和原始记忆不应直接公开。
CPU 使用 fp32 LoRA，只有明确选择 CUDA 量化时才能称为 QLoRA。
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "hw6_imgs"
RESULTS = ROOT / "results"
SFT_JSON = ROOT / "sft_data.json"
SFT_DIR = ROOT / "sft_adapter"
RSI_DIR = ROOT / "rsi_adapter"
MEM_JSON = ROOT / "rsi_memory.json"
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
GRID, SIZE, CELL = 4, 256, 64
TRAIN_N, EVAL_N = 40, 15
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
SHAPES = ("圆", "方")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def draw_scene(rng: random.Random):
    items = []
    for _ in range(rng.randint(1, 4)):
        color, shape = rng.choice(tuple(COLORS)), rng.choice(SHAPES)
        side = rng.randint(34, 54)
        x, y = rng.randint(8, SIZE - side - 8), rng.randint(8, SIZE - side - 8)
        items.append({"class": color + shape, "box": [x, y, x + side, y + side]})
    return items


def render(items, path: Path):
    image = Image.new("RGB", (SIZE, SIZE), "white")
    pen = ImageDraw.Draw(image)
    for item in items:
        cls, box = item["class"], item["box"]
        if cls[1:] == "圆":
            pen.ellipse(box, outline=COLORS[cls[0]], width=4)
        else:
            pen.rectangle(box, outline=COLORS[cls[0]], width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_data():
    gt = {}
    for split, count, seed, prefix in (("train", TRAIN_N, 2026, "t"), ("eval", EVAL_N, 42, "e")):
        rng = random.Random(seed)
        for index in range(count):
            name = f"{prefix}{index:02d}"
            gt[name] = draw_scene(rng)
            render(gt[name], IMG_DIR / split / f"{name}.png")
    save_json(IMG_DIR / "gt.json", gt)
    print(f"[data] {TRAIN_N} 张训练图、{EVAL_N} 张评测图；GT: hw6_imgs/gt.json")
    return gt


def load_gt():
    path = IMG_DIR / "gt.json"
    return read_json(path) if path.exists() else make_data()


def cell_of(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(GRID - 1, int(cy // CELL)), min(GRID - 1, int(cx // CELL))


def cell_box(cell):
    row, col = cell
    return [col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL]


def cell_text(cell):
    return f"第{cell[0] + 1}行第{cell[1] + 1}列"


def parse_cell(value):
    value = str(value)
    patterns = (
        r"第\s*([1-4])\s*行\s*第\s*([1-4])\s*列",
        r"r\s*([1-4])\s*c\s*([1-4])",
        r"(?<!\d)([1-4])\s*[,，、-]\s*([1-4])(?!\d)",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match:
            return int(match[1]) - 1, int(match[2]) - 1
    return None


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def color_components(image: Image.Image, color: str):
    """在纯色合成图中按颜色阈值抽取 8 邻域连通块。"""
    image = image.convert("RGB")
    px = image.load()
    ref = COLORS[color]
    remaining = {(x, y) for y in range(image.height) for x in range(image.width)
                 if sum((px[x, y][k] - ref[k]) ** 2 for k in range(3)) < 9000}
    comps = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        xmin = xmax = seed[0]
        ymin = ymax = seed[1]
        area = 1
        while queue:
            x, y = queue.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    point = x + dx, y + dy
                    if point in remaining:
                        remaining.remove(point)
                        queue.append(point)
                        xmin, xmax = min(xmin, point[0]), max(xmax, point[0])
                        ymin, ymax = min(ymin, point[1]), max(ymax, point[1])
                        area += 1
        comps.append({"box": [xmin, ymin, xmax + 1, ymax + 1], "pixels": area})
    return comps


def verify_cell(image, question, cell):
    """只根据目标颜色复核；最终入池仍需 GT 判分，防止同色异形误判。"""
    if cell is None:
        return False, "位置格式无法解析，请用第X行第Y列回答"
    comps = color_components(image, question["target"][0])
    if not comps:
        return False, "图中没有检测到该颜色"
    grid_box = cell_box(cell)
    best = 0.0
    for comp in comps:
        box = comp["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if grid_box[0] <= cx < grid_box[2] and grid_box[1] <= cy < grid_box[3]:
            return True, "该颜色连通块的中心位于高亮格"
        overlap = max(0, min(grid_box[2], box[2]) - max(grid_box[0], box[0])) * max(
            0, min(grid_box[3], box[3]) - max(grid_box[1], box[1]))
        best = max(best, overlap / ((box[2] - box[0]) * (box[3] - box[1])))
    if best >= 0.5:
        return True, f"高亮格覆盖了目标颜色连通块约{best:.0%}的包围盒"
    return False, f"高亮格与目标颜色连通块最多重叠约{best:.0%}，请重新定位"


def build_questions(gt):
    loc, count, exist = [], [], []
    for index in range(EVAL_N):
        img = f"e{index:02d}"
        items = gt[img]
        target = sorted({item["class"] for item in items})[index % len({item["class"] for item in items})]
        box = next(item["box"] for item in items if item["class"] == target)
        name = f"{target[0]}色{target[1:]}形"
        base = {"img": img, "target": target}
        loc.append({**base, "id": f"L{index + 1:02d}", "type": "loc", "q":
                    f"图中的{name}位于第几行第几列？请用「第X行第Y列」回答（共4行4列）。",
                    "gold": cell_text(cell_of(box)), "cell": list(cell_of(box)), "box": box})
        n = sum(item["class"] == target for item in items)
        count.append({**base, "id": f"C{index + 1:02d}", "type": "count", "q":
                      f"图中有几个{name}？请回答数字和个。", "gold": f"{n}个", "n": n})
        exist.append({**base, "id": f"E{index + 1:02d}", "type": "exist", "q":
                      f"图里有没有{name}？请回答有或没有。", "gold": "有"})
    return loc[:8] + count[:4] + exist[:3]


def judge(q, prediction):
    text = str(prediction).strip().lower()
    if q["type"] == "loc":
        return parse_cell(text) == tuple(q["cell"])
    if q["type"] == "count":
        found = re.search(r"(?<!\d)\d+(?!\d)", text)
        return bool(found) and int(found[0]) == q["n"]
    yes = ("有" in text and "没有" not in text.replace("有没有", "")) or bool(re.search(r"\byes\b", text))
    no = "没有" in text.replace("有没有", "") or bool(re.search(r"\b(no|not)\b", text))
    return yes if q["gold"] == "有" else no


def loc_iou(q, prediction):
    cell = parse_cell(prediction)
    return iou(cell_box(cell), q["box"]) if cell is not None else 0.0


def make_sft(gt=None):
    gt = gt or load_gt()
    samples = [{"img": q["img"], "prompt": q["q"], "answer": q["gold"],
                "source": "seed_gt", "question_id": q["id"]} for q in build_questions(gt)]
    save_json(SFT_JSON, samples)
    print(f"[sft] GT 种子样本 {len(samples)} 条: sft_data.json")
    return samples


def image_for(q):
    return Image.open(IMG_DIR / "eval" / f"{q['img']}.png").convert("RGB")


def prompt_text(processor, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def load_model(adapter=None, trainable=False):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = ModelClass.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def trainable_stats(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "percent": 100 * trainable / total}


def answer(processor, model, image, question, hint=None, max_new=24):
    import torch
    model.eval()
    if hint:
        question = f"{question}\n校验提示：{hint}"
    inputs = processor(text=[prompt_text(processor, question)], images=[image], return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def train_steps(processor, model, samples, steps, adapter_path, tag):
    import torch
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    stats = trainable_stats(model)
    print(f"[{tag}] 可训练参数 {stats['trainable']}/{stats['total']} ({stats['percent']:.4f}%)")
    losses = []
    started = time.monotonic()
    for index in range(steps):
        sample = samples[index % len(samples)]
        image = image_for(sample)
        prefix = prompt_text(processor, sample["prompt"])
        full = processor(text=[prefix + sample["answer"]], images=[image], return_tensors="pt")
        plen = processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :plen] = -100
        full["labels"] = labels
        full = full.to(next(model.parameters()).device)
        model.train()
        loss = model(**full).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))
        print(f"[{tag}] step {index + 1}/{steps} loss={losses[-1]:.4f}", flush=True)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    return {"tag": tag, "steps": steps, "samples": len(samples), "params": stats,
            "losses": losses, "seconds": round(time.monotonic() - started, 2),
            "adapter": str(adapter_path.relative_to(ROOT))}


def score_rows(tag, rows):
    by_type = {}
    for kind in ("loc", "count", "exist"):
        part = [r for r in rows if r["type"] == kind]
        by_type[kind] = {"correct": sum(r["correct"] for r in part), "total": len(part)}
    loc_rows = [r for r in rows if r["type"] == "loc"]
    return {"stage": tag, "by_type": by_type, "correct": sum(r["correct"] for r in rows),
            "total": len(rows), "loc_mean_iou": sum(r["iou"] for r in loc_rows) / len(loc_rows)}


def evaluate(processor, model, questions, tag, correction=False):
    rows = []
    started = time.monotonic()
    for q in questions:
        img = image_for(q)
        first = answer(processor, model, img, q["q"])
        first_ok = judge(q, first)
        pred, hint, verifier_pass = first, None, None
        if correction and not first_ok:
            hint, verifier_pass = tool_hint(img, q, first)
            pred = answer(processor, model, img, q["q"], hint=hint)
        row = {"id": q["id"], "type": q["type"], "img": q["img"], "question": q["q"],
               "gold": q["gold"], "first": first, "first_correct": first_ok,
               "hint": hint, "verifier_pass": verifier_pass, "pred": pred,
               "correct": judge(q, pred), "iou": loc_iou(q, pred) if q["type"] == "loc" else None}
        rows.append(row)
        print(f"[{tag} {q['id']}] {'√' if row['correct'] else '×'} {first!r}"
              + (f" -> {pred!r}" if correction and not first_ok else "")
              + (f" IoU={row['iou']:.3f}" if q["type"] == "loc" else ""), flush=True)
    summary = score_rows(tag, rows)
    summary["seconds"] = round(time.monotonic() - started, 2)
    summary["correction"] = {"triggered": sum(not r["first_correct"] for r in rows),
                             "wrong_to_right": sum(not r["first_correct"] and r["correct"] for r in rows),
                             "right_to_wrong": sum(r["first_correct"] and not r["correct"] for r in rows)}
    print(f"[{tag}] {summary['by_type']} total={summary['correct']}/{summary['total']}"
          f" loc_IoU={summary['loc_mean_iou']:.3f}")
    return {"summary": summary, "rows": rows}


def tool_hint(image, q, pred):
    if q["type"] == "loc":
        passed, msg = verify_cell(image, q, parse_cell(pred))
        return msg, passed
    components = color_components(image, q["target"][0])
    if q["type"] == "count":
        return f"颜色分析发现该颜色有{len(components)}个连通块；注意同色不同形状可能混淆。", None
    return f"颜色分析发现该颜色像素{'存在' if components else '不存在'}。", None


def overlay(image, cell, path):
    result = image.copy()
    pen = ImageDraw.Draw(result)
    box = cell_box(cell)
    pen.rectangle(box, outline=(255, 0, 255), width=5)
    pen.rectangle([box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6],
                  outline=(255, 0, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    return result


def rsi_round(processor, model, questions, round_no, pool):
    rows, new_samples = [], []
    started = time.monotonic()
    for q in questions:
        image = image_for(q)
        first = answer(processor, model, image, q["q"])
        first_ok = judge(q, first)
        current, path = first, "direct"
        self_pred = tool_pred = hint = None
        self_changed = self_success = tool_triggered = tool_success = False
        verifier_pass = None
        if q["type"] == "loc" and not first_ok:
            cell = parse_cell(first)
            if cell is not None:
                ov = overlay(image, cell, IMG_DIR / "overlays" / f"round{round_no}_{q['id']}.png")
                self_pred = answer(processor, model, ov,
                    f"紫色高亮格是你上一次给的位置。请检查是否圈住{q['target'][0]}色{q['target'][1:]}形。"
                    f"若不对，请改写。{q['q']}", max_new=24)
                self_changed = parse_cell(self_pred) is not None and parse_cell(self_pred) != cell
                if self_changed:
                    current, path = self_pred, "self"
                    verifier_pass, _ = verify_cell(image, q, parse_cell(self_pred))
                    self_success = verifier_pass and judge(q, self_pred)
        if not judge(q, current):
            hint, before_pass = tool_hint(image, q, current)
            tool_triggered = True
            tool_pred = answer(processor, model, image, q["q"], hint=hint)
            current, path = tool_pred, "tool"
            if q["type"] == "loc":
                verifier_pass, _ = verify_cell(image, q, parse_cell(tool_pred))
            tool_success = judge(q, tool_pred) and (verifier_pass is not False)
        final_ok = judge(q, current)
        accepted = final_ok and path in ("self", "tool") and (q["type"] != "loc" or verifier_pass is True)
        if accepted:
            new_samples.append({"img": q["img"], "prompt": q["q"], "answer": current,
                                "source": f"rsi_round_{round_no}", "question_id": q["id"],
                                "path": path})
        row = {"round": round_no, "id": q["id"], "type": q["type"], "img": q["img"],
               "question": q["q"], "gold": q["gold"], "first": first,
               "first_correct": first_ok, "first_iou": loc_iou(q, first) if q["type"] == "loc" else None,
               "self_pred": self_pred, "self_changed": self_changed, "self_success": self_success,
               "tool_triggered": tool_triggered, "tool_hint": hint, "tool_pred": tool_pred,
               "tool_success": tool_success, "verifier_pass": verifier_pass,
               "final": current, "final_correct": final_ok, "final_iou": loc_iou(q, current) if q["type"] == "loc" else None,
               "path": path, "added_to_pool": accepted}
        rows.append(row)
        print(f"[RSI {round_no} {q['id']}] {'√' if final_ok else '×'} direct={first!r}"
              f" self={self_pred!r} tool={tool_pred!r} path={path}", flush=True)
    direct = [{"type": r["type"], "correct": r["first_correct"],
               "iou": r["first_iou"]} for r in rows]
    after = [{"type": r["type"], "correct": r["final_correct"],
              "iou": r["final_iou"]} for r in rows]
    metrics = {"round": round_no, "pool_before": len(pool), "pool_added": len(new_samples),
               "pool_after": len(pool) + len(new_samples), "direct": score_rows(f"RSI {round_no} direct", direct),
               "after_correction": score_rows(f"RSI {round_no} corrected", after),
               "self_attempts": sum(r["self_pred"] is not None for r in rows),
               "self_changed": sum(r["self_changed"] for r in rows),
               "self_success": sum(r["self_success"] for r in rows),
               "tool_triggered": sum(r["tool_triggered"] for r in rows),
               "tool_success": sum(r["tool_success"] for r in rows),
               "wrong_to_right": sum(not r["first_correct"] and r["final_correct"] for r in rows),
               "right_to_wrong": sum(r["first_correct"] and not r["final_correct"] for r in rows),
               "seconds": round(time.monotonic() - started, 2)}
    print(f"[RSI {round_no}] {metrics}")
    return rows, new_samples, metrics


def environment():
    import torch
    import transformers
    import peft
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "pillow": Image.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model": MODEL_ID, "training_precision": "fp32", "quantization": "none"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["make_data", "make_sft", "infer", "sft", "eval", "rsi"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.rounds < 1:
        parser.error("--steps 和 --rounds 必须为正整数")
    started = time.monotonic()
    if args.mode == "make_data":
        make_data()
        return
    gt = load_gt()
    if args.mode == "make_sft":
        make_sft(gt)
        return
    questions = build_questions(gt)
    RESULTS.mkdir(exist_ok=True)
    print("[env]", environment(), flush=True)
    if args.mode == "infer":
        processor, model = load_model()
        result = evaluate(processor, model, questions, "baseline")
        save_json(RESULTS / "baseline.json", result)
    elif args.mode == "sft":
        samples = read_json(SFT_JSON) if SFT_JSON.exists() else make_sft(gt)
        processor, model = load_model(trainable=True)
        result = train_steps(processor, model, samples, args.steps, SFT_DIR, "sft")
        result["environment"] = environment()
        save_json(RESULTS / "sft_training.json", result)
    elif args.mode == "eval":
        if not SFT_DIR.exists():
            raise SystemExit("缺少 sft_adapter；请先运行 --mode sft。")
        processor, base = load_model()
        baseline = evaluate(processor, base, questions, "baseline")
        del base
        processor, tuned = load_model(adapter=SFT_DIR)
        sft = evaluate(processor, tuned, questions, "sft")
        corrected = evaluate(processor, tuned, questions, "sft_tool", correction=True)
        result = {"environment": environment(), "stages": [baseline, sft, corrected],
                  "seconds": round(time.monotonic() - started, 2)}
        save_json(RESULTS / "evaluation.json", result)
    else:
        if not SFT_DIR.exists():
            raise SystemExit("缺少 sft_adapter；请先运行 --mode sft。")
        pool = read_json(SFT_JSON) if SFT_JSON.exists() else make_sft(gt)
        processor, model = load_model(adapter=SFT_DIR, trainable=True)
        rounds = []
        memories = []
        for number in range(1, args.rounds + 1):
            rows, added, metrics = rsi_round(processor, model, questions, number, pool)
            memories.extend(rows)
            pool.extend(added)
            save_json(SFT_JSON, pool)
            training = train_steps(processor, model, pool, args.steps, RSI_DIR, f"rsi{number}")
            ending = evaluate(processor, model, questions, f"rsi_round_{number}")
            metrics["training"] = training
            metrics["end_eval"] = ending
            rounds.append(metrics)
            save_json(MEM_JSON, memories)
            save_json(RESULTS / "rsi_metrics.json", {"environment": environment(),
                      "rounds": rounds, "seconds_so_far": round(time.monotonic() - started, 2)})
        print(f"[rsi] 完成 {len(rounds)} 轮；数据池 {len(pool)} 条；记忆 {len(memories)} 条")
    print(f"[done] {args.mode} 耗时 {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
