"""SmolVLM-256M 的 QLoRA 微调与 DPO 偏好对齐实验。

按作业五指南实现：15 个确定性合成图像、SFT、DPO、三阶段评测和手算复核。
在 CUDA 可用时采用 4-bit NF4 QLoRA；CPU 仅用于功能演示并明确回退为 fp32 LoRA。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ROOT = Path(__file__).resolve().parent
IMG_DIR, RESULT_DIR = ROOT / "hw5_imgs", ROOT / "results"
SFT_DIR, DPO_DIR = ROOT / "sft_adapter", ROOT / "dpo_adapter"
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)], "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)], "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)], "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)], "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)], "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)], "img12": [("红", "方", 1), ("蓝", "方", 1)],
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)], "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}
QUESTIONS = [
    ("P01", "count", "img01", "图中有几个红色圆形？", "2"), ("P02", "count", "img02", "图中有几个绿色圆形？", "3"),
    ("P03", "count", "img03", "图中有几个黄色方形？", "2"), ("P04", "count", "img06", "图中有几个黄色圆形？", "4"),
    ("P05", "count", "img10", "图中有几个蓝色圆形？", "3"), ("P06", "attr", "img01", "图中蓝色的图形是什么形状？", "方形"),
    ("P07", "attr", "img05", "图中红色的图形是什么形状？", "圆形"), ("P08", "attr", "img09", "图中绿色的图形是什么形状？", "方形"),
    ("P09", "attr", "img13", "图中黄色的图形是什么形状？", "圆形"), ("P10", "attr", "img14", "图中蓝色的图形是什么形状？", "方形"),
    ("P11", "exist", "img11", "图里有没有黄色方形？", "有"), ("P12", "exist", "img12", "图里有没有红色圆形？", "没有"),
    ("P13", "exist", "img07", "图里有没有绿色圆形？", "有"), ("P14", "exist", "img08", "图里有没有绿色方形？", "没有"),
    ("P15", "exist", "img15", "图里有没有蓝色图形？", "没有"),
]
SFT_DATA = [{"img": i, "prompt": q, "answer": f"{g}个。" if t == "count" else f"{g}。"} for _, t, i, q, g in QUESTIONS]
PRE_PAIRS = [
    ("img01", "图中有几个红色圆形？", "2 个。", "3 个。图中有三个红色圆形整齐排列。", "数错"),
    ("img02", "图中有几个绿色圆形？", "3 个。", "2 个。", "数错"), ("img03", "图中有几个黄色方形？", "2 个。", "没有看到黄色图形。", "数错"),
    ("img06", "图中有几个黄色圆形？", "4 个。", "3 个。", "数错"), ("img10", "图中有几个蓝色圆形？", "3 个。", "图中有三个蓝色方形。", "形状幻觉"),
    ("img01", "图中蓝色的图形是什么形状？", "方形。", "圆形。", "形状幻觉"), ("img05", "图中红色的图形是什么形状？", "圆形。", "方形，而且有两个红色方形。", "形状幻觉"),
    ("img09", "图中绿色的图形是什么形状？", "方形。", "图中没有绿色图形。", "形状幻觉"), ("img11", "图里有没有黄色方形？", "有。", "没有。", "存在性误判"),
    ("img12", "图里有没有红色圆形？", "没有。", "有，图左上角有一个红色圆形。", "硬答有"), ("img08", "图里有没有绿色方形？", "没有。", "有，一共两个绿色方形。", "硬答有"),
    ("img15", "图里有没有蓝色图形？", "没有。图中只有黄色和绿色的圆形。", "有，两个蓝色圆形。", "硬答有"),
]
# 自拟 5 组：与指南给出的 12 组分开，便于报告逐组说明失败模式。
STUDENT_PAIRS = [
    ("img04", "图中有几个红色方形？", "3 个。", "2 个。", "数错"), ("img07", "图中蓝色的图形是什么形状？", "方形。", "圆形。", "形状幻觉"),
    ("img13", "图里有没有蓝色方形？", "没有。", "有，一个蓝色方形。", "硬答有"), ("img14", "图中有几个蓝色方形？", "3 个。", "图中有很多蓝色图形，无法准确判断。", "冗长废话"),
    ("img05", "图里有没有黄色圆形？", "没有。", "有，黄色圆形在图片中央。", "硬答有"),
]

def make_images():
    IMG_DIR.mkdir(parents=True, exist_ok=True); rnd = random.Random(42)
    for image_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white"); draw = ImageDraw.Draw(im)
        for color, shape, count in spec:
            for _ in range(count):
                x, y, size = rnd.randint(10, 200), rnd.randint(10, 200), rnd.randint(30, 50)
                box = [x, y, x + size, y + size]
                (draw.ellipse if shape == "圆" else draw.rectangle)(box, outline=COLORS[color], width=4)
        im.save(IMG_DIR / f"{image_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像：{IMG_DIR}")

def image(image_id):
    if not (IMG_DIR / f"{image_id}.png").exists(): make_images()
    return Image.open(IMG_DIR / f"{image_id}.png").convert("RGB")

def prompt(processor, question):
    return processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}], add_generation_prompt=True)

def load_model(train=False, rank=8, adapter=None, trainable=False):
    # Transformers 4.x used AutoModelForVision2Seq; 5.x renamed the public
    # multimodal factory to AutoModelForImageTextToText.
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(MODEL_ID); cuda = torch.cuda.is_available()
    kwargs = {"torch_dtype": torch.float16 if cuda else torch.float32}
    if train and cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        print("[env] CUDA + NF4 4-bit QLoRA")
    elif train: print("[env] CPU fp32 + LoRA 演示（非 NF4）")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        if trainable: model.print_trainable_parameters()
    elif train:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if cuda: model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    if not (train and cuda): model.to("cuda" if cuda else "cpu")
    return processor, model

@torch.no_grad()
def answer(processor, model, im, question):
    inputs = processor(text=[prompt(processor, question)], images=[im], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def correct(gold, pred): return gold in pred.replace(" ", "")

def tokenized(processor, im, question, response):
    full = processor(text=[prompt(processor, question) + response], images=[im], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    plen = processor(text=[prompt(processor, question)], images=[im], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone(); labels[:, :plen] = -100; return full, labels

def seq_logprob(model, processor, im, question, response):
    full, labels = tokenized(processor, im, question, response); logits = model(**full).logits[:, :-1].float(); targets = labels[:, 1:]
    values = torch.log_softmax(logits, -1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return values[targets.ne(-100)].sum()

def save_curve(name, values):
    RESULT_DIR.mkdir(exist_ok=True); (RESULT_DIR / f"{name}_curve.json").write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

def sft(args):
    processor, model = load_model(True, args.rank); opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4); rows=[]; model.train()
    for step in range(args.steps):
        sample = SFT_DATA[step % len(SFT_DATA)]; full, labels = tokenized(processor, image(sample["img"]), sample["prompt"], sample["answer"]); full["labels"] = labels
        loss = model(**full).loss; loss.backward(); opt.step(); opt.zero_grad(); rows.append({"step": step + 1, "loss": loss.item()})
        print(f"[sft] step {step+1}/{args.steps} loss={loss.item():.4f}")
    model.save_pretrained(SFT_DIR); save_curve("sft", rows); print(f"[sft] adapter: {SFT_DIR}")

def dpo(args):
    if not SFT_DIR.exists(): raise SystemExit("请先运行 --mode sft 生成 SFT adapter。")
    processor, model = load_model(True, args.rank, SFT_DIR, True); opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5); rows=[]; pairs = PRE_PAIRS + STUDENT_PAIRS; model.train()
    for step in range(args.steps):
        im, question, chosen, rejected, _ = pairs[step % len(pairs)]; im=image(im)
        pw, pl = seq_logprob(model, processor, im, question, chosen), seq_logprob(model, processor, im, question, rejected)
        with torch.no_grad(), model.disable_adapter(): rw, rl = seq_logprob(model, processor, im, question, chosen), seq_logprob(model, processor, im, question, rejected)
        margin = args.beta * ((pw-rw)-(pl-rl)); loss = -F.logsigmoid(margin); loss.backward(); opt.step(); opt.zero_grad()
        rows.append({"step": step+1, "loss": loss.item(), "margin": margin.item()}); print(f"[dpo] step {step+1}/{args.steps} loss={loss.item():.4f} margin={margin.item():.4f}")
    model.save_pretrained(DPO_DIR); save_curve("dpo", rows); print(f"[dpo] adapter: {DPO_DIR}")

def evaluate_one(processor, model, tag):
    rows=[]; print(f"\n===== {tag} × 15 =====")
    for pid, kind, iid, q, gold in QUESTIONS:
        pred=answer(processor, model, image(iid), q); ok=correct(gold,pred); rows.append({"id":pid,"type":kind,"gold":gold,"pred":pred,"correct":ok}); print(f"[{pid}·{kind}] {'√' if ok else '×'} {pred} (gold: {gold})")
    summary={kind: sum(r['correct'] for r in rows if r['type']==kind) for kind in ('count','attr','exist')}; print(f"{tag}: count {summary['count']}/5, attr {summary['attr']}/5, exist {summary['exist']}/5, total {sum(summary.values())}/15")
    return rows, summary

def evaluate(args):
    all_rows={}; processor, model=load_model(); all_rows['baseline'], _=evaluate_one(processor,model,'baseline')
    if SFT_DIR.exists(): processor,model=load_model(True,args.rank,SFT_DIR); all_rows['sft'],_=evaluate_one(processor,model,'sft')
    if DPO_DIR.exists(): processor,model=load_model(True,args.rank,DPO_DIR); all_rows['sft_dpo'],_=evaluate_one(processor,model,'sft+dpo')
    RESULT_DIR.mkdir(exist_ok=True); (RESULT_DIR/'evaluation.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),encoding='utf-8')

def dpo_check(value):
    pw,rw,pl,rl,beta=map(float,value.split(',')); margin=beta*((pw-rw)-(pl-rl)); loss=-math.log(1/(1+math.exp(-margin)))
    print(f"margin = {margin:.4f}\nDPO loss = {loss:.4f}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['make_data','infer','sft','dpo','eval'],default='infer'); ap.add_argument('--steps',type=int,default=None); ap.add_argument('--beta',type=float,default=.1); ap.add_argument('--rank',type=int,default=8); ap.add_argument('--dpo_check','--dpo-check',dest='dpo_check'); args=ap.parse_args()
    if args.dpo_check: return dpo_check(args.dpo_check)
    if args.mode=='make_data': return make_images()
    if args.mode=='infer':
        p,m=load_model(); return evaluate_one(p,m,'baseline')
    if args.mode=='sft': args.steps=args.steps or 60; return sft(args)
    if args.mode=='dpo': args.steps=args.steps or 40; return dpo(args)
    evaluate(args)
if __name__=='__main__': main()
