# -*- coding: utf-8 -*-
"""Vector RAG、GraphRAG 与 WikiRAG 对比实验。

默认使用已下载的 paraphrase-multilingual-MiniLM-L12-v2；加载失败时会明确
打印提示并降级为字符 3-gram。评测统一把各范式的检索单元映射回文档编号。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np


CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于1958年，是省属重点大学，现有全日制本科生约2.1万人。校训为格物致知。",
    "doc02": "人工智能学院成立于2018年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为1985年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码AI2101，3学分、32学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码AI3305，2学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修30学分；GPA低于2.0给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金8000元/年，评定比例约2%；校长奖学金20000元/年，全校每年10人；云山一等奖学金3000元/年，比例约5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人竞赛。",
    "doc12": "校历：春季学期3月2日开学，7月5日放暑假；秋季学期9月1日开学，次年1月15日放寒假。",
    "doc13": "图书馆藏书380万册，开放时间为每天7:00-22:00；人工智能分馆位于理科楼B座3层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁3号线云大站距东门200米；校内校巴共5条线路，10分钟一班。",
    "doc15": "国际交流：学校与12个国家的47所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}


QUESTIONS = [
    {"id": "Q01", "type": "multi", "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？", "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi", "q": "AI协会的指导教师主讲课程的课程代码是什么？", "evidence": ["doc11", "doc04", "doc06"]},
    {"id": "Q03", "type": "multi", "q": "认知计算全国重点实验室依托的学院成立于哪一年？", "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi", "q": "《知识图谱导论》授课教师的研究方向是什么？", "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi", "q": "机器人战队的指导教师的研究方向是什么？", "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact", "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact", "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact", "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact", "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact", "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？", "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？", "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？", "evidence": ["doc06", "doc11"]},
    # 五道自拟题：事实 2、多跳 2、全局 1。
    {"id": "Q16", "type": "fact", "q": "春季学期哪一天放暑假？", "evidence": ["doc12"]},
    {"id": "Q17", "type": "fact", "q": "云山大学与多少所海外高校签有交换协议？", "evidence": ["doc15"]},
    {"id": "Q18", "type": "multi", "q": "《知识图谱导论》授课教师所在学院的院长是谁？", "evidence": ["doc07", "doc04", "doc02"]},
    {"id": "Q19", "type": "multi", "q": "认知计算全国重点实验室依托学院的院长是谁？", "evidence": ["doc10", "doc02"]},
    {"id": "Q20", "type": "global", "q": "学校为学生提供哪些社团活动和学习资源？", "evidence": ["doc11", "doc13"]},
]


PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"),
    ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019年", "doc04"),
    ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"),
    ("李文瀚", "指导", "AI协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"),
    ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"),
    ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"),
    ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"),
    ("人工智能学院", "成立于", "2018年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"),
    ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"),
    ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"),
    ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"),
    ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"),
    ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"),
    ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3学分", "doc06"),
    ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"),
    ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI协会", "指导教师", "李文瀚", "doc11"),
    ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000元/年", "doc09"),
    ("校长奖学金", "金额", "20000元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000元/年", "doc09"),
    ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958年", "doc01"),
    ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380万册", "doc13"),
    ("春季学期", "开学日期", "3月2日", "doc12"),
]


COMMUNITY_SUMMARIES = [
    ("社区C1 教学与课程：人工智能学院设三个系，开设大模型通识课和知识图谱导论，课程、教师与先修信息相互关联。", ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建，学院还有国际联合培养。", ["doc02", "doc03", "doc10", "doc15"]),
    ("社区C3 学生生活：奖学金分国家、校长和云山一等奖学金；社团有AI协会和机器人战队；图书馆提供学习资源。", ["doc09", "doc11", "doc13"]),
]


PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于1958年，是省属重点大学，本科生约2.1万人，校训为格物致知。地铁3号线云大站距东门200米，校巴5条线路。春季学期3月2日开学。", ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于2018年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托学院建设，并与南洋理工大学开展本科联合培养。", ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身为1985年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程专业，并参与岭南超算中心运行管理。", ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚教授研究检索增强生成，主讲大模型通识课并指导AI协会；赵婉晴副教授研究知识图谱，主讲知识图谱导论；孙浩然讲师研究强化学习并指导机器人战队。", ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇教授研究分布式系统；林小雨副教授研究数据库系统。", ["doc05"]),
    ("大模型通识课", "大模型通识课代码AI2101，3学分32学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求。", ["doc06"]),
    ("知识图谱导论", "知识图谱导论代码AI3305，2学分，秋季学期开设，赵婉晴主讲，面向人工智能学院研究生，需先修数据结构。", ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金8000元/年，校长奖学金20000元/年，云山一等奖学金3000元/年。", ["doc09"]),
    ("学生社团与校园生活", "AI协会由李文瀚指导并开展论文研读，机器人战队由孙浩然指导并参加全国竞赛。图书馆藏书380万册，每天7:00-22:00开放，设人工智能分馆。", ["doc11", "doc13"]),
]


def chunk_docs(size: int = 120, overlap: int = 20):
    chunks = []
    step = size - overlap
    for doc_id, text in CORPUS.items():
        i = 0
        while True:
            chunks.append((doc_id, text[i : i + size]))
            if i + size >= len(text):
                break
            i += step
    return chunks


class Embedder:
    def __init__(self, model_name: str, force_fallback: bool = False):
        self.ok = False
        self.model_name = model_name
        if not force_fallback:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name, local_files_only=True)
                self.ok = True
                print(f"[embed] 已加载真实嵌入模型：{model_name}")
            except Exception as exc:
                print(f"[embed] 加载失败（{type(exc).__name__}），降级为字符3-gram；该结果不用于K敏感性结论。")

    def encode(self, texts):
        texts = list(texts)
        if self.ok:
            return self.model.encode(texts, normalize_embeddings=True)
        matrix = np.zeros((len(texts), 512), dtype=np.float32)
        for row, text in enumerate(texts):
            text = re.sub(r"\s", "", text)
            for i in range(max(1, len(text) - 2)):
                gram = text[i : i + 3] if len(text) >= 3 else text
                matrix[row, zlib.crc32(gram.encode("utf-8")) % 512] += 1
            norm = np.linalg.norm(matrix[row]) or 1.0
            matrix[row] /= norm
        return matrix


def _dedup(scored, k):
    docs, seen = [], set()
    for doc_id, score in scored:
        if doc_id not in seen:
            docs.append((doc_id, float(score)))
            seen.add(doc_id)
            if len(docs) >= k:
                break
    return docs


def _bigrams(text):
    clean = re.sub(r"\W", "", text.lower())
    return {clean[i : i + 2] for i in range(max(1, len(clean) - 1))}


class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb, hybrid=False, size=120, overlap=20):
        self.emb = emb
        self.hybrid = hybrid
        self.chunks = chunk_docs(size, overlap)
        self.matrix = emb.encode(text for _, text in self.chunks)
        self.last_hits = []

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        vector_scores = self.matrix @ qv
        vector_order = vector_scores.argsort()[::-1]
        if not self.hybrid:
            hits = [(self.chunks[i][0], vector_scores[i]) for i in vector_order]
        else:
            qgrams = _bigrams(query)
            lexical_scores = np.array([
                len(qgrams & _bigrams(text)) / max(1, len(qgrams | _bigrams(text)))
                for _, text in self.chunks
            ])
            lexical_order = lexical_scores.argsort()[::-1]
            rrf = np.zeros(len(self.chunks), dtype=np.float32)
            for rank, idx in enumerate(vector_order, start=1):
                rrf[idx] += 1.0 / (60 + rank)
            for rank, idx in enumerate(lexical_order, start=1):
                rrf[idx] += 1.0 / (60 + rank)
            hits = [(self.chunks[i][0], rrf[i]) for i in rrf.argsort()[::-1]]
        self.last_hits = _dedup(hits, k)
        return [doc_id for doc_id, _ in self.last_hits]


class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb):
        import networkx as nx

        self.emb = emb
        self.graph = nx.DiGraph()
        for head, relation, tail, source in PRE_TRIPLES:
            self.graph.add_edge(head, tail, rel=relation, src=source)
        communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
        self.community_count = len(communities)
        self.summary_matrix = emb.encode(summary for summary, _ in COMMUNITY_SUMMARIES)
        self.last_path = []
        print(f"[graph] 实体{self.graph.number_of_nodes()}个 / 三元组{len(PRE_TRIPLES)}条 / 社区{self.community_count}个")

    def _entities(self, query, topn=3):
        aliases = {"云大": "云山大学", "大模型通识课": "大模型通识课", "知识图谱导论": "知识图谱导论"}
        normalized = query
        for alias, canonical in aliases.items():
            normalized = normalized.replace(alias, canonical)
        entities = sorted((node for node in self.graph.nodes if node in normalized), key=len, reverse=True)
        return entities[:topn]

    def retrieve(self, query, k=5, scope="local"):
        return self._local(query, k) if scope == "local" else self._global(query, k)

    def _local(self, query, k):
        entities = self._entities(query)
        seen, scored = set(), []
        frontier = list(entities)
        self.last_path = []
        for _ in range(3):
            nxt = []
            for entity in frontier:
                edges = list(self.graph.out_edges(entity, data=True)) + list(self.graph.in_edges(entity, data=True))
                for head, tail, data in edges:
                    key = (head, data["rel"], tail)
                    if key in seen:
                        continue
                    seen.add(key)
                    triple_text = f"{head} -{data['rel']}-> {tail}"
                    scored.append((data["src"], triple_text))
                    self.last_path.append((head, data["rel"], tail, data["src"]))
                    other = tail if head == entity else head
                    if other not in frontier:
                        nxt.append(other)
            nxt.sort(key=lambda node: 0 if self.graph.out_degree(node) > 0 else 1)
            frontier = nxt[:8]
        if not scored:
            return self._global(query, k)
        qv = self.emb.encode([query])[0]
        similarities = self.emb.encode(text for _, text in scored) @ qv
        best = {}
        for (doc_id, _), score in zip(scored, similarities):
            best[doc_id] = max(float(score), best.get(doc_id, -1.0))
        return [doc_id for doc_id, _ in sorted(best.items(), key=lambda item: -item[1])[:k]]

    def _global(self, query, k):
        qv = self.emb.encode([query])[0]
        scores = self.summary_matrix @ qv
        docs = []
        for index in scores.argsort()[::-1][:2]:
            for doc_id in COMMUNITY_SUMMARIES[index][1]:
                if doc_id not in docs:
                    docs.append(doc_id)
        return docs[:k]


class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb):
        self.emb = emb
        self.matrix = emb.encode(body for _, body, _ in PRE_ENTRIES)
        self.last_entries = []

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.matrix @ qv
        order = scores.argsort()[::-1][:3]
        self.last_entries = [(PRE_ENTRIES[i][0], float(scores[i])) for i in order]
        docs = []
        for index in order:
            for doc_id in PRE_ENTRIES[index][2]:
                if doc_id not in docs:
                    docs.append(doc_id)
        return docs[:k]


def retrieve_for_question(system, question, k):
    if isinstance(system, GraphRAG):
        scope = "global" if question["type"] == "global" else "local"
        return system.retrieve(question["q"], k, scope)
    return system.retrieve(question["q"], k)


def evaluate(systems, k=5, questions=None):
    questions = questions or QUESTIONS
    output = {"k": k, "question_count": len(questions), "systems": {}}
    print(f"\n===== 检索评测（文档级，Top-{k}，{len(questions)}题） =====")
    print(f"{'范式':<14}{'问题类型':<10}{'Recall@'+str(k):>10}{'MRR':>9}{'完整命中':>10}")
    for name, system in systems.items():
        system_rows = {}
        for question_type in ("fact", "multi", "global"):
            selected = [q for q in questions if q["type"] == question_type]
            recalls, reciprocal_ranks, full_hits, details = [], [], 0, []
            for question in selected:
                docs = retrieve_for_question(system, question, k)
                evidence = set(question["evidence"])
                found = evidence & set(docs[:k])
                recall = len(found) / len(evidence)
                rank = next((i + 1 for i, doc_id in enumerate(docs) if doc_id in evidence), 0)
                rr = 1.0 / rank if rank else 0.0
                recalls.append(recall)
                reciprocal_ranks.append(rr)
                full_hits += found == evidence
                details.append({"id": question["id"], "retrieved": docs, "evidence": question["evidence"], "recall": recall, "rank": rank, "rr": rr})
            row = {
                "recall": sum(recalls) / len(recalls),
                "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "full_hits": full_hits,
                "count": len(selected),
                "details": details,
            }
            system_rows[question_type] = row
            print(f"{name:<14}{question_type:<10}{row['recall']:>10.3f}{row['mrr']:>9.3f}{full_hits:>5}/{len(selected):<4}")
        output["systems"][name] = system_rows
    return output


def llm_chat(prompt, temperature=0.0):
    from openai import OpenAI

    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        raise SystemExit("未配置LLM：请设置LLM_API_BASE、LLM_API_KEY、LLM_MODEL")
    client = OpenAI(base_url=base, api_key=key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


EXTRACT_PROMPT = """从下面大学介绍文档中抽取[头实体-关系-尾实体]三元组，只输出JSON数组，不输出其他文字。
格式如[["李文瀚","任职于","人工智能学院"]]。关系用2~4字短语，一篇文档抽5~10条。
文档：{doc}"""

ANSWER_PROMPT = """你是问答助手。只依据资料回答；资料不足时明确说“资料中未提及”，禁止编造。
问题：{q}
资料：
{ctx}
用1~3句话回答，句末用括号注明资料编号。"""


def extract_triples():
    started, output, bad = time.time(), [], 0
    for doc_id, text in CORPUS.items():
        raw = llm_chat(EXTRACT_PROMPT.replace("{doc}", text))
        try:
            triples = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
        except Exception:
            bad += 1
            print(f"[warn] {doc_id}解析失败：{raw[:60]}...")
            continue
        output.extend((head, relation, tail, doc_id) for head, relation, tail in triples)
        print(f"{doc_id}：抽到{len(triples)}条")
    print(f"共{len(output)}条 / 解析失败{bad}篇 / 耗时{time.time()-started:.0f}s")


def save_json(data, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")


def build_systems(emb, hybrid=False):
    return {
        "Vector RAG" + (" + RRF" if hybrid else ""): VectorRAG(emb, hybrid=hybrid),
        "GraphRAG": GraphRAG(emb),
        "WikiRAG": WikiRAG(emb),
    }


def main():
    parser = argparse.ArgumentParser(description="实验作业四：三种RAG对比实验")
    parser.add_argument("--mode", choices=["eval", "vector", "graph", "wiki", "extract"], default="eval")
    parser.add_argument("--q", default="Q01", help="问题编号，如Q01")
    parser.add_argument("--scope", choices=["local", "global"], default="local")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--emb", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--mrr_check", default=None, help="如3,1,2，打印MRR手算过程")
    parser.add_argument("--builtin-only", action="store_true", help="只评测内置15题，用于K扫描")
    parser.add_argument("--hybrid", action="store_true", help="Vector RAG使用字符bigram与向量RRF融合")
    parser.add_argument("--force-fallback", action="store_true", help="强制使用字符3-gram降级嵌入")
    parser.add_argument("--save-json", default=None, help="保存评测明细JSON")
    parser.add_argument("--scan-k", action="store_true", help="一次运行K=1/2/3/5敏感性扫描（内置15题）")
    parser.add_argument("--benchmark-build", action="store_true", help="记录三种检索器首次构建耗时")
    args = parser.parse_args()

    if args.mrr_check:
        ranks = [int(value) for value in args.mrr_check.split(",")]
        mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
        print("MRR = (" + " + ".join(f"1/{rank}" for rank in ranks) + f") / {len(ranks)} = {mrr:.3f}")
        return
    if args.mode == "extract":
        extract_triples()
        return

    emb = Embedder(args.emb, force_fallback=args.force_fallback)
    if args.benchmark_build:
        timings = {}
        built = {}
        constructors = {
            "Vector RAG": lambda: VectorRAG(emb, hybrid=args.hybrid),
            "GraphRAG": lambda: GraphRAG(emb),
            "WikiRAG": lambda: WikiRAG(emb),
        }
        for name, constructor in constructors.items():
            started = time.perf_counter()
            built[name] = constructor()
            timings[name] = time.perf_counter() - started
        output = {"embedding_model": args.emb if emb.ok else "character-3gram-fallback", "build_seconds": timings}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.save_json:
            save_json(output, args.save_json)
        return
    systems = build_systems(emb, hybrid=args.hybrid)
    if args.mode == "eval":
        if args.scan_k:
            scan = {str(k): evaluate(systems, k, QUESTIONS[:15]) for k in (1, 2, 3, 5)}
            output = {
                "embedding_model": args.emb if emb.ok else "character-3gram-fallback",
                "hybrid": args.hybrid,
                "builtin_question_count": 15,
                "scan": scan,
            }
            if args.save_json:
                save_json(output, args.save_json)
            return
        questions = QUESTIONS[:15] if args.builtin_only else QUESTIONS
        results = evaluate(systems, args.k, questions)
        results["embedding_model"] = args.emb if emb.ok else "character-3gram-fallback"
        results["hybrid"] = args.hybrid
        if args.save_json:
            save_json(results, args.save_json)
        return

    system_name = {"vector": "Vector RAG" + (" + RRF" if args.hybrid else ""), "graph": "GraphRAG", "wiki": "WikiRAG"}[args.mode]
    system = systems[system_name]
    question = next(item for item in QUESTIONS if item["id"].upper() == args.q.upper())
    docs = system.retrieve(question["q"], args.k, args.scope)
    print(f"\n[{question['id']}｜{question['type']}] {question['q']}")
    print("标准evidence：", question["evidence"])
    print("检索到的文档：", docs)
    if isinstance(system, VectorRAG):
        print("文档级分数：", [(doc, round(score, 6)) for doc, score in system.last_hits])
    elif isinstance(system, GraphRAG):
        print("沿边三元组：", system.last_path[:12])
    elif isinstance(system, WikiRAG):
        print("命中条目：", [(title, round(score, 6)) for title, score in system.last_entries])
    context = "\n".join(f"[{doc_id}] {CORPUS[doc_id]}" for doc_id in docs)
    try:
        answer = llm_chat(ANSWER_PROMPT.replace("{q}", question["q"]).replace("{ctx}", context))
        print("\n回答：", answer)
    except SystemExit:
        print("\n未配置LLM API，仅输出检索上下文：\n" + context)


if __name__ == "__main__":
    main()
