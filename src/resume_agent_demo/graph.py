"""LangGraph 编排：build_query → retrieve → score → explain。

LLM 未配置或调用失败时逐节点启发式降级，图结构始终可运行、可测试。
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.engine import Engine

from .clients import OpenAIChatClient, json_or_none
from .config import Settings
from .embeddings import get_llm_client
from .retriever import SearchHit, search
from .tokenizer import tokenize


class MatchState(TypedDict, total=False):
    jd: str
    top_k: int
    mode: str  # llm | heuristic
    query: str
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    explanation: str


def _heuristic_query(jd: str) -> str:
    seen: list[str] = []
    for tok in tokenize(jd):
        # 剔除纯符号/纯数字 token（markdown 标记、序号），保留含字母或 CJK 的词
        if not any(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in tok):
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 12:
            break
    return " ".join(seen) or jd[:60]


def _group_candidates(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """按文档聚合检索片段，供文档级评分。"""
    order: list[str] = []
    by_title: dict[str, dict[str, Any]] = {}
    for i, h in enumerate(hits, 1):
        cand = by_title.setdefault(
            h.doc_title, {"title": h.doc_title, "source": h.source, "snippets": [], "cites": []}
        )
        if h.doc_title not in order:
            order.append(h.doc_title)
        cand["snippets"].append(h.content)
        cand["cites"].append(i)
    return [by_title[t] for t in order]


def _coverage_score(jd_tokens: set[str], text: str) -> int:
    if not jd_tokens:
        return 0
    covered = jd_tokens & set(tokenize(text))
    return round(len(covered) / len(jd_tokens) * 100)


def build_graph(
    engine: Engine | None,
    settings: Settings,
    llm: OpenAIChatClient | None = None,
    retriever=None,
):
    """构建匹配图；llm / retriever 可注入以便测试。"""
    llm = llm if llm is not None else get_llm_client(settings)
    retriever = retriever or search

    def build_query(state: MatchState) -> dict[str, Any]:
        jd = state["jd"]
        if llm is not None:
            try:
                q = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "从 JD 中提取 3-8 个技术检索关键词，用空格分隔，只输出关键词本身。"
                            ),
                        },
                        {"role": "user", "content": jd},
                    ]
                )
                q = q.strip().strip('"')
                if q:
                    return {"query": q, "mode": "llm"}
            except Exception:  # noqa: BLE001
                pass
        return {"query": _heuristic_query(jd), "mode": "heuristic"}

    def retrieve(state: MatchState) -> dict[str, Any]:
        hits = retriever(engine, settings, state["query"], top_k=8)
        return {"candidates": _group_candidates(hits), "top_k": state.get("top_k", 5)}

    def score(state: MatchState) -> dict[str, Any]:
        candidates = state["candidates"]
        jd_tokens = set(tokenize(state["jd"]))
        for cand in candidates:
            cand["score"] = _coverage_score(jd_tokens, "\n".join(cand["snippets"]))
        mode = "heuristic"

        if llm is not None and candidates:
            try:
                blocks = "\n\n".join(
                    f"{i}. {c['title']}\n{chr(10).join(c['snippets'])[:800]}"
                    for i, c in enumerate(candidates, 1)
                )
                raw = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是招聘匹配助手。对每个候选人输出 0-100 的匹配分。"
                                '只输出 JSON 数组：[{"n": 1, "score": 85, "reason": "一句话理由"}]'
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"JD：\n{state['jd'][:1500]}\n\n候选人：\n{blocks}",
                        },
                    ]
                )
                parsed = json_or_none(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        try:
                            cand = candidates[int(item["n"]) - 1]
                            cand["score"] = max(0, min(100, int(item["score"])))
                            cand["reason"] = str(item.get("reason", ""))[:200]
                        except (KeyError, ValueError, IndexError):
                            continue
                    mode = "llm"
            except Exception:  # noqa: BLE001
                pass

        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)[: state["top_k"]]
        return {"ranked": ranked, "mode": mode}

    def explain(state: MatchState) -> dict[str, Any]:
        ranked = state["ranked"]
        if not ranked:
            return {"explanation": "简历库中没有找到与该 JD 相关的候选人。请先入库简历。"}

        lines: list[str] = []
        if state["mode"] == "heuristic":
            lines.append("> 未配置 LLM，以下为关键词覆盖率启发式评分（mode=heuristic）\n")
        lines.append(f"共评估 {len(ranked)} 位候选人，按匹配度排序：\n")
        for i, cand in enumerate(ranked, 1):
            cites = " ".join(f"[{c}]" for c in cand["cites"])
            reason = cand.get("reason") or f"关键词覆盖率 {cand['score']}%"
            lines.append(f"**{i}. {cand['title']} — {cand['score']} 分** {cites}")
            lines.append(f"- 理由：{reason}")
        lines.append("\n引用编号对应检索片段，可通过 search_resumes 查看原文。")
        return {"explanation": "\n".join(lines)}

    builder = StateGraph(MatchState)
    builder.add_node("build_query", build_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("score", score)
    builder.add_node("explain", explain)
    builder.add_edge(START, "build_query")
    builder.add_edge("build_query", "retrieve")
    builder.add_edge("retrieve", "score")
    builder.add_edge("score", "explain")
    builder.add_edge("explain", END)
    return builder.compile()


def run_match(
    engine: Engine | None,
    settings: Settings,
    jd: str,
    top_k: int = 5,
    llm: OpenAIChatClient | None = None,
    retriever=None,
) -> dict[str, Any]:
    graph = build_graph(engine, settings, llm=llm, retriever=retriever)
    final = graph.invoke({"jd": jd, "top_k": top_k})
    return {
        "mode": final["mode"],
        "query": final["query"],
        "candidates": final["ranked"],
        "explanation": final["explanation"],
    }
