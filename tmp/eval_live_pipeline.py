from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

KEY = os.environ["STT_API_KEY"]
BASE = os.environ["STT_BASE"].rstrip("/")

DOCS = [
    "The weekly product standup is tomorrow at 10am in conference room 4.",
    "The Acme customer call is scheduled tomorrow at 3:30pm IST.",
    "The payment service had an outage at 3:27pm lasting about four minutes.",
    "Unused items can be returned within 30 days for a full refund.",
    "On Windows, install Python packages with pip: python -m pip install package_name.",
    "Apple typically unveils new iPhone models at its September event.",
    "Argentina won the 2022 FIFA World Cup in Qatar, beating France on penalties.",
    "Diwali, the festival of lights, is expected in late October this year in India.",
    "Please submit your expense report by Friday.",
    "The office cafeteria opens at 8am every weekday.",
    "Holi is a spring festival known for colored powders.",
    "France won the 2018 FIFA World Cup in Russia.",
    "Granny Smith apples are tart and hold up well in pies.",
    "Pythons are non-venomous snakes found in Asia and Africa.",
    "Christmas is celebrated on 25 December in many countries.",
    "Two-factor authentication can be enabled in security settings.",
]
GOLD = list(range(8))

QUERIES = [
    {"name": "standup_time", "gold": 0, "q": "What time is the product standup tomorrow and in which room?"},
    {"name": "acme_call", "gold": 1, "q": "When is the Acme customer call tomorrow?"},
    {"name": "payment_outage", "gold": 2, "q": "When did the payment service outage happen and how long did it last?"},
    {"name": "refunds", "gold": 3, "q": "What is the refund policy for unused items?"},
    {"name": "python_pip", "gold": 4, "q": "How do I install Python packages on Windows?"},
    {"name": "iphone", "gold": 5, "q": "When does Apple release new iPhone models?"},
    {"name": "world_cup", "gold": 6, "q": "Who won the 2022 FIFA World Cup?"},
    {"name": "diwali", "gold": 7, "q": "When is Diwali in India this year?"},
]

FACTS = [
    ("standup 10am room 4", [r"stand[\s-]?up", r"\b(10|ten)\b", r"room (4|four)"]),
    ("acme 3:30pm", [r"acme", r"(3[:\s]?30|three thirty)"]),
    ("outage 3:27 4 min", [r"(3[:\s]?27|three twenty seven)", r"four minutes|4 minutes"]),
    ("30 day refund", [r"(thirty|30) days", r"refund"]),
    ("pip windows", [r"\bpip\b", r"python"]),
    ("iphone september", [r"iphone", r"september"]),
    ("argentina 2022", [r"argentina", r"world cup"]),
    ("diwali october", [r"diwali", r"october"]),
]


def call(method: str, path: str, data=None, headers=None, timeout=300):
    hdrs = {"Authorization": f"Bearer {KEY}"}
    if headers:
        hdrs.update(headers)
    started = time.perf_counter()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body, int((time.perf_counter() - started) * 1000)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw[:500]}
        return e.code, body, int((time.perf_counter() - started) * 1000)


def transcribe(mp3_path: str):
    boundary = "----EvalBoundary"
    with open(mp3_path, "rb") as f:
        audio = f.read()
    name = os.path.basename(mp3_path)
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: audio/mpeg\r\n\r\n".encode()
        + audio
        + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n--{boundary}--\r\n".encode()
    )
    return call(
        "POST",
        "/v1/audio/transcriptions",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=300,
    )


def embed(texts: list[str]):
    return call(
        "POST",
        "/v1/embeddings",
        data=json.dumps({"input": texts}).encode(),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )


def rerank(query: str, documents: list[str]):
    return call(
        "POST",
        "/v1/rerank",
        data=json.dumps({"query": query, "documents": documents}).encode(),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )


def cosine_rank(query_vec, doc_vecs):
    ranked = []
    for i, vec in enumerate(doc_vecs):
        score = sum(a * b for a, b in zip(query_vec, vec))
        ranked.append((i, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def fact_hits(text: str) -> tuple[int, list[str]]:
    low = text.lower()
    hit = 0
    missing = []
    for name, pats in FACTS:
        if all(re.search(p, low) for p in pats):
            hit += 1
        else:
            missing.append(name)
    return hit, missing


def summarize_ranks(ranked_idx: list[int], gold: int):
    rank = ranked_idx.index(gold) + 1
    return rank, ranked_idx[0] == gold, gold in ranked_idx[:3]


def eval_clip(label: str, mp3: str) -> dict:
    print(f"\n========== {label} ==========")
    status, stt, stt_client = transcribe(mp3)
    if status != 200:
        print("STT FAIL", status, stt)
        return {"label": label, "error": stt}
    text = stt["text"]
    facts_ok, missing = fact_hits(text)
    print(
        f"STT http={status} audio={stt['duration']:.1f}s "
        f"gpu={stt['processing_ms']}ms client={stt_client}ms rtf={stt['rtf']} "
        f"device={stt['device']}"
    )
    print(f"STT fact-spotting {facts_ok}/8 missing={missing or '-'}")
    print("STT text preview:", text[:280].replace("\n", " "), "...")

    # full transcript vs corpus
    status, emb, emb_client = embed([text] + DOCS)
    if status != 200:
        print("EMBED FAIL", status, emb)
        return {"label": label, "error": emb}
    qv, dvs = emb["embeddings"][0], emb["embeddings"][1:]
    full_embed_rank = [i for i, _ in cosine_rank(qv, dvs)]
    golds_top8_embed = sum(1 for g in GOLD if g in full_embed_rank[:8])

    status, rr, rr_client = rerank(text, DOCS)
    if status != 200:
        print("RERANK FAIL", status, rr)
        return {"label": label, "error": rr}
    full_rr_rank = [r["index"] for r in rr["results"]]
    golds_top8_rr = sum(1 for g in GOLD if g in full_rr_rank[:8])
    print(
        f"FULL-TRANSCRIPT embed gpu={emb['processing_ms']}ms client={emb_client}ms "
        f"golds@8={golds_top8_embed}/8 order={full_embed_rank[:8]}"
    )
    print(
        f"FULL-TRANSCRIPT rerank gpu={rr['processing_ms']}ms client={rr_client}ms "
        f"golds@8={golds_top8_rr}/8 order={full_rr_rank[:8]}"
    )

    embed_top1 = embed_top3 = 0
    rr_top1 = rr_top3 = 0
    embed_ranks = []
    rr_ranks = []
    embed_lat = []
    rr_lat = []
    rows = []
    for case in QUERIES:
        status, ebody, eclient = embed([case["q"]] + DOCS)
        ev = ebody["embeddings"][0]
        ed = ebody["embeddings"][1:]
        erank = [i for i, _ in cosine_rank(ev, ed)]
        epos, e1, e3 = summarize_ranks(erank, case["gold"])
        embed_top1 += int(e1)
        embed_top3 += int(e3)
        embed_ranks.append(epos)
        embed_lat.append(ebody["processing_ms"])

        status, rbody, rclient = rerank(case["q"], DOCS)
        rrank = [r["index"] for r in rbody["results"]]
        rpos, r1, r3 = summarize_ranks(rrank, case["gold"])
        rr_top1 += int(r1)
        rr_top3 += int(r3)
        rr_ranks.append(rpos)
        rr_lat.append(rbody["processing_ms"])
        agree = erank[0] == rrank[0]
        rows.append(
            f"  {case['name']:16} gold={case['gold']}  "
            f"embed@{epos} top1={e1}  rerank@{rpos} top1={r1}  agree={agree}  "
            f"e_gpu={ebody['processing_ms']}ms r_gpu={rbody['processing_ms']}ms"
        )

    n = len(QUERIES)
    def mrr(ranks):
        return sum(1.0 / r for r in ranks) / len(ranks)

    print("PER-QUESTION retrieval against 16-doc KB (8 gold + 8 distractors):")
    print("\n".join(rows))
    print(
        f"EMBED  top1={embed_top1}/{n}={embed_top1/n:.0%}  "
        f"top3={embed_top3}/{n}={embed_top3/n:.0%}  MRR={mrr(embed_ranks):.3f}  "
        f"gpu_ms p50={sorted(embed_lat)[len(embed_lat)//2]} avg={sum(embed_lat)/n:.0f}"
    )
    print(
        f"RERANK top1={rr_top1}/{n}={rr_top1/n:.0%}  "
        f"top3={rr_top3}/{n}={rr_top3/n:.0%}  MRR={mrr(rr_ranks):.3f}  "
        f"gpu_ms p50={sorted(rr_lat)[len(rr_lat)//2]} avg={sum(rr_lat)/n:.0f}"
    )

    return {
        "label": label,
        "audio_s": stt["duration"],
        "stt_gpu_ms": stt["processing_ms"],
        "stt_client_ms": stt_client,
        "stt_rtf": stt["rtf"],
        "stt_facts": f"{facts_ok}/8",
        "full_embed_golds@8": golds_top8_embed,
        "full_rerank_golds@8": golds_top8_rr,
        "full_embed_gpu_ms": emb["processing_ms"],
        "full_embed_client_ms": emb_client,
        "full_rerank_gpu_ms": rr["processing_ms"],
        "full_rerank_client_ms": rr_client,
        "q_embed_top1": f"{embed_top1}/{n}",
        "q_rerank_top1": f"{rr_top1}/{n}",
        "q_embed_mrr": round(mrr(embed_ranks), 3),
        "q_rerank_mrr": round(mrr(rr_ranks), 3),
        "q_embed_gpu_avg_ms": round(sum(embed_lat) / n),
        "q_rerank_gpu_avg_ms": round(sum(rr_lat) / n),
        "device": stt["device"],
    }


def main():
    print("health", call("GET", "/health")[1])
    results = [
        eval_clip("3min", r"C:\Coding\stt-model\tmp\eval_3min.mp3"),
        eval_clip("5min", r"C:\Coding\stt-model\tmp\eval_5min.mp3"),
    ]
    print("\n========== SUMMARY ==========")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
