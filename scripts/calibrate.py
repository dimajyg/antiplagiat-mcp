"""Run the deployed MCP server against a tiny labelled corpus to inspect the
ai_probability / perplexity / burstiness ranges it produces. Use this to
sanity-check thresholds in `src/detectors/ai_local.py`.

Not a unit test — it talks to the production endpoint over the network.
Intended to be re-run by hand after each model or threshold change.

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --url https://antiplagiat-mcp.tailf03eb8.ts.net/mcp/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics

import httpx

DEFAULT_URL = "https://antiplagiat-mcp.tailf03eb8.ts.net/mcp/"

# (label, language, source, text). "human" / "ai" labels are only ground truth
# in spirit — these are illustrative samples, not a benchmark.
CORPUS: list[tuple[str, str, str, str]] = [
    (
        "human",
        "ru",
        "diary",
        "Сегодня с утра был дождь, я опоздал на работу из-за лужи у подъезда. "
        "В метро какой-то парень слушал музыку без наушников, бесило. Купил "
        "себе кофе на углу, заварили крепко, как я люблю. Дома ждёт кот, "
        "которому уже пора менять корм. В общем, обычный понедельник.",
    ),
    (
        "human",
        "ru",
        "blog",
        "Знаете, я долго не мог понять, почему меня так раздражает эта реклама "
        "в подъезде. Потом сообразил — она крепится прямо на дверь лифта и "
        "хлопает каждый раз, когда кто-то едет. Уже хотел сорвать, но соседка "
        "сверху меня опередила. Уважаю.",
    ),
    (
        "ai",
        "ru",
        "chatgpt",
        "В современном мире, характеризующемся стремительным развитием "
        "информационных технологий, особое значение приобретает способность "
        "человека эффективно адаптироваться к постоянно изменяющимся условиям. "
        "Данный навык является ключевым фактором достижения успеха как в "
        "профессиональной, так и в личной сфере.",
    ),
    (
        "ai",
        "ru",
        "chatgpt",
        "Машинное обучение представляет собой один из наиболее перспективных "
        "разделов искусственного интеллекта. Оно позволяет компьютерным "
        "системам автоматически улучшать свою производительность на основе "
        "опыта без явного программирования. Применение методов машинного "
        "обучения охватывает широкий спектр областей, включая медицину, "
        "финансы и транспорт.",
    ),
    (
        "human",
        "en",
        "blog",
        "I tried to fix the dishwasher again last weekend and ended up making "
        "things worse. The drain hose came off in my hand and I just sat on "
        "the kitchen floor laughing. Called a plumber on Monday. He took five "
        "minutes and charged me eighty bucks. Worth every penny.",
    ),
    (
        "ai",
        "en",
        "chatgpt",
        "In todays rapidly evolving technological landscape, it is essential "
        "to consider the multifaceted dimensions of artificial intelligence "
        "and its profound implications for society. By embracing these "
        "transformative innovations, organizations can unlock unprecedented "
        "opportunities for growth and competitive advantage.",
    ),
]


async def call_tool(client: httpx.AsyncClient, url: str, text: str) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "analyze_text",
            "arguments": {"text": text, "check_plagiarism": False},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Origin": "https://antiplagiat-mcp.tailf03eb8.ts.net",
    }
    resp = await client.post(url, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    body = resp.json()
    inner = json.loads(body["result"]["content"][0]["text"])
    return inner["ai"]


async def main(url: str) -> None:
    rows: list[tuple[str, str, str, dict]] = []
    async with httpx.AsyncClient() as client:
        for label, lang, source, text in CORPUS:
            ai = await call_tool(client, url, text)
            rows.append((label, lang, source, ai))

    header = f"{'label':<6} {'lang':<4} {'src':<8} {'ppl':>8} {'burst':>7} {'aiP':>6} {'conf':<7}"
    print(header)
    print("-" * len(header))
    for label, lang, source, ai in rows:
        print(
            f"{label:<6} {lang:<4} {source:<8} "
            f"{(ai['perplexity'] or 0):>8.1f} "
            f"{ai['burstiness']:>7.1f} "
            f"{ai['ai_probability']:>6.2f} "
            f"{ai['confidence']:<7}"
        )

    print()
    for lang in ("ru", "en"):
        for label in ("human", "ai"):
            ppls = [
                r[3]["perplexity"]
                for r in rows
                if r[1] == lang and r[0] == label and r[3]["perplexity"]
            ]
            if not ppls:
                continue
            avg = statistics.mean(ppls)
            print(f"{lang} {label:<6} mean ppl = {avg:.1f}  (n={len(ppls)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    asyncio.run(main(args.url))
