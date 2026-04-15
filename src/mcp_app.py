"""FastMCP server: `analyze_text` tool + workflow prompts for Claude Code.

The trick to passing per-request credentials into a stateless MCP tool is a
`contextvars.ContextVar` set by a Starlette/FastAPI middleware before each
request reaches the MCP handler. Tools read it via `current_credentials()`.
"""

from __future__ import annotations

import contextvars
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from .config import RequestCredentials, settings
from .pipeline import Pipeline

_creds_var: contextvars.ContextVar[RequestCredentials | None] = contextvars.ContextVar(
    "antiplagiat_creds", default=None
)


def set_credentials(creds: RequestCredentials) -> contextvars.Token:
    return _creds_var.set(creds)


def reset_credentials(token: contextvars.Token) -> None:
    _creds_var.reset(token)


def current_credentials() -> RequestCredentials:
    c = _creds_var.get()
    if c is None:
        return RequestCredentials.from_headers({}, settings)
    return c


pipeline = Pipeline()

PUBLIC_HOSTNAME = "antiplagiat-mcp.tailf03eb8.ts.net"
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        PUBLIC_HOSTNAME,
        "127.0.0.1",
        "127.0.0.1:8765",
        "localhost",
        "localhost:8765",
    ],
    allowed_origins=[f"https://{PUBLIC_HOSTNAME}"],
)

# Server-level instructions. MCP clients (including Claude Code) surface these
# to the model as context about when and how to use this server. Keep it tight
# and task-oriented — this is not the place for API documentation.
_INSTRUCTIONS = """\
Scores arbitrary text for **AI-generation probability** and **plagiarism**.
Russian is the primary target language; English works but is less reliable.

When to call `analyze_text`:
- user asks whether a text was written by an LLM
- user wants to audit a document, essay, thesis, or article for AI-style or copied passages
- user pastes a fragment and asks "is this AI?" or "is this plagiarism?"

How to use it well:
- For documents > ~5000 characters, split by headings or paragraphs and call once per section. One call per document loses precision and may time out on CPU inference.
- Start with `mode="fast"` (free, local, ~1-3s per 1000 chars). `mode="deep"` adds a Sapling call (~$0.005 per 1000 chars) — useful as a second opinion but **not** more trustworthy than the local heuristic on Russian academic prose (Sapling returns ~100% on a 2016 control paper; see README "Known limitations").
- Set `check_plagiarism=False` on private documents the user doesn't want sent to external services. Private data otherwise leaves the server via Serper search and Sapling.
- Always surface the *signals*, not just `ai_probability`: perplexity, burstiness, suspicious_sentences and matched URLs are what lets a human decide. Never present the score as a verdict.
- For long documents, compute the **document-wide median perplexity** yourself and flag sections that are significantly below it. Absolute thresholds do not work for Russian academic prose; relative anomalies within one document do.

Three workflow prompts are exposed as slash commands:
- `/mcp__antiplagiat__check_fragment` — analyse one pasted fragment
- `/mcp__antiplagiat__deep_check` — drill into a specific suspicious passage with Sapling
- `/mcp__antiplagiat__thesis_audit` — multi-section audit of a large document (diploma, article)

API keys flow through request headers per call; the server stores nothing per user.
"""

mcp = FastMCP(
    name="antiplagiat-mcp",
    instructions=_INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    transport_security=_transport_security,
)


# ---------- Tool ----------


@mcp.tool()
async def analyze_text(
    text: Annotated[
        str,
        Field(
            description=(
                "The text to analyse, 50+ characters, ideally one section of ≤5000 "
                "characters. For longer documents, split by headings and call once "
                "per section — do not concatenate."
            ),
            min_length=1,
        ),
    ],
    mode: Annotated[
        str,
        Field(
            description=(
                "'fast' = local perplexity+burstiness, free, ~1-3s per 1000 chars. "
                "'deep' = also calls Sapling AI detector (trained classifier, much "
                "more reliable on Russian), adds ~2s and costs ~$0.005/1000 chars. "
                "Use 'deep' only for passages already flagged as suspicious in 'fast'."
            ),
            pattern="^(fast|deep)$",
        ),
    ] = "fast",
    check_ai: Annotated[
        bool,
        Field(description="Run AI-generation detection. Default true."),
    ] = True,
    check_plagiarism: Annotated[
        bool,
        Field(
            description=(
                "Run web-search plagiarism detection (Serper + source fetching). "
                "Costs ~6 Serper queries and reveals the text to Google. Set "
                "False for private / NDA documents."
            )
        ),
    ] = True,
) -> dict:
    """Score a text fragment for AI-generation and plagiarism (RU primary, EN supported).

    Returns JSON:
      - `ai.ai_probability` (0-1) — local perplexity+burstiness heuristic; NOT overridden by Sapling (see README known limitations — Sapling returns ~100% on Russian academic prose regardless of authorship)
      - `ai.local_heuristic_probability` — same as above when deep mode was used, for symmetry with external_sources
      - `ai.perplexity`, `ai.burstiness`, `ai.confidence` — raw signals
      - `ai.suspicious_sentences[]` — top-5 most predictable sentences with char offsets
      - `ai.external_sources[]` — populated only when mode='deep' and a key was passed
      - `ai.notes[]` — human-readable explanations of what the signals mean
      - `plagiarism.match_percentage` (0-1) — word coverage of matched passages
      - `plagiarism.matches[]` — list of {quote, source_url, source_title, similarity, kind}
      - `plagiarism.skipped_reason` — set when plagiarism couldn't run (no Serper key, text too short)
      - `summary` — one-line human string combining both layers

    DO NOT:
    - Present `ai_probability` as a verdict — always show the raw signals and suspicious_sentences so the user can judge.
    - Treat `plagiarism.matches` as proof of cheating — properly quoted citations also match. Cross-check URLs against the user's reference list.
    - Call on texts shorter than ~50 characters — confidence will be 'low' and results are noise.
    - Call on the same text in both 'fast' and 'deep' without reason — deep costs money.
    """
    creds = current_credentials()
    result = await pipeline.analyze(
        text=text,
        creds=creds,
        mode=mode,  # type: ignore[arg-type]
        check_ai=check_ai,
        check_plagiarism=check_plagiarism,
    )
    return result.to_mcp()


# ---------- Prompts (slash commands in Claude Code) ----------


@mcp.prompt(
    name="check_fragment",
    title="Проверить фрагмент на AI и плагиат",
    description="Одноразовый fast-mode анализ пасты. Обнаружит AI-style и совпадения в вебе.",
)
def check_fragment(
    text: Annotated[str, Field(description="Текст для проверки (50+ символов).")],
    deep: Annotated[
        bool,
        Field(description="Использовать deep mode (Sapling + стоит денег). Default false."),
    ] = False,
) -> str:
    return f"""Прогони этот фрагмент через `analyze_text` инструмент antiplagiat-mcp \
(mode={'deep' if deep else 'fast'}, check_plagiarism=true), затем покажи результаты структурно:

1. **Вердикт** — ai_probability + confidence в одну строку
2. **AI-сигналы** — perplexity с baseline, burstiness, sentence_count
3. **Топ-3 подозрительных предложения** — с их perplexity и позициями
4. **Плагиат** — match_percentage + список matches с URL и similarity (или skipped_reason если не запускалось)
5. **Заметки** — всё из поля notes дословно
6. **Итог** — одно предложение: вероятно AI / вероятно человек / пограничный случай; с обоснованием из сигналов

Не пересказывай сам текст и не вноси правки — только диагностика.

Фрагмент:
\"\"\"
{text}
\"\"\"
"""


@mcp.prompt(
    name="deep_check",
    title="Глубокая проверка подозрительного абзаца",
    description="Deep mode на один абзац — зовёт Sapling classifier, стоит ~$0.005 на 1K символов.",
)
def deep_check(
    text: Annotated[str, Field(description="Подозрительный фрагмент для углублённой проверки.")],
) -> str:
    return f"""Запусти `analyze_text` с mode="deep", check_plagiarism=true, check_ai=true \
для этого фрагмента. Я уже видел что он подозрительный в fast mode, сейчас нужен \
честный вердикт от Sapling.

Покажи:
- **ai_probability** (из Sapling) vs **local_heuristic_probability** (наш локальный) — сравнение
- **external_sources** — что вернул Sapling в явном виде
- **confidence** — должен стать 'high'
- **suspicious_sentences** — совпадают ли они с тем что подсвечивал Sapling?
- **plagiarism.matches** — если нашлись, список URL + процент совпадения

Не пересказывай текст. Не предлагай правки. Только диагностика + короткий вывод "AI / человек / не решаемо по данным".

Фрагмент:
\"\"\"
{text}
\"\"\"
"""


@mcp.prompt(
    name="thesis_audit",
    title="Аудит диплома / длинной работы по разделам",
    description="Читает файл, разбивает на разделы по заголовкам, прогоняет каждый, сводит в таблицу и углубляется в самый подозрительный.",
)
def thesis_audit(
    file_path: Annotated[
        str,
        Field(
            description="Путь до файла диплома (md/txt). Если .docx/.pdf — предложи сначала конвертировать."
        ),
    ] = "thesis.md",
    deep_top_n: Annotated[
        int,
        Field(
            description="Сколько самых подозрительных разделов прогнать повторно в deep mode. Default 1, максимум 3.",
            ge=0,
            le=3,
        ),
    ] = 1,
) -> str:
    return f"""Проведи аудит диплома `{file_path}` на AI-генерацию и плагиат.

## Протокол

1. **Прочитай файл** через Read tool. Если расширение `.docx`, `.pdf`, `.doc` — остановись и попроси пользователя сконвертировать в `.md` или `.txt` (`pandoc thesis.docx -o thesis.md` или `pdftotext thesis.pdf`).

2. **Разбей на разделы** по заголовкам верхнего уровня (`#` или `##`). Если заголовков нет — разбей на куски по ~2000 символов по границам параграфов. Каждый кусок обрабатывай как "раздел".

3. **Fast pass.** Для каждого раздела вызови `analyze_text` с параметрами:
   - mode="fast"
   - check_ai=true
   - check_plagiarism=false  (экономим Serper, пойдём в плагиат только точечно в deep pass)

4. **Сведи в markdown-таблицу:**
   ```
   | # | Раздел | chars | ai_prob | ppl | burst | confidence | топ-подозр. предложение |
   ```
   Отсортируй по ai_prob убывающе.

5. **Выдели жирным** разделы где `ai_probability > 0.55` ИЛИ `burstiness < 2.5` ИЛИ `perplexity` далеко ниже baseline (для ru: <10, для en: <15).

6. **Deep pass.** Возьми топ-{deep_top_n} самых подозрительных по ai_probability. Для каждого вызови `analyze_text` с:
   - mode="deep"
   - check_ai=true
   - check_plagiarism=true

   Для каждого покажи: ai_probability (Sapling) vs local_heuristic, external_sources, suspicious_sentences с цитатами, plagiarism.matches с URL.

7. **Финальный вердикт.** Выдай короткое саммари:
   - Сколько разделов выглядят AI-сгенерированными (high confidence)
   - Сколько пограничных (medium confidence, требует ручной проверки)
   - Какие разделы имеют совпадения с источниками (с URL)
   - Оценка: "работу можно защищать / нужны правки в разделах X, Y / серьёзные проблемы"

## Правила

- НЕ переписывай текст диплома и не предлагай правки, пока пользователь явно не попросит
- НЕ отправляй большие (>5000 симв) блоки в analyze_text одним куском — разбивай
- Если файл не найден — спроси у меня правильный путь, не угадывай
- Если какой-то раздел даёт ошибку — продолжай с остальными, в конце перечисли сбойные
- Все числа показывай округлёнными до 2 знаков
"""


# ---------- ASGI mount helper ----------


def streamable_http_asgi():
    """Return the Streamable HTTP ASGI app for mounting under FastAPI."""
    return mcp.streamable_http_app()
