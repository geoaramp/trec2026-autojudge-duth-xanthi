#!/usr/bin/env python3

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)

from minima_llm import (
    MinimaLlmConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    OpenAIMinimaLlm,
)


SPEC = LeaderboardSpec(
    measures=(
        MeasureSpec(
            "DUTH_FINE100",
            description="Fine-grained pointwise report quality score in [0,1]",
        ),
    )
)


def get_topic_text(t: Request) -> str:
    parts = []

    for field in ("title", "background", "problem_statement"):
        value = getattr(t, field, None)
        if value:
            parts.append(str(value).strip())

    return "\n".join(parts)


def get_response_text(response: Report) -> str:
    texts = []

    for r in response.responses:
        if getattr(r, "text", None):
            texts.append(r.text.strip())

    return "\n".join(texts)


def parse_score(text: str) -> float:
    text = text.strip()

    try:
        obj = json.loads(text)
        score = float(obj["score"])
        return max(0.0, min(1.0, score / 100.0))
    except Exception:
        pass

    # JSON-like fallback
    m = re.search(
        r'"?score"?\s*[:=]\s*(100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)',
        text,
        flags=re.I,
    )
    if m:
        score = float(m.group(1))
        return max(0.0, min(1.0, score / 100.0))

    return 0.0


class DuthFine100Judge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        filebase: str = "duth_fine100",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:

        topics = {
            t.request_id: get_topic_text(t)
            for t in rag_topics
        }

        expected_topic_ids = list(topics.keys())

        requests_info: List[
            Tuple[str, str, MinimaLlmRequest]
        ] = []

        for i, response in enumerate(rag_responses):
            topic_id = response.metadata.topic_id
            query = topics.get(topic_id, "")[:6000]
            answer = get_response_text(response)[:12000]

            prompt = f"""Evaluate the overall quality of this generated research/report answer.

USER REQUEST:
{query}

SYSTEM ANSWER:
{answer}

Assign a fine-grained score from 0 to 100 based on:
- relevance to the request,
- completeness and coverage,
- factual plausibility,
- specificity and usefulness,
- coherence and clarity.

Penalize irrelevant material, repetition, unsupported details, and verbosity without substance.

Anchors:
95-100 exceptional
85-94 very strong
70-84 good but imperfect
50-69 mixed
30-49 weak
0-29 very poor

Use the full scale and distinguish close cases.
Output JSON only:
{{"score": 81}}
"""

            request = MinimaLlmRequest(
                request_id=f"duth-fine100-{i}",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict and consistent evaluator "
                            "of RAG-generated reports."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
            )

            requests_info.append(
                (
                    response.metadata.run_id,
                    topic_id,
                    request,
                )
            )

        full_config = (
            MinimaLlmConfig.from_dict(llm_config.raw)
            if llm_config.raw
            else MinimaLlmConfig.from_env()
        )

        backend = OpenAIMinimaLlm(full_config)

        print(
            f"[DuthFine100] sending "
            f"{len(requests_info)} judgments to LLM backend"
        )

        results = asyncio.run(
            backend.run_batched(
                [req for _, _, req in requests_info]
            )
        )

        builder = LeaderboardBuilder(SPEC)

        for i, ((run_id, topic_id, _), result) in enumerate(
            zip(requests_info, results),
            start=1,
        ):
            score = 0.0

            if isinstance(result, MinimaLlmResponse):
                score = parse_score(result.text)
            else:
                print(
                    f"[DuthFine100] LLM failure "
                    f"run={run_id} topic={topic_id}: {result}"
                )

            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={"DUTH_FINE100": score},
            )

            if i % 100 == 0:
                print(f"[DuthFine100] completed {i}")

        return builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing="fix_aggregate",
        )
