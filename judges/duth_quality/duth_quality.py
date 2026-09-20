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
            "DUTH_QUALITY",
            description="Pointwise report quality score in [0,1]",
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
        return max(0.0, min(1.0, score / 4.0))
    except Exception:
        pass

    m = re.search(r'"?score"?\s*[:=]\s*([0-4](?:\.\d+)?)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 4.0))

    return 0.0


class DuthQualityJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        filebase: str = "duth_quality",
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
            query = topics.get(topic_id, "")
            answer = get_response_text(response)

            query = query[:6000]
            answer = answer[:12000]

            prompt = f"""Evaluate the quality of a generated research/report answer for the user request.

USER REQUEST:
{query}

SYSTEM ANSWER:
{answer}

Judge the answer as a whole.

Use this 0-4 scale:
4 = directly addresses the request, comprehensive, specific, coherent, and appears factually reliable.
3 = mostly useful and correct, with minor omissions, redundancy, or unsupported details.
2 = partially useful but has substantial omissions, vagueness, repetition, or questionable claims.
1 = mostly inadequate, poorly focused, or largely unsupported.
0 = irrelevant, empty, nonsensical, or fundamentally fails the request.

Important:
- Reward coverage of all important parts of the request.
- Penalize repetition and irrelevant material.
- Penalize unsupported or suspiciously specific claims.
- Do not reward verbosity by itself.
- Judge only the supplied request and answer.
- Output JSON only, exactly like: {{"score": 3}}
"""

            request = MinimaLlmRequest(
                request_id=f"duth-quality-{i}",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict, consistent evaluator "
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
            f"[DuthQuality] sending "
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
                    f"[DuthQuality] LLM failure "
                    f"run={run_id} topic={topic_id}: {result}"
                )

            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={"DUTH_QUALITY": score},
            )

            if i % 100 == 0:
                print(f"[DuthQuality] completed {i}")

        return builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing="fix_aggregate",
        )
