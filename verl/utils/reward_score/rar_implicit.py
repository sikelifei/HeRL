# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LLM-as-a-judge scoring with explicit per-rubric aggregation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Iterable

import aiohttp
import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert evaluator. For each rubric item, decide whether the response satisfies it. "
    "Return a JSON object with a single key \"results\" whose value is a list of 0/1 integers. "
    "The list length must equal the number of rubrics and each entry corresponds to the rubric with the same index."
)

# USER_PROMPT_TEMPLATE = (
#     "Evaluate the response against each rubric independently.\n\n"
#     "<prompt>\n{prompt}\n</prompt>\n\n"
#     "<response>\n{response}\n</response>\n\n"
#     "<rubrics>\n{rubrics}\n</rubrics>\n\n"
#     "Return only JSON: {\"results\": [0 or 1, ...]}"
# )
USER_PROMPT_TEMPLATE = (
    "Evaluate the response against each rubric independently.\n\n"
    "<prompt>\n{prompt}\n</prompt>\n\n"
    "<response>\n{response}\n</response>\n\n"
    "<rubrics>\n{rubrics}\n</rubrics>\n\n"
    "Return only JSON: {{\"results\": [0 or 1, ...]}}"
)


_MODEL_NAME_CACHE: str | None = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value]


def _try_parse_list(text: str) -> list[Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_rubrics(extra_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not extra_info or not isinstance(extra_info, dict):
        return []

    rubrics = extra_info.get("rubric")
    if rubrics is None:
        rubrics = extra_info.get("rubric_list")

    if isinstance(rubrics, str):
        parsed = _try_parse_list(rubrics)
        if parsed is not None:
            rubrics = parsed
        else:
            rubrics = [line.strip() for line in rubrics.splitlines() if line.strip()]

    rubric_items = _as_list(rubrics)
    results: list[dict[str, Any]] = []
    for item in rubric_items:
        if isinstance(item, dict):
            desc = item.get("description") or item.get("title") or json.dumps(item, ensure_ascii=False)
            weight = _maybe_float(item.get("weight"))
            results.append(
                {
                    "description": str(desc),
                    "title": str(item.get("title") or ""),
                    "weight": weight,
                }
            )
        elif isinstance(item, str):
            results.append({"description": item, "title": "", "weight": None})
        else:
            results.append({"description": str(item), "title": "", "weight": None})
    return results


def _extract_prompt(extra_info: dict[str, Any] | None) -> str:
    if not extra_info or not isinstance(extra_info, dict):
        return ""
    if "question" in extra_info:
        return str(extra_info.get("question") or "")
    prompt = extra_info.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        user_contents = [msg.get("content", "") for msg in prompt if msg.get("role") == "user"]
        if user_contents:
            return "\n".join(user_contents)
        return "\n".join(str(msg.get("content", "")) for msg in prompt)
    return ""


def _format_rubrics(rubrics: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, rubric in enumerate(rubrics, start=1):
        desc = str(rubric.get("description") or "").strip()
        if not desc:
            continue
        weight = _maybe_float(rubric.get("weight"))
        weight_text = f" (weight: {weight:g})" if weight is not None else ""
        lines.append(f"{idx}. {desc}{weight_text}")
    return "\n".join(lines)


def _coerce_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 1 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "pass", "met", "satisfied"}:
            return 1
        if lowered in {"0", "false", "no", "n", "fail", "not", "unsatisfied"}:
            return 0
    return None


def _parse_rubric_results(text: str, expected: int) -> list[int] | None:
    if expected <= 0:
        return []
    if not text:
        return None

    json_block = None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        json_block = fenced.group(1)
    else:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            json_block = text[start_obj : end_obj + 1]
        elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            json_block = text[start_arr : end_arr + 1]

    if json_block is None:
        return None

    try:
        parsed = json.loads(json_block)
    except Exception:
        return None

    raw_results: Any = None
    if isinstance(parsed, dict):
        raw_results = parsed.get("results")
        if raw_results is None:
            raw_results = parsed.get("rubrics")
        if raw_results is None:
            raw_results = parsed.get("checks")
    else:
        raw_results = parsed

    if not isinstance(raw_results, list):
        return None

    results: list[int | None] = [None] * expected
    assigned = 0

    if raw_results and all(isinstance(item, dict) for item in raw_results):
        for idx, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            pos = None
            for key in ("index", "id", "rubric_index"):
                if key in item:
                    try:
                        pos = int(item[key]) - 1
                    except Exception:
                        pos = None
                    break
            if pos is None or pos < 0 or pos >= expected:
                pos = idx if idx < expected else None
            if pos is None:
                continue
            value = item.get("satisfied")
            if value is None:
                value = item.get("met")
            if value is None:
                value = item.get("pass")
            coerced = _coerce_binary(value)
            if coerced is None:
                coerced = 0
            if results[pos] is None:
                results[pos] = coerced
                assigned += 1
    else:
        for idx, item in enumerate(raw_results[:expected]):
            coerced = _coerce_binary(item)
            if coerced is None:
                coerced = 0
            results[idx] = coerced
            assigned += 1

    if assigned == 0:
        return None

    return [int(val or 0) for val in results]


def _normalize_weight(value: Any) -> float:
    weight = _maybe_float(value)
    if weight is None:
        return 1.0
    if weight < 0:
        return 0.0
    return float(weight)


def _build_score_dict(
    score: float,
    is_following_list: list[int] | None = None,
    constraint_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "dense": float(score),
        "sparse": float(score),
        "is_following_list": is_following_list,
        "constraint_results": constraint_results,
    }


def _truncate_text(text: str, limit: int) -> str:
    if not text:
        return ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _resolve_model_name(base_url: str, api_key: str, timeout_sec: float) -> str | None:
    global _MODEL_NAME_CACHE
    if _MODEL_NAME_CACHE:
        return _MODEL_NAME_CACHE
    if not base_url:
        return None

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.get(url, headers=headers, timeout=timeout_sec)
        response.raise_for_status()
        data = response.json()
        models = data.get("data") or []
        if models:
            _MODEL_NAME_CACHE = models[0].get("id")
    except Exception as exc:
        logger.warning("Failed to fetch model list from %s: %s", url, exc)
    return _MODEL_NAME_CACHE


def _make_payload(model: str, prompt: str, response: str, rubrics: str, temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(prompt=prompt, response=response, rubrics=rubrics)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


async def _request_score(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> dict[str, Any] | None:
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                async with session.post(url, json=payload, headers=headers, timeout=timeout_sec) as resp:
                    text = await resp.text()
                    if resp.status >= 500 or resp.status == 429:
                        raise RuntimeError(f"HTTP {resp.status}: {text}")
                    return json.loads(text)
        except Exception as exc:
            if attempt >= max_retries:
                logger.warning("LLM judge request failed: %s", exc)
                return None
            await asyncio.sleep(retry_backoff_sec * (2**attempt) + random.random() * 0.1)
    return None


async def _score_batch_async(
    prompts: list[str],
    responses: list[str],
    extra_infos: list[dict[str, Any] | None],
    base_url: str,
    api_key: str,
    model: str,
    endpoint: str,
    max_concurrency: int,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
    temperature: float,
    max_tokens: int,
    default_score_on_error: float,
    score_scale: float,
    score_offset: float,
    clamp_min: float | None,
    clamp_max: float | None,
    extra_body: dict[str, Any] | None,
    debug_print: bool,
    debug_print_n: int,
    debug_max_chars: int,
) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + endpoint
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    semaphore = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession() as session:
        payloads = []
        rubric_items_list: list[list[dict[str, Any]]] = []
        for prompt, response, extra_info in zip(prompts, responses, extra_infos, strict=True):
            rubric_items = _extract_rubrics(extra_info)
            rubrics = _format_rubrics(rubric_items)
            payload = _make_payload(model, prompt, response, rubrics, temperature, max_tokens)
            if extra_body:
                payload.update(extra_body)
            payloads.append(payload)
            rubric_items_list.append(rubric_items)

        tasks = [
            _request_score(
                session=session,
                semaphore=semaphore,
                url=url,
                headers=headers,
                payload=payload,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
            for payload in payloads
        ]
        results = await asyncio.gather(*tasks)

    scores: list[dict[str, Any]] = []
    debug_printed = 0
    for sample_idx, (result, rubric_items, prompt, response) in enumerate(
        zip(results, rubric_items_list, prompts, responses, strict=True)
    ):
        debug_this = debug_print and debug_printed < debug_print_n
        if debug_this:
            debug_printed += 1

        if not rubric_items:
            if debug_this:
                print(f"[REWARD][DEBUG] idx={sample_idx} no rubrics; default_score={default_score_on_error}")
            scores.append(_build_score_dict(default_score_on_error))
            continue
        if not result:
            if debug_this:
                print(f"[REWARD][DEBUG] idx={sample_idx} no result; default_score={default_score_on_error}")
            scores.append(_build_score_dict(default_score_on_error))
            continue
        content = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(result, dict) else ""
        )
        rubric_results = _parse_rubric_results(str(content), len(rubric_items))
        if rubric_results is None:
            if debug_this:
                print(f"[REWARD][DEBUG] idx={sample_idx} parse failed; default_score={default_score_on_error}")
                print("[REWARD][DEBUG] raw_judge:", _truncate_text(str(content), debug_max_chars))
            scores.append(_build_score_dict(default_score_on_error))
            continue

        weights = [_normalize_weight(item.get("weight")) for item in rubric_items]
        total_weight = sum(weights)
        if total_weight <= 0:
            aggregated = 0.0
        else:
            aggregated = sum(w * float(r) for w, r in zip(weights, rubric_results, strict=True)) / total_weight

        score = aggregated * score_scale + score_offset
        if clamp_min is not None:
            score = max(clamp_min, score)
        if clamp_max is not None:
            score = min(clamp_max, score)
        constraint_results: list[dict[str, Any]] = []
        for ridx, (rubric, flag, weight) in enumerate(
            zip(rubric_items, rubric_results, weights, strict=True), start=1
        ):
            desc = rubric.get("description") or rubric.get("title") or ""
            constraint_results.append(
                {
                    "id": ridx,
                    "description": str(desc),
                    "title": str(rubric.get("title") or ""),
                    "weight": weight,
                    "is_following": bool(flag),
                }
            )
        if debug_this:
            satisfied = sum(1 for flag in rubric_results if flag)
            unmet = len(rubric_results) - satisfied
            print(
                "[REWARD][DEBUG] idx="
                f"{sample_idx} rubrics={len(rubric_items)} satisfied={satisfied} unmet={unmet} "
                f"norm={aggregated:.4f} score={score:.4f} weight_sum={total_weight:.4f}"
            )
            print("[REWARD][DEBUG] prompt:", _truncate_text(prompt, debug_max_chars))
            print("[REWARD][DEBUG] response:", _truncate_text(response, debug_max_chars))
            print("[REWARD][DEBUG] raw_judge:", _truncate_text(str(content), debug_max_chars))
            for ridx, (rubric, weight, flag) in enumerate(
                zip(rubric_items, weights, rubric_results, strict=True), start=1
            ):
                desc = _truncate_text(str(rubric.get("description") or rubric.get("title") or ""), 200)
                print(f"[REWARD][DEBUG] rubric[{ridx}] w={weight:g} flag={flag} desc={desc}")
        scores.append(
            _build_score_dict(float(score), is_following_list=rubric_results, constraint_results=constraint_results)
        )
    return scores


def rarcompute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[dict[str, Any] | None] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]] | dict[str, Any]:
    base_url = kwargs.get("base_url") or os.getenv("LLM_AS_A_JUDGE_BASE", "")
    api_key = kwargs.get("api_key") or os.getenv("LLM_AS_A_JUDGE_API_KEY", "EMPTY")
    model = kwargs.get("model") or os.getenv("LLM_AS_A_JUDGE_MODEL", "")
    endpoint = kwargs.get("endpoint") or os.getenv("LLM_AS_A_JUDGE_ENDPOINT", "/v1/chat/completions")

    max_concurrency = max(1, int(kwargs.get("max_concurrency", 8)))
    timeout_sec = float(kwargs.get("timeout_sec", 60.0))
    max_retries = int(kwargs.get("max_retries", 2))
    retry_backoff_sec = float(kwargs.get("retry_backoff_sec", 0.5))
    temperature = float(kwargs.get("temperature", 0.1))
    max_tokens = int(kwargs.get("max_tokens", 64))
    default_score_on_error = float(kwargs.get("default_score_on_error", 0.0))
    score_scale = float(kwargs.get("score_scale", 1.0))
    score_offset = float(kwargs.get("score_offset", 0.0))
    clamp_min = _maybe_float(kwargs.get("clamp_min", None))
    clamp_max = _maybe_float(kwargs.get("clamp_max", None))
    extra_body = kwargs.get("extra_body", None)
    debug_print = bool(kwargs.get("debug_print", False))
    debug_print_n = int(kwargs.get("debug_print_n", 3))
    debug_max_chars = int(kwargs.get("debug_max_chars", 800))
    allowed_sources = kwargs.get("allowed_data_sources", None)
    if isinstance(allowed_sources, str):
        parsed = _try_parse_list(allowed_sources)
        if parsed is not None:
            allowed_sources = parsed
        else:
            allowed_sources = [s.strip() for s in allowed_sources.split(",") if s.strip()]

    if not base_url:
        logger.warning("LLM judge base_url is empty.")
        fallback = _build_score_dict(default_score_on_error)
        return fallback if data_sources is None else [fallback] * len(data_sources)

    if not model:
        model = _resolve_model_name(base_url, api_key, timeout_sec) or ""

    if not model:
        logger.warning("LLM judge model is empty.")
        fallback = _build_score_dict(default_score_on_error)
        return fallback if data_sources is None else [fallback] * len(data_sources)

    if data_sources is None:
        data_sources = [data_source or "unknown"]
        solution_strs = [solution_str or ""]
        extra_infos = [extra_info]

    if solution_strs is None:
        solution_strs = ["" for _ in data_sources]
    if extra_infos is None:
        extra_infos = [None for _ in data_sources]

    if allowed_sources:
        allowed = set(allowed_sources)
        scores = []
        batch_prompts = []
        batch_responses = []
        batch_extras = []
        index_map = []
        for idx, source in enumerate(data_sources):
            if source in allowed:
                batch_prompts.append(_extract_prompt(extra_infos[idx]))
                batch_responses.append(solution_strs[idx])
                batch_extras.append(extra_infos[idx])
                index_map.append(idx)
                scores.append(None)
            else:
                scores.append(_build_score_dict(default_score_on_error))
        batch_scores = _run_async_scores(
            batch_prompts,
            batch_responses,
            batch_extras,
            base_url,
            api_key,
            model,
            endpoint,
            max_concurrency,
            timeout_sec,
            max_retries,
            retry_backoff_sec,
            temperature,
            max_tokens,
            default_score_on_error,
            score_scale,
            score_offset,
            clamp_min,
            clamp_max,
            extra_body,
            debug_print,
            debug_print_n,
            debug_max_chars,
        )
        for idx, score in zip(index_map, batch_scores, strict=True):
            scores[idx] = score
        return scores[0] if len(scores) == 1 else scores

    prompts = [_extract_prompt(info) for info in extra_infos]
    scores = _run_async_scores(
        prompts,
        solution_strs,
        extra_infos,
        base_url,
        api_key,
        model,
        endpoint,
        max_concurrency,
        timeout_sec,
        max_retries,
        retry_backoff_sec,
        temperature,
        max_tokens,
        default_score_on_error,
        score_scale,
        score_offset,
        clamp_min,
        clamp_max,
        extra_body,
        debug_print,
        debug_print_n,
        debug_max_chars,
    )
    return scores[0] if len(scores) == 1 else scores


def _run_async_scores(
    prompts: list[str],
    responses: list[str],
    extra_infos: list[dict[str, Any] | None],
    base_url: str,
    api_key: str,
    model: str,
    endpoint: str,
    max_concurrency: int,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
    temperature: float,
    max_tokens: int,
    default_score_on_error: float,
    score_scale: float,
    score_offset: float,
    clamp_min: float | None,
    clamp_max: float | None,
    extra_body: dict[str, Any] | None,
    debug_print: bool,
    debug_print_n: int,
    debug_max_chars: int,
) -> list[dict[str, Any]]:
    async def runner():
        return await _score_batch_async(
            prompts=prompts,
            responses=responses,
            extra_infos=extra_infos,
            base_url=base_url,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            max_concurrency=max_concurrency,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            temperature=temperature,
            max_tokens=max_tokens,
            default_score_on_error=default_score_on_error,
            score_scale=score_scale,
            score_offset=score_offset,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            extra_body=extra_body,
            debug_print=debug_print,
            debug_print_n=debug_print_n,
            debug_max_chars=debug_max_chars,
        )

    try:
        return asyncio.run(runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(runner())
        finally:
            loop.close()
