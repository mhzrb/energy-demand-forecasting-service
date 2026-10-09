"""Deterministic stakeholder language, with an explicitly opt-in remote provider."""

import os

import httpx


def template(forecast, monitor):
    drift = monitor["drift"]
    if drift["status"] == "insufficient_data":
        status = "There are not yet enough recent forecasts to assess input drift."
    elif drift["drifted"]:
        status = (
            "Recent inputs differ from training data; review sensor readings and forecast error."
        )
    else:
        status = "No input drift was detected in the recent monitoring window."
    return (
        f"Expected demand is {forecast.prediction_kw:.0f} kW at "
        f"{forecast.forecast_at.isoformat()}, with an estimated range of "
        f"{forecast.lower_kw:.0f}–{forecast.upper_kw:.0f} kW "
        f"(nominal {forecast.nominal_coverage:.0%} interval, not a guarantee). "
        f"{status} This demonstration uses synthetic data."
    )


def summarize(forecast, monitor):
    text = template(forecast, monitor)
    provider = os.getenv("INSIGHT_PROVIDER", "template")
    if provider == "template":
        return {"summary": text, "provider": "template"}
    if provider != "openai":
        raise ValueError("INSIGHT_PROVIDER must be template or openai")
    key = os.getenv("LLM_API_KEY")
    if not key:
        return {"summary": text, "provider": "template", "fallback_reason": "missing_llm_key"}
    # Only the aggregate template leaves the process; raw history is never sent.
    try:
        response = httpx.post(
            os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
            json={
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "system",
                        "content": "Rewrite concisely for an energy manager. "
                        "Preserve numbers, uncertainty, drift status and synthetic-data caveat. "
                        "Do not recommend operational actions or invent causes.",
                    },
                    {"role": "user", "content": text},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty provider response")
        return {"summary": content, "provider": "openai", "unverified_generated_text": True}
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return {"summary": text, "provider": "template", "fallback_reason": "llm_unavailable"}
