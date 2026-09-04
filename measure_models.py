"""
Compare two OpenAI models for the summarizer app: mini vs nano.

Run from the terminal, NOT inside Streamlit:
    python measure_models.py

Reads the API key from .streamlit/secrets.toml -- the same file the app uses.
Falls back to the OPENAI_API_KEY environment variable if that file is missing.

Differences from the in-class script:
  - Two models instead of four, same generation, so tier is the only variable.
  - Uses the same prompt shape the app builds (document + language + summary type).
  - Sweeps the language / summary-type combinations the sidebar actually offers,
    because a model can hold up in English and fall apart in Mandarin.
  - Measures time-to-first-token as well as total time. The app streams, so TTFT
    is what a user actually experiences as "fast."
"""

import csv
import os
import statistics
import sys
import time
import tomllib
from pathlib import Path

import pymupdf
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# CONFIG -- edit before running
# ---------------------------------------------------------------------------

PDF_PATH = "Orlikowski & Gash.pdf"
RUNS_PER_CASE = 3

# USD per 1,000,000 tokens.
# PLACEHOLDERS -- replace from https://platform.openai.com/pricing before
# reporting anything. Sources disagreed on these; do not trust them as written.
# Prices retrieved on: 9/3/26
PRICES = {
    "gpt-5.4-nano": {"in": 0.20, "out": 1.25},   # the cheap default candidate
    "gpt-5.6-terra": {"in": 2.00, "out": 12.00},   # the "advanced model" candidate
}

# The sidebar combinations worth testing. Not the full 4x3 grid -- that is 12
# cases per model and mostly redundant.
TEST_CASES = [
    {"language": "English",  "summary_type": "100 words"},
    {"language": "English",  "summary_type": "5 bullet points"},
    {"language": "English", "summary_type": "2 connecting paragraphs"},
]

# ---------------------------------------------------------------------------


def get_api_key():
    """
    Load the key the same way the app does, from .streamlit/secrets.toml.

    Checks next to this script first, then the current working directory, so it
    works whether you run it from the project folder or elsewhere. Falls back to
    the OPENAI_API_KEY environment variable if no secrets file turns up.
    """
    candidates = [
        Path(__file__).parent / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                key = tomllib.load(f).get("OPENAI_API_KEY")
            if key:
                print(f"Using key from {path}")
                return key

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        print("Using key from OPENAI_API_KEY environment variable")
        return key

    return None


def extract_pdf_text(path):
    """Pull all text out of the PDF once so every model gets an identical string."""
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def build_prompt(document, language, summary_type):
    """Mirror the prompt the Streamlit app builds, so results transfer to the app."""
    return (
        f"Here's a document: {document}. "
        f"Provide a summary in language {language} "
        f"and make sure it is {summary_type}."
    )


def run_once(client, model_id, prompt):
    """
    One streamed call. Returns (ttft, total, usage, text).

    stream_options include_usage makes the API send a final chunk carrying token
    counts. Without it, streamed responses report no usage and cost can't be computed.
    That final chunk has an empty choices list, hence the guard below.
    """
    start = time.perf_counter()
    ttft = None
    pieces = []
    usage = None

    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            pieces.append(delta)

    total = time.perf_counter() - start
    return ttft, total, usage, "".join(pieces)


def main():
    api_key = get_api_key()
    if not api_key:
        sys.exit(
            "No API key found.\n"
            "Expected OPENAI_API_KEY in .streamlit/secrets.toml "
            "or in the environment."
        )

    client = OpenAI(api_key=api_key)
    document = extract_pdf_text(PDF_PATH)

    with open("syllabus.txt", "w") as f:
        f.write(document)
    print(f"Extracted {len(document)} characters to syllabus.txt\n")

    rows = []
    answers = []

    for model_id, price in PRICES.items():
        print(f"{'=' * 60}\n{model_id}\n{'=' * 60}")

        for case in TEST_CASES:
            label = f"{case['language']} / {case['summary_type']}"
            print(f"\n  {label}")

            prompt = build_prompt(document, case["language"], case["summary_type"])
            totals = []

            for run in range(1, RUNS_PER_CASE + 1):
                try:
                    ttft, total, usage, text = run_once(client, model_id, prompt)
                except OpenAIError as e:
                    # Report the real error rather than guessing at the cause.
                    print(f"    run {run}: FAILED -- {type(e).__name__}: {e}")
                    break

                if usage is None:
                    print(f"    run {run}: no usage returned, cost unavailable")
                    continue

                cost = (
                    (usage.prompt_tokens / 1_000_000) * price["in"]
                    + (usage.completion_tokens / 1_000_000) * price["out"]
                )
                totals.append(total)

                ttft_str = f"{ttft:5.2f}" if ttft is not None else "  n/a"
                print(
                    f"    run {run}: ttft={ttft_str}s  total={total:6.2f}s  "
                    f"in={usage.prompt_tokens:>6}  out={usage.completion_tokens:>5}  "
                    f"${cost:.6f}"
                )

                rows.append({
                    "model": model_id,
                    "language": case["language"],
                    "summary_type": case["summary_type"],
                    "run": run,
                    "ttft_seconds": round(ttft, 3) if ttft is not None else "",
                    "total_seconds": round(total, 3),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost_usd": round(cost, 6),
                })

                # Keep the first run's text for side-by-side quality reading.
                if run == 1:
                    answers.append((model_id, label, text))

            if totals:
                print(f"    median total: {statistics.median(totals):.2f}s")

        print()

    if not rows:
        sys.exit("No successful runs. Check the errors above.")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open("answers.txt", "w") as f:
        for model_id, label, text in answers:
            f.write(f"{'=' * 70}\n{model_id}  --  {label}\n{'=' * 70}\n\n{text}\n\n")

    # Per-model rollup: the numbers that answer "which should be the default."
    print(f"{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for model_id in PRICES:
        model_rows = [r for r in rows if r["model"] == model_id]
        if not model_rows:
            continue
        costs = [r["cost_usd"] for r in model_rows]
        times = [r["total_seconds"] for r in model_rows]
        ttfts = [r["ttft_seconds"] for r in model_rows if r["ttft_seconds"] != ""]
        print(f"\n{model_id}")
        print(f"  runs:              {len(model_rows)}")
        print(f"  median cost/call:  ${statistics.median(costs):.6f}")
        print(f"  cost per 1k calls: ${statistics.median(costs) * 1000:.2f}")
        print(f"  median total time: {statistics.median(times):.2f}s")
        if ttfts:
            print(f"  median ttft:       {statistics.median(ttfts):.2f}s")

    print("\nWrote results.csv and answers.txt")


if __name__ == "__main__":
    main()