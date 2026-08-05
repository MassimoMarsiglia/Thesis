import asyncio
import json
import os
import random
from collections.abc import AsyncIterator

import aiofiles
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# ============================================================
# Config
# ============================================================

load_dotenv(dotenv_path="./bin/.env")

MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


TRAIN_FILE = "data/training_pairs.jsonl"
NOISE_FILE = "data/noise_patterns.jsonl"

OUTPUT_FILE = "data/enriched_training_pairs.jsonl"


CONCURRENCY = 10
MAX_PENDING = 50
NOISE_CONTEXT_SIZE = 7


# ============================================================
# Schema
# ============================================================


class EnrichedExample(BaseModel):
    title: str
    description: str
    title_strategy: str
    description_strategy: str
    rationale: str


class GenerationResult(BaseModel):
    examples: list[EnrichedExample]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You generate synthetic training examples for an embedding model.

The source entity contains:

- title
- description

Generate alternative references to the SAME entity.

For every example produce:

title:
- an alternative title reference
- may be shortened, abbreviated, identifier-only, partial, etc.

description:
- an alternative description
- may remove boilerplate
- may shorten descriptive clauses
- may extract key phrases
- must remain faithful to source facts

title_strategy:
- short label describing title transformation

description_strategy:
- short label describing description transformation

rationale:
- one sentence explaining retrieval robustness.

Rules:

- Never invent facts.
- Never add entities.
- Never add dates.
- Never add organizations.
- Never add citations.
- Description MUST NOT be empty.
- Description MUST differ from source wording.
- Return 3-5 examples.
"""


def build_prompt(pair, noise_patterns):

    return f"""
Reference patterns:

{json.dumps(noise_patterns, indent=2)}

The Following additional data is allowed to be used

Aliases:
{pair["json"]["positive"]["aliases"]}

Short Title:
{pair["json"]["positive"]["short_title"]}

Official Title:
{pair["json"]["positive"]["title"]}

Source entity:

Title:
{pair["json"]["reference"]["ref_title"]}

Description:
{pair["json"]["reference"]["description"]}

Generate 3-5 examples.

Each example must contain:

- title
- description
- title_strategy
- description_strategy
- rationale

The description must:
- preserve facts
- be rewritten
- not be empty
- use description patterns
"""


# ============================================================
# Streaming
# ============================================================


async def stream_jsonl(path: str) -> AsyncIterator[dict]:

    async with aiofiles.open(path, "r") as f:
        async for line in f:
            if line.strip():
                yield json.loads(line)


async def reservoir_sample(stream, size):

    result = []

    i = 0

    async for item in stream:
        if i < size:
            result.append(item)

        else:
            j = random.randint(0, i)

            if j < size:
                result[j] = item

        i += 1

    return result


# ============================================================
# LLM
# ============================================================


async def enrich(semaphore, pair, noise_patterns, idx):

    async with semaphore:
        try:
            response = await client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0.3,
                response_format=GenerationResult,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_prompt(pair, noise_patterns),
                    },
                ],
            )

            result = response.choices[0].message.parsed

            return {
                "source": pair,
                "generated": [x.model_dump() for x in result.examples],
                "synthetic": True,
            }

        except Exception as e:
            print(f"[FAILED {idx}] {e}")

            return None


# ============================================================
# Pipeline
# ============================================================


async def run():

    noise_patterns = await reservoir_sample(
        stream_jsonl(NOISE_FILE),
        NOISE_CONTEXT_SIZE,
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)

    pending = set()

    async with aiofiles.open(OUTPUT_FILE, "w") as output:
        idx = 0

        async for pair in stream_jsonl(TRAIN_FILE):
            task = asyncio.create_task(enrich(semaphore, pair, noise_patterns, idx))

            pending.add(task)

            idx += 1

            if len(pending) >= MAX_PENDING:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    result = await task

                    if result:
                        await output.write(json.dumps(result) + "\n")

        for result in await asyncio.gather(*pending):
            if result:
                await output.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    asyncio.run(run())
