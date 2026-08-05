import asyncio
import json
import os

import aiofiles
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ============================================================
# Config
# ============================================================

load_dotenv("./bin/.env")

MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


INPUT_FILE = "data/unlabeled_refs.jsonl"
OUTPUT_FILE = "data/noise_patterns.jsonl"


BATCH_SIZE = 128


# ============================================================
# Schema
# ============================================================


class NoisePatterns(BaseModel):
    title_patterns: list[str] = Field(
        description="Canonical reusable title transformation patterns"
    )

    description_patterns: list[str] = Field(
        description="Canonical reusable description transformation patterns"
    )


# ============================================================
# Prompt
# ============================================================


SYSTEM_PROMPT = """
You are building a retrieval noise taxonomy.

You receive:
1. Existing discovered patterns.
2. A new batch of document references.

Your task is to refine the taxonomy.

Think contrastively:
Compare references in the batch and identify how users might refer to
the same document without typing the complete original title.

Only discover surface-form retrieval transformations.

Separate:

TITLE PATTERNS:
How the identifying title changes.

DESCRIPTION PATTERNS:
How descriptive clauses are removed or changed.

Allowed examples:

TITLE:
- title shortened
- partial title
- acronym substitution
- abbreviation used
- identifier used instead of title
- informal short name
- word order variation
- punctuation variation

DESCRIPTION:
- boilerplate removed
- date removed
- descriptive clauses omitted
- document type retained while description removed
- key phrase extracted
- organization removed

Rules:

- Merge duplicates.
- Prefer short canonical names.
- Do not invent document-specific patterns.
- Do not mention names, dates, countries, organizations, laws, or topics.
- Do not explain.

Return only the updated pattern lists.
"""


# ============================================================
# Streaming dataset
# ============================================================


async def stream_batches(size):

    dataset = load_dataset(
        "json",
        data_files=INPUT_FILE,
        split="train",
        streaming=True,
    )

    batch = []

    for item in dataset:
        batch.append(item)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


# ============================================================
# LLM refinement
# ============================================================


async def refine_patterns(
    current_patterns,
    batch,
):

    response = await client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.3,
        response_format=NoisePatterns,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "existing_patterns": current_patterns,
                        "new_references": batch,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )

    return response.choices[0].message.parsed.model_dump()


# ============================================================
# Main pipeline
# ============================================================


async def run():

    patterns = {
        "title_patterns": [],
        "description_patterns": [],
    }

    batch_count = 0

    async with aiofiles.open(
        OUTPUT_FILE,
        "w",
    ) as output:
        async for batch in stream_batches(BATCH_SIZE):
            batch_count += 1

            print(f"Processing batch {batch_count}")

            patterns = await refine_patterns(
                patterns,
                batch,
            )

            line = (
                json.dumps(
                    patterns,
                    ensure_ascii=False,
                )
                + "\n"
            )

            await output.write(line)

            # force data to disk
            await output.flush()

            print(
                f"Wrote batch {batch_count}: "
                f"{len(patterns['title_patterns'])} title, "
                f"{len(patterns['description_patterns'])} description"
            )

    print("Done")


if __name__ == "__main__":
    asyncio.run(run())
