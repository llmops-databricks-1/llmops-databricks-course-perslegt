# Databricks notebook source
"""
Recipe LLM experiments notebook.

Flow:
1. Load recipes_curated from Unity Catalog.
2. Retrieve top matches for user ingredients + fitness goal.
3. Ask Databricks-hosted LLM to produce final recommendations.
"""

import json
import os
import re
from urllib.request import Request, urlopen

import pandas as pd
from databricks.sdk import WorkspaceClient

from recipe_curator.config import get_env, load_config

# COMMAND ----------

spark_session = globals().get("spark")
if spark_session is None:
    raise RuntimeError("This notebook must run with a Spark session on Databricks.")

env = get_env(spark_session)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
CURATED_TABLE = cfg.recipes_curated_table
MODEL_ENDPOINT = cfg.llm_endpoint

USER_INGREDIENTS = "chicken, rice, pepper"
USER_QUERY = "I have chicken, rice, and pepper. What can I make?"
USER_GOAL = "high_protein"
TOP_K = 5

INGREDIENT_ALIASES = {
    "kip": "chicken",
    "rijst": "rice",
    "paprika": "pepper",
    "ei": "egg",
    "eieren": "egg",
    "tomaat": "tomato",
    "linzen": "lentils",
}

try:
    _ = display  # type: ignore[name-defined]
except NameError:

    def display(value: object) -> None:
        print(value)


def normalize_token(value: str) -> str:
    return value.lower().strip().replace("-", "_")


def parse_csv_tokens(value: str) -> set[str]:
    if not value:
        return set()
    return {normalize_token(part) for part in value.split(",") if part.strip()}


def ingredients_from_query(query: str) -> str:
    words = [normalize_token(token) for token in re.findall(r"[a-zA-Z_]+", query)]
    normalized = [INGREDIENT_ALIASES.get(word, word) for word in words]
    filtered = [
        word
        for word in normalized
        if len(word) > 2
        and word
        not in {
            "heb",
            "wat",
            "kan",
            "maken",
            "met",
            "and",
            "have",
            "what",
            "make",
            "with",
        }
    ]

    # Keep insertion order while removing duplicates.
    unique = list(dict.fromkeys(filtered))
    return ", ".join(unique)


def stringify_llm_content(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "\n".join(part for part in text_parts if part)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=True)

    return str(content)


def ingredient_overlap_score(user_tokens: set[str], recipe_tokens: set[str]) -> int:
    return len(user_tokens & recipe_tokens)


def goal_match_score(user_goal: str, recipe_goal_tags: set[str]) -> int:
    return 2 if user_goal in recipe_goal_tags else 0


def compute_ranked_candidates(
    df: pd.DataFrame, user_ingredients: str, user_goal: str, top_k: int
) -> pd.DataFrame:
    user_tokens = parse_csv_tokens(user_ingredients)
    user_goal_token = normalize_token(user_goal)

    work_df = df.copy()
    work_df["ingredient_token_set"] = (
        work_df["ingredient_tokens"].fillna("").apply(parse_csv_tokens)
    )
    work_df["goal_tag_set"] = work_df["goal_tags"].fillna("").apply(parse_csv_tokens)

    work_df["ingredient_overlap"] = work_df["ingredient_token_set"].apply(
        lambda recipe_tokens: ingredient_overlap_score(user_tokens, recipe_tokens)
    )
    work_df["goal_score"] = work_df["goal_tag_set"].apply(
        lambda tags: goal_match_score(user_goal_token, tags)
    )
    work_df["rank_score"] = work_df["ingredient_overlap"] + work_df["goal_score"]

    ranked = work_df.sort_values(
        by=["rank_score", "ingredient_overlap", "ingredient_count"],
        ascending=[False, False, True],
    )

    ranked = ranked[ranked["rank_score"] > 0].head(top_k)
    return ranked.reset_index(drop=True)


def build_llm_messages(
    user_ingredients: str,
    user_goal: str,
    candidates: list[dict[str, object]],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a sports nutrition recipe assistant. "
        "Recommend recipes grounded only in provided candidates. "
        "Return concise advice with substitutions if ingredients are missing."
    )

    user_payload = {
        "goal": user_goal,
        "ingredients": user_ingredients,
        "candidates": candidates,
        "required_output": {
            "format": "markdown",
            "rules": [
                "Return top 3 recommendations.",
                (
                    "For each recommendation include: name, short reason, "
                    "and optional substitution."
                ),
                "Keep total response under 220 words.",
            ],
        },
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


# COMMAND ----------

recipes_df = spark_session.table(CURATED_TABLE).toPandas()
print(f"Loaded curated rows: {len(recipes_df)}")

effective_ingredients = (
    ingredients_from_query(USER_QUERY) if USER_QUERY.strip() else USER_INGREDIENTS
)
print(f"Using ingredients: {effective_ingredients}")

# COMMAND ----------

ranked_df = compute_ranked_candidates(
    recipes_df,
    user_ingredients=effective_ingredients,
    user_goal=USER_GOAL,
    top_k=TOP_K,
)

display(
    ranked_df[
        [
            "recipe_id",
            "name",
            "ingredient_overlap",
            "rank_score",
            "goal_tags",
            "ingredient_tokens",
        ]
    ]
)

if ranked_df.empty:
    raise RuntimeError("No candidate recipes found. Try a broader ingredient list.")

# COMMAND ----------

candidates_for_llm = ranked_df[
    [
        "recipe_id",
        "name",
        "category",
        "goal_tags",
        "ingredients",
        "ingredient_tokens",
        "instructions",
    ]
].to_dict(orient="records")

messages = build_llm_messages(effective_ingredients, USER_GOAL, candidates_for_llm)

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

payload = {
    "model": MODEL_ENDPOINT,
    "messages": messages,
    "temperature": 0.2,
}

request = Request(
    url=f"{host}/serving-endpoints/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)

with urlopen(request, timeout=120) as response:  # noqa: S310
    response_payload = json.loads(response.read().decode("utf-8"))

llm_text = stringify_llm_content(response_payload["choices"][0]["message"]["content"])
print(llm_text)

# COMMAND ----------

os.makedirs("data", exist_ok=True)
output_path = "data/recipe_llm_latest.txt"

with open(output_path, "w", encoding="utf-8") as file:
    file.write(llm_text or "")

print(f"Saved LLM output to {output_path}")
