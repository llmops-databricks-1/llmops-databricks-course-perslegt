# Databricks notebook source
"""
Recipe preprocessing notebook.

Reads recipes_raw, normalizes ingredient text, derives retrieval-friendly fields,
and writes a curated table for downstream filtering and LLM prompting.
"""

import os
import re

import pandas as pd

from recipe_curator.config import get_env, load_config

# COMMAND ----------

spark_session = globals().get("spark")
env = get_env(spark_session)
cfg = load_config("../project_config.yml", env)

SOURCE_TABLE = cfg.recipes_raw_table
TARGET_TABLE = cfg.recipes_curated_table
LOCAL_SOURCE_CSV = "data/recipes_raw.csv"
LOCAL_TARGET_CSV = "data/recipes_curated.csv"

STOPWORDS = {
    "and",
    "or",
    "with",
    "fresh",
    "ground",
    "to",
    "taste",
    "for",
    "into",
    "extra",
    "virgin",
    "optional",
}


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_ingredients(raw_ingredients: str) -> list[str]:
    if not raw_ingredients:
        return []
    return [normalize_text(part) for part in raw_ingredients.split("|") if part.strip()]


def tokenize_ingredients(ingredients: list[str]) -> list[str]:
    tokens: set[str] = set()
    for ingredient in ingredients:
        for token in ingredient.split():
            if token.isdigit() or len(token) <= 2 or token in STOPWORDS:
                continue
            tokens.add(token)
    return sorted(tokens)


def derive_goal_tags(tokens: list[str], category: str, ingredient_count: int) -> list[str]:
    tags: set[str] = set()
    token_set = set(tokens)

    protein_tokens = {
        "chicken",
        "beef",
        "turkey",
        "tuna",
        "salmon",
        "egg",
        "eggs",
        "yogurt",
        "lentils",
        "lentil",
        "beans",
        "tofu",
        "prawn",
    }
    carb_tokens = {"rice", "oats", "pasta", "potato", "bread", "noodles", "banana"}

    if protein_tokens & token_set:
        tags.add("high_protein")
    if carb_tokens & token_set:
        tags.add("pre_workout")
    if ingredient_count <= 5:
        tags.add("quick_meal")
    if category.lower() in {"vegetarian", "vegan"}:
        tags.add("plant_based")

    if not tags:
        tags.add("general")

    return sorted(tags)


def build_recipe_text(name: str, category: str, area: str, tags: list[str], ingredients: list[str]) -> str:
    fields = [name, category, area, " ".join(tags), " ".join(ingredients)]
    return normalize_text(" ".join(field for field in fields if field))


def load_source_dataframe() -> pd.DataFrame:
    if spark_session is not None:
        print(f"Loading source table: {SOURCE_TABLE}")
        return spark_session.table(SOURCE_TABLE).toPandas()

    print(f"Spark not available, loading local CSV: {LOCAL_SOURCE_CSV}")
    return pd.read_csv(LOCAL_SOURCE_CSV)


# COMMAND ----------

raw_df = load_source_dataframe()
print(raw_df.shape)
raw_df.head(10)

# COMMAND ----------

records: list[dict] = []

for row in raw_df.fillna("").to_dict(orient="records"):
    parsed_ingredients = parse_ingredients(row.get("ingredients", ""))
    ingredient_tokens = tokenize_ingredients(parsed_ingredients)
    goal_tags = derive_goal_tags(
        ingredient_tokens,
        str(row.get("category", "")),
        int(row.get("ingredient_count") or len(parsed_ingredients) or 0),
    )

    records.append(
        {
            "recipe_id": str(row.get("recipe_id", "")),
            "name": str(row.get("name", "")).strip(),
            "category": str(row.get("category", "")).strip(),
            "area": str(row.get("area", "")).strip(),
            "instructions": str(row.get("instructions", "")).strip(),
            "ingredients": " | ".join(parsed_ingredients),
            "ingredient_tokens": ", ".join(ingredient_tokens),
            "ingredient_count": len(parsed_ingredients),
            "goal_tags": ", ".join(goal_tags),
            "primary_goal": goal_tags[0],
            "recipe_text": build_recipe_text(
                str(row.get("name", "")),
                str(row.get("category", "")),
                str(row.get("area", "")),
                goal_tags,
                parsed_ingredients,
            ),
            "youtube_url": str(row.get("youtube_url", "")).strip(),
            "source_url": str(row.get("source_url", "")).strip(),
            "data_source": str(row.get("data_source", "")).strip(),
        }
    )

curated_df = pd.DataFrame(records).drop_duplicates(subset=["recipe_id"]).reset_index(drop=True)
print(curated_df.shape)
curated_df.head(10)

# COMMAND ----------

print("Primary goal distribution:")
print(curated_df["primary_goal"].value_counts(dropna=False))

print("\nSample ingredient tokens:")
print(curated_df[["name", "ingredient_tokens", "goal_tags"]].head(10))

# COMMAND ----------

os.makedirs("data", exist_ok=True)
curated_df.to_csv(LOCAL_TARGET_CSV, index=False)
print(f"Saved local CSV: {LOCAL_TARGET_CSV}")

if spark_session is not None:
    spark_session.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")
    spark_df = spark_session.createDataFrame(curated_df)
    spark_df.write.mode("overwrite").saveAsTable(TARGET_TABLE)
    print(f"Saved Unity Catalog table: {TARGET_TABLE}")
else:
    print("Spark session not found. Skipped Unity Catalog write.")