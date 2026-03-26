# Databricks notebook source
"""
Recipe data ingestion notebook.

Pulls recipes from TheMealDB API, flattens ingredients, adds lightweight
sports-oriented tags, and optionally writes to a Unity Catalog table.
"""

import json
import os
import re
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from recipe_curator.config import get_env, load_config

# COMMAND ----------

spark_session = globals().get("spark")
env = get_env(spark_session)
cfg = load_config("../project_config.yml", env)

BASE_URL = "https://www.themealdb.com/api/json/v1/1"
MAX_RECIPES = 300
REQUEST_DELAY_SECONDS = 0.05
CATALOG = cfg.catalog
SCHEMA = cfg.schema
TARGET_TABLE = cfg.recipes_raw_table

FALLBACK_MEALS = [
    {
        "idMeal": "fallback-001",
        "strMeal": "Chicken Rice Power Bowl",
        "strCategory": "Chicken",
        "strArea": "International",
        "strInstructions": (
            "Cook the rice, grill the chicken, and serve with spinach and yogurt."
        ),
        "strIngredient1": "Chicken Breast",
        "strMeasure1": "200g",
        "strIngredient2": "Rice",
        "strMeasure2": "150g",
        "strIngredient3": "Spinach",
        "strMeasure3": "2 handfuls",
        "strIngredient4": "Greek Yogurt",
        "strMeasure4": "3 tbsp",
        "strYoutube": "",
        "strSource": "",
    },
    {
        "idMeal": "fallback-002",
        "strMeal": "Oats Banana Pre-Workout",
        "strCategory": "Breakfast",
        "strArea": "International",
        "strInstructions": "Mix oats with milk, top with banana and peanut butter.",
        "strIngredient1": "Oats",
        "strMeasure1": "80g",
        "strIngredient2": "Banana",
        "strMeasure2": "1",
        "strIngredient3": "Milk",
        "strMeasure3": "250ml",
        "strIngredient4": "Peanut Butter",
        "strMeasure4": "1 tbsp",
        "strYoutube": "",
        "strSource": "",
    },
    {
        "idMeal": "fallback-003",
        "strMeal": "Lentil Tomato Recovery Soup",
        "strCategory": "Vegetarian",
        "strArea": "International",
        "strInstructions": "Simmer lentils, tomato, onion, and spices until tender.",
        "strIngredient1": "Lentils",
        "strMeasure1": "200g",
        "strIngredient2": "Tomato",
        "strMeasure2": "2",
        "strIngredient3": "Onion",
        "strMeasure3": "1",
        "strIngredient4": "Vegetable Stock",
        "strMeasure4": "750ml",
        "strYoutube": "",
        "strSource": "",
    },
]


def api_get(endpoint: str, params: dict | None = None) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{BASE_URL}/{endpoint}{query}"
    with urlopen(url, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def list_recipe_ids_by_letter(letter: str) -> list[str]:
    payload = api_get("search.php", {"f": letter})
    meals = payload.get("meals") or []
    return [meal["idMeal"] for meal in meals if "idMeal" in meal]


def fetch_recipe_by_id(recipe_id: str) -> dict | None:
    payload = api_get("lookup.php", {"i": recipe_id})
    meals = payload.get("meals") or []
    return meals[0] if meals else None


def fetch_api_recipes() -> list[dict]:
    all_ids: list[str] = []
    for char_code in range(ord("a"), ord("z") + 1):
        letter = chr(char_code)
        all_ids.extend(list_recipe_ids_by_letter(letter))

    all_ids = sorted(set(all_ids))
    selected_ids = all_ids[:MAX_RECIPES]

    print(f"Discovered recipe IDs: {len(all_ids)}")
    print(f"Selected for ingestion: {len(selected_ids)}")

    recipes: list[dict] = []
    for idx, recipe_id in enumerate(selected_ids, start=1):
        meal = fetch_recipe_by_id(recipe_id)
        if meal:
            recipes.append(meal)
        if idx % 50 == 0:
            print(f"Fetched {idx}/{len(selected_ids)} recipes")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Fetched recipe payloads: {len(recipes)}")
    return recipes


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    return re.sub(r"\s+", " ", value)


def extract_ingredients(meal: dict) -> list[str]:
    ingredients: list[str] = []
    for i in range(1, 21):
        ingredient = (meal.get(f"strIngredient{i}") or "").strip()
        measure = (meal.get(f"strMeasure{i}") or "").strip()
        if ingredient:
            merged = f"{measure} {ingredient}".strip()
            ingredients.append(normalize_text(merged))
    return ingredients


def infer_goal_tags(ingredients: list[str], category: str) -> str:
    ingredient_blob = " ".join(ingredients)
    tags: list[str] = []

    protein_keywords = [
        "chicken",
        "beef",
        "turkey",
        "fish",
        "tuna",
        "salmon",
        "egg",
        "yogurt",
        "lentil",
        "beans",
        "tofu",
        "prawn",
    ]
    carb_keywords = ["rice", "pasta", "potato", "oats", "bread", "noodle"]
    quick_keywords = ["sandwich", "salad", "omelette", "wrap", "soup"]

    if any(k in ingredient_blob for k in protein_keywords):
        tags.append("high_protein")
    if any(k in ingredient_blob for k in carb_keywords):
        tags.append("pre_workout")
    if any(k in ingredient_blob for k in quick_keywords):
        tags.append("quick_meal")
    if category.lower() in {"vegetarian", "vegan"}:
        tags.append("plant_based")

    if not tags:
        tags.append("general")

    return ",".join(sorted(set(tags)))


# COMMAND ----------

try:
    recipes = fetch_api_recipes()
    data_source = "themealdb_api"
except Exception as exc:
    print(f"API ingestion failed: {exc}")
    print("Falling back to embedded sample recipes so the table can still be created.")
    recipes = FALLBACK_MEALS
    data_source = "embedded_fallback"

print(f"Using data source: {data_source}")

# COMMAND ----------

records: list[dict] = []
for meal in recipes:
    ingredients = extract_ingredients(meal)
    category = (meal.get("strCategory") or "unknown").strip()
    area = (meal.get("strArea") or "unknown").strip()

    records.append(
        {
            "recipe_id": meal.get("idMeal"),
            "name": (meal.get("strMeal") or "").strip(),
            "category": category,
            "area": area,
            "instructions": (meal.get("strInstructions") or "").strip(),
            "ingredients": " | ".join(ingredients),
            "ingredient_count": len(ingredients),
            "youtube_url": (meal.get("strYoutube") or "").strip(),
            "source_url": (meal.get("strSource") or "").strip(),
            "goal_tags": infer_goal_tags(ingredients, category),
            "data_source": data_source,
        }
    )

df = pd.DataFrame(records)
print(df.shape)
df.head(10)

# COMMAND ----------

print("Goal tag distribution:")
print(df["goal_tags"].value_counts().head(15))

# COMMAND ----------

os.makedirs("data", exist_ok=True)
local_output = "data/recipes_raw.csv"
df.to_csv(local_output, index=False)
print(f"Saved local CSV: {local_output}")

if spark_session is not None:
    spark_df = spark_session.createDataFrame(df)
    spark_session.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark_df.write.mode("overwrite").saveAsTable(TARGET_TABLE)
    print(f"Saved Unity Catalog table: {TARGET_TABLE}")
else:
    print("Spark session not found. Skipped Unity Catalog write.")
