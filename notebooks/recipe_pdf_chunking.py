# Databricks notebook source
"""Chunk parsed recipe PDF elements as complete records.

Reads parsed_recipes_raw, extracts document.elements via Python,
and writes each element as a chunk record with all metadata.
"""

from __future__ import annotations

import os
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = os.getenv("BUNDLE_TARGET") or os.getenv("ENV") or "dev"

CATALOG = "gfmnndipdapmlopsdev"
SCHEMA = "per_slegt"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.parsed_recipes_raw"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.parsed_recipe_chunks"

# Each element from parsed_result.document.elements IS ONE CHUNK RECORD.
# ALL element metadata (bbox, content, description, id, type) preserved as columns.

print(f"Environment: {env}")
print(f"Source table: {SOURCE_TABLE}")
print(f"Target table: {TARGET_TABLE}")

# COMMAND ----------

def extract_elements_from_parsed(variant_json_str: str) -> list:
    """Parse VARIANT and extract document.elements."""
    try:
        if not variant_json_str:
            return []
        parsed = json.loads(variant_json_str) if isinstance(variant_json_str, str) else variant_json_str
        if isinstance(parsed, dict):
            doc = parsed.get("document", {})
            elements = doc.get("elements", []) if isinstance(doc, dict) else []
            return elements if isinstance(elements, list) else []
    except Exception as e:
        print(f"Error: {e}")
    return []

# Register UDF to extract elements from VARIANT
extract_elements_udf = F.udf(
    extract_elements_from_parsed,
    T.ArrayType(T.MapType(T.StringType(), T.StringType()))
)

# COMMAND ----------

# Read source table with parsed recipes
# Convert VARIANT parsed_result to JSON string for UDF processing
source_df = spark.table(SOURCE_TABLE).select(
    "file_path",
    "file_name",
    F.to_json(F.col("parsed_result")).alias("parsed_result_json"),
)

# COMMAND ----------

# Extract elements from parsed_result using UDF
# Each element becomes one row (exploded from array)
elements_df = source_df.withColumn(
    "elements_array",
    extract_elements_udf(F.col("parsed_result_json")),
).select(
    "file_path",
    "file_name",
    F.explode(F.col("elements_array")).alias("element"),
)

print(f"Total elements extracted: {elements_df.count()}")

# COMMAND ----------

# Extract element fields as individual columns
# Each element = ONE CHUNK RECORD with all metadata
chunks_df = elements_df.select(
    F.col("file_path"),
    F.col("file_name"),
    F.col("element")["id"].cast(T.StringType()).alias("element_id"),
    F.col("element")["type"].cast(T.StringType()).alias("element_type"),
    F.col("element")["content"].cast(T.StringType()).alias("content"),
    F.col("element")["description"].cast(T.StringType()).alias("description"),
    F.col("element")["bbox"].cast(T.StringType()).alias("bbox_json"),
)

# Build `content` column used for embedding:
# - For figures/images: use description (richer text than bare filename)
# - For all others: combine content + description if both present
# This gives the embedding model enough context per chunk.
chunks_df = chunks_df.withColumn(
    "content",
    F.when(
        F.col("element_type").isin("figure", "image"),
        F.coalesce(
            F.col("description"),
            F.col("content"),
        ),
    ).otherwise(
        F.when(
            F.col("description").isNotNull() & (F.trim(F.col("description")) != ""),
            F.concat_ws(" | ", F.trim(F.col("content")), F.trim(F.col("description"))),
        ).otherwise(F.col("content"))
    ),
).filter(F.col("content").isNotNull() & (F.trim(F.col("content")) != ""))

# Add chunk_id as SHA256 of (file_path, element_id, content)
chunks_with_id_df = chunks_df.withColumn(
    "chunk_id",
    F.sha2(
        F.concat_ws("::", F.col("file_path"), F.col("element_id"), F.col("content")),
        256,
    )
).withColumn(
    "created_at",
    F.current_timestamp(),
)

# COMMAND ----------

print(f"Total chunks created: {chunks_with_id_df.count()}")
print("\nChunk preview (first 10):")
chunks_with_id_df.select(
    "chunk_id",
    "element_id",
    "element_type",
    "content",
    "file_name",
).limit(10).show(truncate=False)

# Write chunks to target table (one row per element)
chunks_with_id_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)
print(f"✅ Saved {chunks_with_id_df.count()} chunk records to: {TARGET_TABLE}")
