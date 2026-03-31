# Databricks notebook source
"""Create embeddings-based vector search for recipe chunks.

This notebook:
1. Validates parsed chunk data.
2. Ensures a Vector Search endpoint exists.
3. Creates (or reuses) a Delta Sync vector index.
4. Triggers sync and runs a sample similarity query.
"""

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from __future__ import annotations

import os
import time

from pyspark.sql import SparkSession

try:
    from databricks.vector_search.client import VectorSearchClient
except ImportError as exc:
    raise ImportError(
        "databricks-vectorsearch is required in this Databricks runtime."
    ) from exc

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = os.getenv("BUNDLE_TARGET") or os.getenv("ENV") or "dev"


def _load_env_config(config_path: str, env_name: str) -> dict[str, str]:
    """Load a simple environment section from project_config.yml."""
    current_env = ""
    parsed: dict[str, str] = {}

    with open(config_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue

            if not line.startswith(" ") and line.endswith(":"):
                current_env = line[:-1].strip()
                continue

            if current_env != env_name:
                continue

            if line.startswith("  ") and ":" in line:
                key, value = line.strip().split(":", 1)
                parsed[key.strip()] = value.strip().strip('"').strip("'")

    if not parsed:
        raise ValueError(f"No config found for env='{env_name}' in {config_path}")

    return parsed


cfg = _load_env_config("../project_config.yml", env)

CATALOG = cfg["catalog"]
SCHEMA = cfg["schema"]
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.parsed_recipe_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.parsed_recipe_chunks_vs_index"
VECTOR_SEARCH_ENDPOINT = cfg["vector_search_endpoint"]
EMBEDDING_ENDPOINT = cfg["embedding_endpoint"]

user_email = spark.sql("SELECT current_user() AS user_email").first()["user_email"]
USER_SUFFIX = (
    user_email.split("@")[0].replace(".", "_").replace("-", "_")[:20]
)

print(f"Environment: {env}")
print(f"Chunks table: {CHUNKS_TABLE}")
print(f"Vector search endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Vector index: {INDEX_NAME}")
print(f"Embedding endpoint: {EMBEDDING_ENDPOINT}")
print(f"User suffix for fallback resources: {USER_SUFFIX}")

# COMMAND ----------


def _get_nested(data: dict[str, object], *keys: str) -> object | None:
    current: object = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)  # type: ignore[assignment]
        else:
            return None
    return current


def _wait_for_endpoint_online(
    client: VectorSearchClient,
    endpoint_name: str,
    timeout_seconds: int = 1200,
) -> None:
    start = time.time()
    while True:
        endpoint = client.get_endpoint(endpoint_name)
        endpoint_dict = endpoint if isinstance(endpoint, dict) else {}
        state = _get_nested(endpoint_dict, "endpoint_status", "state") or _get_nested(
            endpoint_dict, "status", "state"
        )
        state_str = str(state).upper() if state is not None else "UNKNOWN"

        if state_str == "ONLINE":
            print(f"Endpoint '{endpoint_name}' is ONLINE.")
            return

        if state_str in {"FAILED", "ERROR"}:
            raise RuntimeError(
                f"Endpoint '{endpoint_name}' failed with state: {state_str}"
            )

        elapsed = int(time.time() - start)
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for endpoint '{endpoint_name}' to become ONLINE."
            )

        print(
            f"Waiting for endpoint '{endpoint_name}'... "
            f"state={state_str}, elapsed={elapsed}s"
        )
        time.sleep(20)


def _wait_for_index_ready(
    index: object, timeout_seconds: int = 1200
) -> tuple[bool, str]:
    start = time.time()
    while True:
        if not hasattr(index, "describe"):
            raise RuntimeError("Index object does not expose describe().")
        details = index.describe()
        details_dict = details if isinstance(details, dict) else {}
        ready = _get_nested(details_dict, "status", "ready")
        detailed_state = _get_nested(details_dict, "status", "detailed_state")
        detailed_state_str = (
            str(detailed_state).upper()
            if detailed_state is not None
            else "UNKNOWN"
        )

        if ready is True or detailed_state_str in {"ONLINE", "READY"}:
            print("Vector index is READY.")
            return True, detailed_state_str

        if detailed_state_str in {"FAILED", "ERROR"}:
            raise RuntimeError(
                f"Vector index failed with state: {detailed_state_str}"
            )

        elapsed = int(time.time() - start)
        if elapsed >= timeout_seconds:
            print(
                "Vector index is still provisioning after timeout window. "
                f"Current state={detailed_state_str}."
            )
            return False, detailed_state_str

        print(
            "Waiting for index readiness... "
            f"state={detailed_state_str}, elapsed={elapsed}s"
        )
        time.sleep(20)


def _parse_vector_search_results(
    raw_results: dict[str, object],
) -> list[dict[str, object]]:
    manifest = raw_results.get("manifest", {})
    result = raw_results.get("result", {})

    columns: list[str] = []
    if isinstance(manifest, dict):
        raw_columns = manifest.get("columns", [])
        if isinstance(raw_columns, list):
            for col in raw_columns:
                if isinstance(col, dict):
                    col_name = col.get("name")
                    if isinstance(col_name, str):
                        columns.append(col_name)

    data_array: list[object] = []
    if isinstance(result, dict):
        raw_data_array = result.get("data_array", [])
        if isinstance(raw_data_array, list):
            data_array = raw_data_array

    parsed_rows: list[dict[str, object]] = []
    for row in data_array:
        if isinstance(row, list):
            parsed_rows.append(dict(zip(columns, row, strict=False)))
    return parsed_rows


# COMMAND ----------

valid_chunks_df = spark.table(CHUNKS_TABLE).where(
    "content IS NOT NULL AND TRIM(content) <> ''"
)
chunk_count = valid_chunks_df.count()

if chunk_count == 0:
    raise RuntimeError(
        f"No valid chunk rows found in {CHUNKS_TABLE}. Run chunking first."
    )

print(f"Valid chunk rows available: {chunk_count}")

# Required for robust Delta Sync indexing
spark.sql(
    f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print("Enabled delta.enableChangeDataFeed=true on source chunk table.")

# COMMAND ----------

vsc = VectorSearchClient()
resolved_endpoint = VECTOR_SEARCH_ENDPOINT
resolved_index = INDEX_NAME

try:
    vsc.get_endpoint(resolved_endpoint)
    print(f"Endpoint '{resolved_endpoint}' already exists.")
except Exception as endpoint_exc:
    endpoint_error_text = str(endpoint_exc).lower()
    if any(
        token in endpoint_error_text
        for token in ["not authorized", "permission", "already exists"]
    ):
        resolved_endpoint = f"{VECTOR_SEARCH_ENDPOINT}_{USER_SUFFIX}"
        resolved_index = (
            f"{CATALOG}.{SCHEMA}.parsed_recipe_chunks_vs_index_{USER_SUFFIX}"
        )
        print(
            "Switching to user-scoped endpoint "
            f"'{resolved_endpoint}' due to access/conflict on "
            f"'{VECTOR_SEARCH_ENDPOINT}'."
        )

    try:
        vsc.get_endpoint(resolved_endpoint)
        print(f"Endpoint '{resolved_endpoint}' already exists.")
    except Exception:
        print(f"Creating endpoint '{resolved_endpoint}'...")
        vsc.create_endpoint(name=resolved_endpoint, endpoint_type="STANDARD")

_wait_for_endpoint_online(vsc, resolved_endpoint)

# COMMAND ----------

try:
    index = vsc.get_index(endpoint_name=resolved_endpoint, index_name=resolved_index)
    print(f"Index '{resolved_index}' already exists.")
except Exception:
    print(f"Creating index '{resolved_index}'...")
    vsc.create_delta_sync_index(
        endpoint_name=resolved_endpoint,
        index_name=resolved_index,
        source_table_name=CHUNKS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="content",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    index = vsc.get_index(endpoint_name=resolved_endpoint, index_name=resolved_index)

index_ready, index_state = _wait_for_index_ready(index)

if index_ready:
    print("Triggering index sync...")
    index.sync()
    index_ready, index_state = _wait_for_index_ready(index)
else:
    print(
        "Skipping sync for now because the index is still provisioning. "
        f"Current state={index_state}."
    )

# COMMAND ----------

if index_ready:
    query = "high protein recipe"

    results = index.similarity_search(
        query_text=query,
        columns=["chunk_id", "file_name", "element_type", "content"],
        num_results=5,
    )

    parsed_rows = _parse_vector_search_results(results)

    print(f"Similarity query: {query}")
    print(f"Rows returned: {len(parsed_rows)}")

    for idx, row in enumerate(parsed_rows, start=1):
        content_preview = (row.get("content") or "")[:140]
        print(
            f"{idx}. file={row.get('file_name')} "
            f"type={row.get('element_type')} "
            f"score={row.get('score')}"
        )
        print(f"   content={content_preview}")
else:
    print(
        "Vector index setup request completed, but the index is not query-ready yet. "
        f"Current state={index_state}. Re-run this notebook later "
        "to verify search results."
    )
