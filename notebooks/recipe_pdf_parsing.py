# Databricks notebook source
# Recipe PDF Parser with AI Parse Documents
# Extract and parse text from recipe PDFs using Databricks AI Parse Documents API

# COMMAND ----------

# Setup & Configuration
# Load required libraries and initialize Databricks connection

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from databricks.sdk import WorkspaceClient
    HAS_DATABRICKS_SDK = True
except ImportError:
    HAS_DATABRICKS_SDK = False
    logger.warning("⚠️ databricks-sdk not available. Some features may not work.")

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

print("✅ Setup complete. AI Parse Documents ready.")
print(f"Databricks SDK available: {HAS_DATABRICKS_SDK}")

# COMMAND ----------

# Constants
CATALOG = "gfmnndipdapmlopsdev"
SCHEMA = "per_slegt"
VOLUME = "recipe_files"

VOLUME_RAW_PDFS_DIR = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_pdfs")
VOLUME_OUTPUT_DIR = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/parsed_outputs")
LOCAL_DATA_DIR = Path("data")

# Prefer Unity Catalog volume paths in Databricks jobs.
if VOLUME_RAW_PDFS_DIR.exists():
    PDF_DIR = VOLUME_RAW_PDFS_DIR
    OUTPUT_DIR = VOLUME_OUTPUT_DIR
else:
    PDF_DIR = LOCAL_DATA_DIR
    OUTPUT_DIR = LOCAL_DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using PDF_DIR: {PDF_DIR}")
print(f"Using OUTPUT_DIR: {OUTPUT_DIR}")

# COMMAND ----------

# Function: Discover PDFs
# Scan the data/ folder and identify all PDF files

def discover_pdfs(search_dir: Path) -> list[dict]:
    """
    Discover all PDF files in a directory.
    
    Returns:
        List of dicts with file_name, file_path, file_size_bytes
    """
    pdfs = []
    
    if not search_dir.exists():
        logger.error(f"Directory not found: {search_dir}")
        return pdfs
    
    for pdf_file in sorted(search_dir.glob("*.pdf")):
        file_size = pdf_file.stat().st_size
        pdfs.append({
            "file_name": pdf_file.name,
            "file_path": str(pdf_file.absolute()),
            "file_size_bytes": file_size,
        })

    # Fallback for Databricks workspace paths where pathlib glob may not see files.
    if not pdfs and str(search_dir).startswith("/Workspace") and HAS_DATABRICKS_SDK:
        try:
            w = WorkspaceClient()
            for obj in w.workspace.list(str(search_dir)):
                if obj.path and obj.path.lower().endswith(".pdf"):
                    pdf_path = obj.path
                    pdfs.append(
                        {
                            "file_name": Path(pdf_path).name,
                            "file_path": pdf_path,
                            "file_size_bytes": 0,
                        }
                    )
        except Exception as exc:
            logger.warning(f"Workspace listing fallback failed: {exc}")

    # Fallback without SDK: search recursively from current working directory.
    if not pdfs:
        try:
            cwd = Path.cwd()
            for pdf_file in sorted(cwd.rglob("*.pdf")):
                try:
                    file_size = pdf_file.stat().st_size
                except OSError:
                    file_size = 0
                pdfs.append(
                    {
                        "file_name": pdf_file.name,
                        "file_path": str(pdf_file),
                        "file_size_bytes": file_size,
                    }
                )
        except Exception as exc:
            logger.warning(f"Recursive PDF fallback failed: {exc}")
    
    logger.info(f"Found {len(pdfs)} PDF files")
    return pdfs

pdfs = discover_pdfs(PDF_DIR)
for pdf in pdfs:
    print(f"  - {pdf['file_name']} ({pdf['file_size_bytes'] / 1024:.1f} KB)")

# COMMAND ----------

# Initialize Spark
# Required for SQL queries with ai_parse_document

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    print("✅ SparkSession initialized")
except Exception as e:
    logger.error(f"Failed to initialize Spark: {e}")
    raise

# COMMAND ----------

# Build and execute parsing SQL
# Use Databricks ai_parse_document to extract text, tables, and structure

def build_parse_sql() -> str:
    """Build SQL query using ai_parse_document for PDF parsing."""
    volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_pdfs"
    
    sql_query = f"""
    SELECT 
        path AS file_path,
        reverse(split(path, '/'))[0] AS file_name,
        ai_parse_document(
            content,
            map(
                'language', 'en',
                'format_version', '2.0'
            )
        ) AS parsed_result
    FROM read_files('{volume_path}/*.pdf', format => 'binaryFile')
    """
    
    return sql_query

# COMMAND ----------

# Execute parsing
# Run ai_parse_document on all PDFs in volume

print("\n🔍 Parsing PDFs with Databricks ai_parse_document...\n")

try:
    sql_query = build_parse_sql()
    logger.info("Executing ai_parse_document SQL query...")
    
    # Execute the query
    parsed_df = spark.sql(sql_query)
    
    # Show results
    print(f"✅ Parsing complete. Found {parsed_df.count()} PDFs processed.\n")
    
    # Display sample
    display_df = parsed_df.select("file_path", "file_name")
    if display_df.count() > 0:
        display_df.show(truncate=False, n=5)
    
    # Save to Delta table
    parsed_table_name = f"{CATALOG}.{SCHEMA}.parsed_recipes_raw"
    (
        parsed_df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(parsed_table_name)
    )
    logger.info(f"✅ Results saved to Delta table: {parsed_table_name}")
    
    # Also save a summary CSV
    summary_df = parsed_df.select("file_path", "file_name")
    csv_output = OUTPUT_DIR / "parsing_summary.csv"
    summary_df.toPandas().to_csv(csv_output, index=False)
    logger.info(f"✅ Summary saved to CSV: {csv_output}")
    
    print(f"\n📊 Parsing Results:")
    print(f"  Total PDFs processed: {parsed_df.count()}")
    
except Exception as e:
    logger.error(f"❌ Parsing failed: {e}")
    raise
