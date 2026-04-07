# Databricks notebook source
# MAGIC %md
# MAGIC # Recipe Agent Orchestration (Week 3 Homework)
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook implements a simple recipe agent with:
# MAGIC 1. Intent classification
# MAGIC 2. Tool calling (recipe vector search)
# MAGIC 3. Conversation memory
# MAGIC 4. Multi-turn conversation demo

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch openai loguru

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

import json
import os
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = os.getenv("BUNDLE_TARGET") or os.getenv("ENV") or "dev"


# Load config
def _load_env_config(config_path: str, env_name: str) -> dict[str, str]:
    """Load environment configuration from project_config.yml."""
    current_env = ""
    parsed: dict[str, str] = {}
    with open(config_path, encoding="utf-8") as file:
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
LLM_ENDPOINT = cfg["llm_endpoint"]

# Initialize clients
w = WorkspaceClient()
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
)

client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

# Resolve index name — same fallback logic as recipe_embeddings_vector_search.py:
# if the shared endpoint was inaccessible during setup, a user-scoped index was created.
user_email = spark.sql("SELECT current_user() AS user_email").first()["user_email"]
USER_SUFFIX = user_email.split("@")[0].replace(".", "_").replace("-", "_")[:20]

_base_index = f"{CATALOG}.{SCHEMA}.parsed_recipe_chunks_vs_index"
_user_index = f"{_base_index}_{USER_SUFFIX}"

try:
    vsc.get_index(index_name=_base_index)
    INDEX_NAME = _base_index
    logger.info(f"Using shared index: {INDEX_NAME}")
except Exception:
    INDEX_NAME = _user_index
    logger.info(f"Shared index not found, using user-scoped index: {INDEX_NAME}")

logger.info(f"Environment: {env}")
logger.info(f"Catalog: {CATALOG}, Schema: {SCHEMA}")
logger.info(f"Vector Index: {INDEX_NAME}")
logger.info(f"LLM Endpoint: {LLM_ENDPOINT}")
logger.info("✓ Clients initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Intent Classification

# COMMAND ----------


class IntentClassifier:
    """Classify user intent for recipe queries."""

    INTENTS = {
        "recipe_search": (
            "Search for recipes based on ingredients, cuisine, or preferences"
        ),
        "recipe_adapt": "Adapt or modify an existing recipe",
        "general_info": "General questions about cooking or nutrition",
        "unknown": "Intent not recognized",
    }

    @staticmethod
    def classify(query: str) -> str:
        """Classify query into one of the defined intents.

        Args:
            query: User input

        Returns:
            Intent label
        """
        query_lower = query.lower()

        # Recipe search keywords
        search_keywords = [
            "find",
            "give me",
            "show",
            "search",
            "what",
            "recipe",
            "meal",
            "dish",
        ]

        # Recipe adapt keywords
        adapt_keywords = [
            "change",
            "make",
            "adapt",
            "modify",
            "substitute",
            "without",
            "replace",
        ]

        # Check for adapt first (more specific)
        if any(word in query_lower for word in adapt_keywords):
            return "recipe_adapt"

        # Check for search
        if any(word in query_lower for word in search_keywords):
            return "recipe_search"

        # Check for general questions
        if any(word in query_lower for word in ["how", "why", "what is", "tell me"]):
            return "general_info"

        return "unknown"


# Test classifier
test_queries = [
    "Give me a quick vegetarian dinner",
    "Make this recipe without dairy",
    "What is a roux?",
]

logger.info("Testing Intent Classifier:")
for q in test_queries:
    intent = IntentClassifier.classify(q)
    logger.info(f"  '{q}' → {intent}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Recipe Search Tool

# COMMAND ----------


class RecipeSearchTool:
    """Tool for searching recipes using vector search."""

    def __init__(self, vsc_client: VectorSearchClient, index_name: str):
        self.vsc = vsc_client
        self.index_name = index_name

    def execute(self, query: str, num_results: int = 3) -> dict[str, object]:
        """Search for recipes matching the query.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Dictionary with recipes and metadata
        """
        try:
            index = self.vsc.get_index(index_name=self.index_name)

            results = index.similarity_search(
                query_text=query,
                columns=["content", "file_name", "element_type", "chunk_id"],
                num_results=num_results,
                query_type="hybrid",
            )

            recipes = []
            if results and "result" in results:
                manifest = results.get("manifest", {})
                columns = []
                if isinstance(manifest, dict):
                    raw_columns = manifest.get("columns", [])
                    if isinstance(raw_columns, list):
                        for col in raw_columns:
                            if isinstance(col, dict) and "name" in col:
                                columns.append(col["name"])

                data_array = results["result"].get("data_array", [])
                for row in data_array:
                    row_dict = dict(zip(columns, row, strict=False)) if columns else {}
                    recipes.append(
                        {
                            "text": str(row_dict.get("content", "")),
                            "recipe_name": str(row_dict.get("file_name", "Unknown")),
                            "source": str(row_dict.get("element_type", "Unknown")),
                            "chunk_id": str(row_dict.get("chunk_id", "")),
                        }
                    )

            return {
                "success": True,
                "query": query,
                "results": recipes,
                "count": len(recipes),
            }
        except Exception as e:
            logger.error(f"Recipe search error: {e}")
            return {"success": False, "error": str(e), "query": query}


# Initialize tool
recipe_tool = RecipeSearchTool(vsc, INDEX_NAME)

# Test the tool
logger.info("Testing Recipe Search Tool:")
test_search = recipe_tool.execute("vegetarian quick dinner", num_results=2)
if test_search["success"]:
    logger.info(f"Search result: {test_search['count']} recipes found")
    if test_search["results"]:
        logger.info(f"  First result: {test_search['results'][0]['recipe_name']}")
else:
    logger.warning(f"Search not available yet: {test_search['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tool Specification (for LLM)

# COMMAND ----------

RECIPE_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "search_recipes",
        "description": (
            "Search for recipes based on ingredients, cuisine, cooking time, "
            "or dietary preferences"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing what kind of recipe you want "
                        "(e.g., 'quick pasta', 'vegan dessert')"
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of recipes to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("Tool specification for LLM:")
logger.info(json.dumps(RECIPE_SEARCH_SPEC, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Conversation Memory

# COMMAND ----------


class ConversationMemory:
    """Simple in-memory conversation history."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Get all messages in conversation."""
        return self.messages.copy()

    def get_context(self) -> str:
        """Get summary of conversation for context."""
        if not self.messages:
            return ""

        summary_parts = []
        for msg in self.messages[-4:]:  # Last 4 messages for context
            role = msg["role"].upper()
            content = msg["content"][:100]  # Truncate long messages
            summary_parts.append(f"{role}: {content}...")

        return "\n".join(summary_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []


# Test memory
session_id = f"recipe-agent-{uuid4()}"
memory = ConversationMemory(session_id)

memory.add_message("user", "Give me a vegetarian recipe")
memory.add_message("assistant", "Here's a nice vegetable pasta...")
memory.add_message("user", "Can you make it spicy?")

logger.info(f"Session: {session_id}")
logger.info(f"Messages in memory: {len(memory.get_messages())}")
logger.info(f"Context:\n{memory.get_context()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Recipe Agent Orchestration

# COMMAND ----------


class RecipeAgent:
    """Simple recipe agent with orchestration logic."""

    @staticmethod
    def _as_text(content: object) -> str:
        """Normalize model content blocks to readable plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        text_parts.append(item["content"])
            if text_parts:
                return "\n".join(text_parts)
        return str(content)

    def __init__(
        self, llm_endpoint: str, tool: RecipeSearchTool, memory: ConversationMemory
    ):
        self.llm_endpoint = llm_endpoint
        self.tool = tool
        self.memory = memory
        self.client = client
        self.system_prompt = (
            "You are a helpful recipe assistant. Your role is to help users find "
            "and adapt recipes.\n\n"
            "When a user asks for recipes, use the search_recipes tool to find "
            "relevant options.\n"
            "Always cite the recipe names when recommending them.\n"
            "Be helpful, concise, and practical."
        )

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool based on name and arguments."""
        if tool_name == "search_recipes":
            result = self.tool.execute(
                query=tool_args.get("query", ""),
                num_results=tool_args.get("num_results", 3),
            )

            if result["success"]:
                output_parts = [f"Found {result['count']} recipes:\n"]
                for i, recipe in enumerate(result["results"], 1):
                    output_parts.append(
                        f"{i}. **{recipe['recipe_name']}** ({recipe['source']})"
                    )
                    output_parts.append(f"   {recipe['text'][:200]}...\n")
                return "\n".join(output_parts)
            else:
                return f"Error searching recipes: {result.get('error', 'Unknown error')}"

        return f"Unknown tool: {tool_name}"

    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """Process user message with tool calling."""
        # Add user message to memory
        self.memory.add_message("user", user_message)

        # Classify intent
        intent = IntentClassifier.classify(user_message)
        logger.info(f"Intent: {intent}")

        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.memory.get_messages()

        # Determine if we should use tools
        tools_to_use = None
        if intent in ["recipe_search", "recipe_adapt"]:
            tools_to_use = [RECIPE_SEARCH_SPEC]

        # Step 1: Ask the model (with tools if applicable)
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            tools=tools_to_use,
            temperature=0.7,
            max_tokens=500,
        )

        assistant_message = response.choices[0].message

        # If no tool calls are requested, return direct answer.
        if not assistant_message.tool_calls:
            final_answer = self._as_text(assistant_message.content)
            self.memory.add_message("assistant", final_answer)
            return final_answer

        logger.info(f"Tool calls requested: {len(assistant_message.tool_calls)}")

        # Add assistant tool-call message
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        # Execute requested tools once
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            logger.info(f"Executing: {tool_name}({tool_args})")
            tool_result = self._execute_tool(tool_name, tool_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        # Step 2: Ask for final synthesized answer without allowing new tool calls
        final_response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        final_answer = self._as_text(final_response.choices[0].message.content)
        self.memory.add_message("assistant", final_answer)
        return final_answer


logger.info("✓ Recipe Agent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Demo: Multi-Turn Conversation

# COMMAND ----------

# Create fresh session for demo
demo_session_id = f"demo-{uuid4()}"
demo_memory = ConversationMemory(demo_session_id)
demo_agent = RecipeAgent(LLM_ENDPOINT, recipe_tool, demo_memory)

logger.info("=" * 80)
logger.info("DEMO: Multi-Turn Recipe Conversation")
logger.info("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 1: Search for vegetarian recipes

# COMMAND ----------

query1 = "Give me a quick vegetarian dinner idea"
logger.info(f"\n👤 User: {query1}")

response1 = demo_agent.chat(query1)
logger.info(f"\n🤖 Agent: {response1}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 2: Follow-up with memory

# COMMAND ----------

query2 = "Can you make one of those without garlic?"
logger.info(f"\n👤 User: {query2}")

response2 = demo_agent.chat(query2)
logger.info(f"\n🤖 Agent: {response2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conversation Summary

# COMMAND ----------

logger.info("\n" + "=" * 80)
logger.info("CONVERSATION SUMMARY")
logger.info("=" * 80)

all_messages = demo_memory.get_messages()
logger.info(f"Total messages: {len(all_messages)}")
logger.info(f"Session ID: {demo_session_id}")

logger.info("\nFull conversation:")
for i, msg in enumerate(all_messages, 1):
    role = msg["role"].upper()
    content = (
        msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
    )
    logger.info(f"{i}. [{role}] {content}")
