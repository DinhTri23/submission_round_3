import json
import os
import logging
from openai import OpenAI

from assistant_setup import RESPONSES_MODEL, SYSTEM_PROMPT
from env_utils import get_required_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_vector_store_id() -> str:
    """Load the vector store ID from the persisted config files."""
    candidates = [
        "./config/vector_store_id.txt",
        "./config/optibot_config.json",
        "./config/optibot_config.txt",
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        try:
            if path.endswith("vector_store_id.txt"):
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
                if value:
                    return value

            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            vector_store_id = config.get("vector_store_id")
            if vector_store_id:
                return vector_store_id
        except Exception as exc:
            logger.warning(f"Could not read vector store ID from {path}: {exc}")

    raise FileNotFoundError("Vector Store ID not found. Run assistant_setup.py first.")


def run_chat_session(query: str):
    """Executes a single test query against OptiBot using the vector store."""
    client = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))
    vector_store_id = load_vector_store_id()

    logger.info(f"Sending query to vector store {vector_store_id}: '{query}'")
    response = client.responses.create(
        model=RESPONSES_MODEL,
        instructions=SYSTEM_PROMPT,
        input=query,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )

    print("\n--- OPTIBOT RESPONSE ---")
    print(response.output_text)
    print("------------------------\n")

if __name__ == "__main__":
    test_question = "How do I pair a new screen in OptiSigns? List the main steps."
    run_chat_session(test_question)