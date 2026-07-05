import os
import logging
from openai import OpenAI

from env_utils import get_required_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_chat_session(query: str):
    """Executes a single test query against OptiBot."""
    client = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))
    
    if not os.path.exists("./config/assistant_id.txt"):
        raise FileNotFoundError("Assistant ID not found. Run assistant_setup.py first.")
        
    with open("./config/assistant_id.txt", "r") as f:
        assistant_id = f.read().strip()

    logger.info(f"Creating thread and sending user query: '{query}'")
    thread = client.beta.threads.create()
    
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for msg in messages.data:
            if msg.role == "assistant":
                print("\n--- OPTIBOT RESPONSE ---")
                for content_block in msg.content:
                    if content_block.type == "text":
                        print(content_block.text.value)
                print("------------------------\n")
                break
    else:
        logger.error(f"Run failed with status: {run.status}")

if __name__ == "__main__":
    test_question = "How do I pair a new screen in OptiSigns? List the main steps."
    run_chat_session(test_question)