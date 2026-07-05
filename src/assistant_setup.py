import json
import os
import glob
import logging
import math
from typing import Tuple
from openai import OpenAI
import tiktoken

from env_utils import get_required_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants matching our design choices
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
VECTOR_STORE_NAME = "OptiSigns Help Center Knowledge Base"
ASSISTANT_NAME = "OptiBot"
RESPONSES_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.

Tone:
Helpful, factual, concise.

Rules:
1. Only answer using uploaded documentation. Do not assume or extrapolate outside the provided documents.
2. Maximum five bullet points per response.
3. If the answer cannot be fully answered or requires further reading, provide the document link.
4. Always cite up to three Article URLs retrieved from the document front matter or metadata."""


def estimate_local_chunks(file_paths: list[str], chunk_size: int, overlap: int) -> int:
    """Estimates chunk count locally using tiktoken matching OpenAI tokenization."""
    encoder = tiktoken.get_encoding("cl100k_base")
    total_chunks = 0
    
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            tokens = len(encoder.encode(content))
            if tokens <= chunk_size:
                total_chunks += 1
            else:
                # Calculate sliding window chunks
                stride = chunk_size - overlap
                chunks_for_file = math.ceil((tokens - overlap) / stride)
                total_chunks += max(1, chunks_for_file)
        except Exception as e:
            logger.warning(f"Failed to calculate tokens for {path}: {e}")
            
    return total_chunks


def setup_knowledge_base(data_dir: str = "./data") -> Tuple[str, str]:
    """Provisions Vector Store, uploads files, and creates OptiBot."""
    api_key = get_required_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # 1. Gather files
    markdown_files = glob.glob(os.path.join(data_dir, "*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found in {data_dir}. Run scraper first.")

    file_count = len(markdown_files)
    estimated_chunks = estimate_local_chunks(markdown_files, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    logger.info(f"Preparing ingestion for {file_count} files (~{estimated_chunks} estimated chunks).")

    # 2. Create Vector Store with static chunking strategy
    logger.info(f"Creating Vector Store: '{VECTOR_STORE_NAME}'...")
    vector_store = client.vector_stores.create(
        name=VECTOR_STORE_NAME,
        chunking_strategy={
            "type": "static",
            "static": {
                "max_chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS
            }
        }
    )
    logger.info(f"Vector Store created successfully. ID: {vector_store.id}")

    # 3. Upload files in batch
    logger.info("Uploading file streams and polling for completion...")
    file_streams = [open(path, "rb") for path in markdown_files]
    
    try:
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
    finally:
        for fs in file_streams:
            fs.close()

    # Log ingestion metrics
    counts = file_batch.file_counts
    logger.info("=" * 40)
    logger.info("INGESTION SUMMARY:")
    logger.info(f"  Status:           {file_batch.status}")
    logger.info(f"  Files Uploaded:   {counts.completed} completed, {counts.failed} failed")
    logger.info(f"  Estimated Chunks: {estimated_chunks} chunks generated")
    logger.info("=" * 40)

    if counts.failed > 0:
        logger.warning(f"Warning: {counts.failed} files failed to process.")

    # 4. Create or Update Assistant
    config = {
        "assistant_name": ASSISTANT_NAME,
        "model": RESPONSES_MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "vector_store_id": vector_store.id,
    }

    
    # Persist IDs locally for easy execution in Phase 3 / Chat testing
    os.makedirs("./config", exist_ok=True)
    with open("./config/optibot_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open("./config/vector_store_id.txt", "w", encoding="utf-8") as f:
        f.write(vector_store.id)

    logger.info("Config saved to ./config/optibot_config.json")
    return vector_store.id, config

def query_optibot(user_message: str, vector_store_id: str, client: OpenAI = None) -> str:
    client = client or OpenAI()
 
    response = client.responses.create(
        model=RESPONSES_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_message,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )
 
    return response.output_text

if __name__ == "__main__":
    try:
        vs_id, cfg = setup_knowledge_base()
        logger.info("Setup complete. Running a quick smoke-test query...")
        answer = query_optibot("What is OptiSigns?", vs_id)
        logger.info(f"Sample response:\n{answer}")
    except Exception as e:
        logger.critical(f"Setup failed: {e}", exc_info=True)