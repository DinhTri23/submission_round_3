import os
# import glob
import json
import hashlib
import logging
import time
import functools
from typing import Dict, Any, Callable, Optional, Set
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from scraper import HelpCenterScraper
from utils import save_to_markdown, slugify
from env_utils import get_required_env

get_required_env("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

SYNC_STATE_PATH = "./config/sync_state.json"
DATA_DIR = "./data"
CONFIG_JSON_PATH = "./config/optibot_config.json"
LEGACY_VECTOR_STORE_PATH = "./config/vector_store_id.txt"

# Retry tuning
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2

# How many articles to pull per sync run. If your help center can have more
# than this many articles change in a single day, raise this or paginate.
ARTICLE_FETCH_LIMIT = 30


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

def with_retries(max_retries: int = MAX_RETRIES, base_backoff: float = BASE_BACKOFF_SECONDS):
    """
    Decorator that retries a function on transient OpenAI API errors
    (rate limits, connection errors, 5xx) with exponential backoff.
    Does NOT retry on non-retriable errors (e.g. bad request, auth failure).
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError as e:
                    last_exc = e
                    wait = base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"[{fn.__name__}] Rate limited (attempt {attempt}/{max_retries}). "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                except APIConnectionError as e:
                    last_exc = e
                    wait = base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"[{fn.__name__}] Connection error (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                except APIError as e:
                    # Only retry server-side (5xx) errors; anything else (4xx) is not transient.
                    status = getattr(e, "status_code", None)
                    if status is not None and status >= 500:
                        last_exc = e
                        wait = base_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            f"[{fn.__name__}] Server error {status} (attempt {attempt}/{max_retries}). "
                            f"Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise
            logger.error(f"[{fn.__name__}] Exhausted {max_retries} retries.")
            raise last_exc
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# State / config helpers
# --------------------------------------------------------------------------- #

def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a local file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_sync_state() -> Dict[str, Dict[str, Any]]:
    """Loads local synchronization ledger. Tolerates a corrupt/missing file."""
    if os.path.exists(SYNC_STATE_PATH):
        try:
            with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"{SYNC_STATE_PATH} is corrupt/unreadable. Starting from empty state.")
            backup_path = SYNC_STATE_PATH + ".corrupt"
            os.replace(SYNC_STATE_PATH, backup_path)
            logger.error(f"Corrupt state backed up to {backup_path}")
    return {}


def save_sync_state(state: Dict[str, Dict[str, Any]]) -> None:
    """
    Persists synchronization ledger to disk atomically (write to temp file,
    then rename) so a crash mid-write can't corrupt the ledger.
    """
    os.makedirs(os.path.dirname(SYNC_STATE_PATH), exist_ok=True)
    tmp_path = SYNC_STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, SYNC_STATE_PATH)


def get_vector_store_id() -> str:
    """
    Reads the vector store ID, preferring the current config format
    (optibot_config.json) and falling back to the legacy plain-text file.
    """
    if os.path.exists(CONFIG_JSON_PATH):
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        vector_store_id = config.get("vector_store_id")
        if vector_store_id:
            return vector_store_id

    if os.path.exists(LEGACY_VECTOR_STORE_PATH):
        with open(LEGACY_VECTOR_STORE_PATH, "r") as f:
            vector_store_id = f.read().strip()
        if vector_store_id:
            return vector_store_id

    raise FileNotFoundError(
        "Vector Store ID not found in optibot_config.json or vector_store_id.txt. "
        "Run assistant_setup.py first."
    )


# --------------------------------------------------------------------------- #
# OpenAI operations (wrapped with retries)
# --------------------------------------------------------------------------- #

@with_retries()
def delete_vector_store_file(client: OpenAI, vector_store_id: str, file_id: str) -> None:
    client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)


@with_retries()
def upload_vector_store_file(client: OpenAI, vector_store_id: str, filepath: str):
    """Uploads and polls until the file is fully processed (or failed)."""
    with open(filepath, "rb") as f:
        uploaded_file = client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store_id,
            file=f
        )
    if uploaded_file.status != "completed":
        raise RuntimeError(
            f"Vector store file ended in status '{uploaded_file.status}' "
            f"(expected 'completed') for {filepath}"
        )
    return uploaded_file


# --------------------------------------------------------------------------- #
# Core sync logic
# --------------------------------------------------------------------------- #

def sync_article(
    client: OpenAI,
    vector_store_id: str,
    article: Dict[str, Any],
    sync_state: Dict[str, Dict[str, Any]],
    stats: Dict[str, int],
) -> None:
    """Processes a single article: skip / update / new. Mutates sync_state and stats in place."""
    article_id = str(article.get("id"))
    title = article.get("title", "Untitled")
    api_updated_at = article.get("updated_at", "")
    slug = slugify(title)

    cached_record = sync_state.get(article_id)

    # Pass 1: cheap check using the API's own updated_at timestamp
    if cached_record and cached_record.get("updated_at") == api_updated_at:
        stats["skipped"] += 1
        logger.debug(f"Skipped (timestamp match): {title}")
        return

    # Render to markdown so we can hash actual content, not just metadata
    filepath = save_to_markdown(article, DATA_DIR)
    current_hash = compute_sha256(filepath)

    # Pass 2: timestamp moved but content is byte-identical -- just refresh the ledger
    if cached_record and cached_record.get("sha256") == current_hash:
        sync_state[article_id]["updated_at"] = api_updated_at
        stats["skipped"] += 1
        logger.info(f"Skipped (content hash identical): {title}")
        return

    logger.info(f"Processing change for article: '{title}'...")

    old_file_id: Optional[str] = cached_record.get("file_id") if cached_record else None

    # Remove the old vector store file FIRST, but don't drop it from the ledger
    # until the replacement upload actually succeeds -- otherwise a failed
    # upload leaves the article silently missing from search with no record of it.
    if old_file_id:
        try:
            logger.info(f"Removing outdated vector store file: {old_file_id}")
            delete_vector_store_file(client, vector_store_id, old_file_id)
        except Exception as e:
            logger.warning(f"Could not delete old file {old_file_id}: {e}")

    try:
        uploaded_file = upload_vector_store_file(client, vector_store_id, filepath)

        sync_state[article_id] = {
            "title": title,
            "slug": slug,
            "updated_at": api_updated_at,
            "sha256": current_hash,
            "file_id": uploaded_file.id,
        }

        if cached_record:
            stats["updated"] += 1
            logger.info(f"Successfully UPDATED article: {title}")
        else:
            stats["new"] += 1
            logger.info(f"Successfully INGESTED NEW article: {title}")

    except Exception as e:
        stats["failed"] += 1
        logger.error(f"Failed to sync article '{title}': {e}", exc_info=True)
        # Mark the ledger entry as broken rather than leaving stale/incorrect
        # data in place, so the next run treats it as needing a full re-sync.
        if article_id in sync_state:
            sync_state[article_id]["file_id"] = None
            sync_state[article_id]["sha256"] = None


def remove_orphaned_articles(
    client: OpenAI,
    vector_store_id: str,
    sync_state: Dict[str, Dict[str, Any]],
    live_article_ids: Set[str],
    stats: Dict[str, int],
) -> None:
    """
    Deletes vector store files (and ledger entries) for articles that exist
    in local state but were NOT returned by the scraper this run -- i.e.
    they were deleted, unpublished, or moved on the source help center.
    """
    orphaned_ids = [aid for aid in sync_state.keys() if aid not in live_article_ids]

    for article_id in orphaned_ids:
        record = sync_state[article_id]
        title = record.get("title", article_id)
        file_id = record.get("file_id")

        logger.info(f"Article no longer present upstream, removing: '{title}'")

        if file_id:
            try:
                delete_vector_store_file(client, vector_store_id, file_id)
            except Exception as e:
                logger.warning(f"Could not delete orphaned file {file_id} for '{title}': {e}")
                # Don't drop the ledger entry if we couldn't clean up the
                # remote file -- we'd lose track of it and orphan it forever.
                continue

        del sync_state[article_id]
        stats["removed"] += 1


def run_daily_sync() -> Dict[str, int]:
    """Executes the incremental daily sync job. Returns the run's stats dict."""
    logger.info("Starting OptiBot Daily Synchronization Job...")

    api_key = get_required_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    vector_store_id = get_vector_store_id()
    sync_state = load_sync_state()

    stats = {"new": 0, "updated": 0, "skipped": 0, "failed": 0, "removed": 0}

    try:
        scraper = HelpCenterScraper("https://support.optisigns.com")
        articles = scraper.fetch_articles(limit=ARTICLE_FETCH_LIMIT)

        if len(articles) == ARTICLE_FETCH_LIMIT:
            logger.warning(
                f"Fetched exactly the limit ({ARTICLE_FETCH_LIMIT}) articles -- "
                "there may be more changes than this run can see. Consider "
                "raising ARTICLE_FETCH_LIMIT or paginating."
            )

        live_article_ids = {str(a.get("id")) for a in articles}

        for article in articles:
            sync_article(client, vector_store_id, article, sync_state, stats)

        remove_orphaned_articles(client, vector_store_id, sync_state, live_article_ids, stats)

    finally:
        # Always persist whatever progress was made, even if the run
        # raised partway through -- otherwise a mid-run crash silently
        # discards every successful update from this execution.
        save_sync_state(sync_state)

    logger.info("=" * 40)
    logger.info("DAILY SYNC COMPLETED:")
    logger.info(f"  New Articles:      {stats['new']}")
    logger.info(f"  Updated Articles:  {stats['updated']}")
    logger.info(f"  Removed Articles:  {stats['removed']}")
    logger.info(f"  Skipped (No diff): {stats['skipped']}")
    logger.info(f"  Failed:            {stats['failed']}")
    logger.info("=" * 40)

    if stats["failed"] > 0:
        logger.warning(f"{stats['failed']} article(s) failed to sync -- check logs above for details.")

    return stats


if __name__ == "__main__":
    try:
        run_daily_sync()
    except Exception as e:
        logger.critical(f"Daily sync failed: {e}", exc_info=True)
        raise