# Use an explicit, lightweight slim Python image
FROM python:3.12-slim

# Force Python outputs directly to terminal stream without buffering
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies separately to leverage Docker layer caching.
# build-essential is intentionally omitted: our dependencies ship
# prebuilt wheels for slim Python images. Add it back only if a
# `pip install` step fails while compiling from source.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source only. Config/data are intentionally NOT copied into the
# image -- they hold runtime state (sync_state.json, vector_store_id)
# that must persist across container runs, so they're mounted as a
# volume at `docker run` time instead. Baking them in would silently
# reset incremental sync state on every rebuild/redeploy.
COPY src/ ./src/

# Create the directories the app writes to at runtime, and a non-root
# user to own them / run the process as.
RUN mkdir -p /app/config /app/data && \
    useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Define execution target. Running the container performs one
# synchronization job and exits (exit code 0 on success).
CMD ["python", "src/main.py"]