# PolicyLens — API image.
#
# Built to start with no API key and no network: the index is constructed at build
# time from the PDF corpus, and the default embedding and generation backends are the
# dependency-free ones. A container that needed a key to answer its first request
# would be untestable, and the point of the offline path is that it is always there.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first, so edits to source do not invalidate the install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir \
      numpy scikit-learn faiss-cpu networkx pydantic \
      fastapi "uvicorn[standard]" langgraph langchain-core pypdf

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/corpus/ ./data/corpus/
COPY data/corpus_pdf/ ./data/corpus_pdf/
COPY data/eval/ ./data/eval/
COPY scripts/ ./scripts/

# Build the index into the image. First-request latency then reflects retrieval rather
# than a cold build, and the image fails loudly at build time if ingest is broken.
RUN python -m policylens.cli index --force

# Run as a non-root user. The index is already built, so nothing needs to write to
# the application directory at runtime.
RUN useradd --create-home --uid 10001 policylens \
    && chown -R policylens:policylens /app
USER policylens

EXPOSE 8000

# The readiness probe exercises the loaded index, not just the process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r.get('n_chunks',0)>0 else 1)"

CMD ["uvicorn", "policylens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
