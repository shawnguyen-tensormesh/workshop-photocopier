# Workshop RAG website (stock LlamaIndex). The served model's tokenizer is baked at build so
# chunking is 256-aligned even air-gapped (HF_HUB_OFFLINE=1 at runtime).
FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-fetch the tokenizer at build time (needs network here; runtime stays offline).
ARG MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
RUN HF_HUB_OFFLINE=0 python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL}')" || \
    echo "tokenizer not prefetched (provide /models at runtime)"

COPY rag_app.py server.py ./
COPY corpus ./corpus
COPY static ./static

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
