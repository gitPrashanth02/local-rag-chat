# Docker image that runs the API and the chat page together (chat page on port 7860).
# The models and the index for data/chatgpt.pdf are prepared at BUILD time,
# so the container starts quickly instead of downloading ~4 GB on every start.
FROM python:3.12-slim

# System libraries needed by OpenCV / MinerU
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user (id 1000), not root: safer if the server is ever compromised
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    ALLOW_UPLOADS=false
WORKDIR /home/user/app

# 1. Python packages. CPU-only PyTorch first: the default build includes GPU libraries (several GB)
#    that a CPU Space can't use. Copying requirements.txt alone lets Docker reuse this layer
#    when only the code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.12.1 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# 2. The code and the default document
COPY --chown=user . .

# 3. Build the index now. chatgpt.json is already parsed, so `touch` marks it newer than the PDF
#    and the slow MinerU step is skipped; chunking + embedding run (and download the embedding model).
RUN sed -i 's/\r$//' start.sh \
    && touch data/chatgpt.json \
    && python src/pipeline.py

# 4. Download the reranker and the LLM now, so startup doesn't have to
RUN python -c "import sys; sys.path.insert(0, 'src'); \
from retrieval import load_reranker; from generation import load_generator; \
load_reranker(); load_generator()"

EXPOSE 7860
CMD ["bash", "start.sh"]
