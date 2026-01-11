# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .

# Install build dependencies and generate wheels
RUN apt-get update && apt-get install -y gcc g++ \
    && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y curl jq && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY . .

# Performance Hardening for CPU
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1 

EXPOSE 8888

CMD ["python", "-m", "src.main"]