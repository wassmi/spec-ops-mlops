# Speculative-Quantized Inference Engine (SQIE)

SQIE is a high-performance LLM inference engine optimized for **extreme resource-constrained environments** (like 8GB RAM Cloud IDEs). It leverages **Speculative Decoding** and **INT8 Quantization** to bypass memory bandwidth bottlenecks, packaged in a production-ready Docker container with advanced memory-swap management.

---

## 🎯 Project Overview

In standard inference, LLMs are "Memory Bandwidth Bound"—the CPU spends more time fetching weights from RAM than performing calculations. This project solves that via:

* **Speculative Decoding:** A small "Draft" model (160M) predicts token sequences, which a larger "Target" model (1.1B) verifies in a single parallel pass.
* **INT8 Quantization:** Weights are compressed from FP32 to 8-bit integers via ONNX Runtime, reducing the memory footprint by ~70% and increasing execution speed on CPUs.
* **Virtual Memory Orchestration:** Utilizing a **4GB Physical / 12GB Swap** split to prevent OOM (Out of Memory) crashes while maintaining model stability in GitHub Codespaces.



---

## 🗺️ MLOps Roadmap

### ✅ Phase 1: Optimization & Conversion (Complete)
* **Model Selection:** Integrated `TinyLlama-1.1B-Chat` (Target) and `Llama-160M` (Draft).
* **ONNX Export:** Migrated models from PyTorch to ONNX format for hardware-agnostic acceleration.
* **INT8 Quantization:** Applied AVX2-optimized dynamic quantization to fit both models into ~2.5GB of combined space.

### ✅ Phase 2: Engine Development (Complete)
* **Draft-then-Verify Logic:** Implemented core speculative loop in `src/engine.py`.
* **Memory Arena Tuning:** Integrated `SessionOptions` to force ONNX to release memory back to the OS after every token, preventing "Memory Creep."
* **Tokenizer Synchronization:** Resolved vocabulary alignment issues by leveraging the `TinyLlama` tokenizer configuration.

### 🚀 Phase 3: Production & Deployment (Current)
* **Containerization:** Custom Docker architecture using `python:3.11-slim` for minimal overhead.
* **API Layer:** FastAPI production server with strict Pydantic response modeling and lazy-loading initialization.
* **Resource Guarding:** Deployed with Docker memory limits (`--memory="4g"`) to ensure IDE stability.

---

## 🛠️ Tech Stack

* **Models:** TinyLlama-1.1B (Target), Llama-160M (Draft)
* **Runtime:** ONNX Runtime (CPU Execution Provider)
* **API:** FastAPI & Uvicorn
* **Environment:** Docker (WSL / GitHub Codespaces)
* **Language:** Python 3.11+

---

## ⚡ Quick Start

### 1. Clean & Prepare
Before building, ensure the environment is clear of old artifacts:
```bash
docker rm -f speculative-api
docker system prune -af