# Speculative-Quantized Inference Engine (SQIE)

SQIE is a high-performance LLM inference engine optimized for extreme resource-constrained environments (like 4GB/8GB RAM Cloud IDEs). I have engineered this to leverage Speculative Decoding and INT8 Quantization to bypass memory bandwidth bottlenecks, packaged in a production-ready Docker container with advanced memory-swap management.

🎯 Project Overview
In standard inference, LLMs are "Memory Bandwidth Bound"—the CPU spends more time fetching weights from RAM than performing calculations. I solved this via:

* **Speculative Decoding:** I implemented a small "Draft" model (160M) to predict token sequences, which a larger "Target" model (1.1B) verifies in parallel.
* **INT8 Quantization:** I compressed weights from FP32 to 8-bit integers via ONNX Runtime, reducing memory footprint by ~70% and increasing CPU execution speed.
* **Virtual Memory Orchestration:** I utilize a 4GB Physical / 8GB Swap split to prevent OOM (Out of Memory) crashes while maintaining model stability in GitHub Codespaces.

📊 My Performance Benchmarks
I am tracking the evolution of this engine using an observability `/health` endpoint and a custom benchmarking loop.

| Phase | Optimization | Throughput (TPS) | Acceptance Rate | CPU % | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Initial Engine | 2.09 | 0.15 | -- | ❌ Bottlenecked |
| **Phase 4a** | Vectorized Verification | 4.35 | 0.50 | 15.8% | ✅ 108% Gain |
| **Phase 4b** | Multi-threading (nproc=2) | **4.73** | **0.78** | **24.5%** | 🚀 Current |

> **Analysis:** By matching the engine's threading to the hardware's core count, I increased CPU utilization and achieved a more stable throughput. The primary focus is now improving the Draft model's acceptance rate.

🗺️ MLOps Roadmap
- [x] **Phase 1: Optimization & Conversion**
    - Integrated TinyLlama-1.1B-Chat (Target) and Llama-160M (Draft).
    - Migrated models from PyTorch to ONNX format.
    - Applied AVX2-optimized dynamic quantization to fit models into ~2.5GB combined space.
- [x] **Phase 2: Engine Development**
    - Implemented core speculative loop logic.
    - Integrated `SessionOptions` for aggressive memory release back to the OS.
- [x] **Phase 3: Production & Deployment**
    - Containerized using `python:3.11-slim`.
    - Built FastAPI production server with Pydantic modeling.
    - Deployed with strict Docker memory limits (`--memory="4g"`).
- [ ] **Phase 4: Advanced Optimization (Current)**
    - [x] Implement Vectorized Verification (Draft-then-Verify).
    - [x] Hardware-aligned Multi-threading.
    - [ ] KV-Caching Integration for $O(1)$ token generation.
    - [ ] K-value hyperparameter tuning.

🛠️ Tech Stack
- **Models:** TinyLlama-1.1B (Target), Llama-160M (Draft)
- **Runtime:** ONNX Runtime (CPU Execution Provider)
- **API:** FastAPI & Uvicorn
- **Environment:** Docker (WSL / GitHub Codespaces / Cursor)
- **Language:** Python 3.11+

⚡ Quick Start
### 1. Clean & Prepare
Before building, I ensure the environment is clear of old artifacts:
```bash
docker rm -f speculative-api
docker system prune -af
docker build -t spec-ops-api .
docker run -d -p 8000:8000 --name speculative-api --memory="4g" spec-ops-api
```



## 🏆 Final Optimization Discovery
After hyperparameter tuning on a 2-core (nproc=2) environment, I have identified the optimal configuration for this engine:

| Metric | K=3 (Aggressive) | K=1 (Balanced) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 2.35 TPS | **3.13 TPS** | **+33%** |
| **CPU Load** | 54.1% | ~48.0% | 🟢 More Efficient |

**Key Finding:** For the TinyLlama-1.1B (Target) and Llama-160M (Draft) pair on dual-core hardware, a smaller speculation window (=1$) provides the most stable and highest throughput by minimizing "discarded" token computations.
