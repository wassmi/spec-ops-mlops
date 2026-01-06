# Speculative-Quantized Inference Engine (SQIE)

**SQIE** is a high-performance LLM inference engine designed to deliver low-latency text generation on consumer-grade hardware. It leverages **Speculative Decoding** and **INT8 Quantization** to bypass the memory bandwidth bottlenecks typical of standard CPU-based LLM deployments.

## 🎯 Project Overview
In standard inference, LLMs are "Memory Bandwidth Bound"—the CPU spends more time fetching weights from RAM than performing calculations. This project solves that by:
1. **Speculative Decoding:** Using a small "Draft" model (160M) to predict token sequences, which a larger "Target" model (1.1B) verifies in a single parallel pass.
2. **INT8 Quantization:** Compressing weights from FP32 to 8-bit integers using ONNX Runtime, reducing memory pressure by ~70% and increasing execution speed on CPUs.

## 🗺️ MLOps Roadmap

### Phase 1: Optimization & Framework Conversion (Current)
- [x] **Model Selection:** Integrated `TinyLlama-1.1B-Chat` (Target) and `Llama-160M` (Draft).
- [x] **ONNX Export:** Migrated models from PyTorch to ONNX format for hardware-agnostic acceleration.
- [x] **INT8 Quantization:** Implemented AVX2-optimized dynamic quantization via Hugging Face Optimum.
- [ ] **Artifact Management:** Establishing a Cloud-to-Local bridge via GitHub/Colab for optimized model weights.

### Phase 2: Speculative Engine Development
- [ ] **Inference Loop:** Develop the core "Draft-then-Verify" sampling logic.
- [ ] **Heuristic Tuning:** Optimize the $k$ (look-ahead) parameter to maximize acceptance rates.
- [ ] **Tokenizer Sync:** Align vocabulary spaces across heterogeneous model architectures.

### Phase 3: Benchmarking & Production
- [ ] **Latency Profiling:** Compare Tokens Per Second (TPS) between standard and speculative modes.
- [ ] **Containerization:** Package the engine into a lightweight **Docker** image.
- [ ] **API Layer:** Deploy a **FastAPI** endpoint for real-time model interaction.

## 🛠️ Tech Stack
* **Models:** TinyLlama-1.1B, Llama-160M
* **Optimization:** Hugging Face Optimum, ONNX Runtime
* **Infrastructure:** Google Colab (Compute), WSL/Cursor (Dev), GitHub (CI/CD)
* **Backend:** Python 3.12+

