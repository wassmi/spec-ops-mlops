# Spec-Ops: Speculative Decoding API 🚀

A high-performance LLM inference engine using ONNX Runtime and Speculative Decoding to achieve **8.19 TPS** on standard CPU environments.



## 📊 Performance Benchmark
- **Environment:** GitHub Actions Runner (Standard Ubuntu)
- **Engine:** ONNX Runtime (CPU)
- **Optimization:** 4-bit Quantization
- **Result:** **8.19 Tokens Per Second** (Verified January 2026)

---
# 🗺️ Project Roadmap

## ✅ Phase 1: The Core Engine
- [x] Implement `SpeculativeEngine` in Python.
- [x] Integrate ONNX Runtime for CPU-bound inference.
- [x] Implement Draft-Target verification logic.

## ✅ Phase 2: Optimization & Containerization
- [x] 4-bit quantization of Phi-3 models.
- [x] Dockerization of the API (FastAPI).
- [x] Local performance profiling (Cursor/WSL).

## ✅ Phase 3: Automated Quality Gate (CI/CD)
- [x] **Linting:** Automated code style enforcement (Black/Flake8).
- [x] **Security:** Static analysis for vulnerabilities (Bandit).
- [x] **Resource Mgmt:** GitHub Action disk optimization (reclaimed 10GB).

## ✅ Phase 4: Systems Hardening & Stability (COMPLETED)
- [x] **Heuristic Pivot:** Implemented O(1) memory complexity drafting to survive 8GB RAM constraints.
- [x] **Runtime Optimization:** Disabled ONNX Arena and utilized `mmap` for zero-crash weight streaming.
- [x] **Lifecycle Management:** Hardened server startup to bypass Cloud/Codespace watchdog timeouts.
- [x] **Weight Decoupling:** Successfully transitioned to Hugging Face Model Hub registry.

## 🏗️ Phase 5: System Maturity & Observability (NEXT)
- [ ] **Heuristic Refinement:** Implement Bigram/N-Gram caching to maximize Speculative Acceptance Rate.
- [ ] **Observability:** Docker Compose integration for real-time Prometheus/Grafana monitoring.
- [ ] **Orchestration:** Kubernetes `deployment.yaml` with specific resource requests/limits.
- [ ] **Concurrency:** Stress testing with asynchronous request queuing.
---

## 🛠️ Infrastructure Stack
- **Runtime:** Python 3.11, ONNX Runtime
- **CI/CD:** GitHub Actions
- **Storage:** Git LFS (Migrating to Hugging Face)
- **Environment:** Docker (WSL / GitHub Runners)

## 🚦 Getting Started
1. **Clone:** `git clone ...`
2. **Setup:** `pip install -r requirements.txt`
3. **Run CI Locally:** `docker build -t spec-ops-api .`