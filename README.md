# Spec-Ops: Speculative Decoding API

A high-performance LLM inference engine using ONNX Runtime and Speculative Decoding to achieve 8+ TPS on CPU-only environments.

## 🚀 Quick Start
1. **Clone with LFS:** `git clone https://github.com/your-username/spec-ops-mlops.git`
2. **Build:** `docker build -t spec-ops-api .`
3. **Run:** `docker run -p 8000:8000 spec-ops-api`

## 📊 Performance Benchmarks (GitHub Cloud)
- **Engine:** ONNX Runtime (CPU)
- **Optimization:** 4-bit Quantization
- **Strategy:** Speculative Decoding (k=1)
- **Result:** **8.19 Tokens Per Second** (Verified via GitHub Actions)

## 🛠️ CI/CD Pipeline
Our automated pipeline ensures:
- **Linting:** Black (Code Style)
- **Security:** Bandit (Vulnerability Scanning)
- **Smoke Test:** Performance gate > 2.0 TPS
