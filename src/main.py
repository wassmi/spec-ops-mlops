import logging
import time
import traceback
import gc
import psutil
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import SpeculativeEngine

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")

app = FastAPI(title="Spec-Ops Speculative Inference API", version="1.2.0")

# Global engine variable
engine = None


# --- Data Models ---
class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    temperature: float = 0.7
    k_draft: int = 3


class PredictionResponse(BaseModel):
    generated_text: str
    time_taken_ms: float
    tokens_per_second: float
    avg_tokens_per_jump: float
    status: str


# --- Endpoints ---


@app.get("/health")
async def health():
    """MLOps Step 3: Monitor RAM and Swap usage to debug performance."""
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    swap = psutil.swap_memory()

    return {
        "status": "online",
        "engine_loaded": engine is not None,
        "process_ram_mb": round(ram_usage_mb, 2),
        "system_swap_percent": swap.percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "message": (
            "Swap > 50% usually explains low TPS"
            if swap.percent > 50
            else "Resources healthy"
        ),
    }


@app.post("/generate", response_model=PredictionResponse)
async def generate(query: Query):
    global engine

    if engine is None:
        logger.info("🤖 [INIT] Loading ONNX Models into RAM (Target + Draft)...")
        try:
            start_load = time.time()
            engine = SpeculativeEngine(
                "/app/models/target/model_quantized.onnx",
                "/app/models/draft/model_quantized.onnx",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )
            logger.info(f"✅ [INIT] Models loaded in {time.time() - start_load:.2f}s")
        except Exception as e:
            logger.error(f"❌ [INIT] Failed: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Model engine failed to load.")

    try:
        safe_limit = min(query.max_new_tokens, 50)
        result_text, stats = engine.generate(
            query.prompt, max_new_tokens=safe_limit, K=query.k_draft
        )

        logger.info(
            f"📊 [METRICS] TPS: {stats['tokens_per_second']} | Jump Avg: {stats['avg_tokens_per_jump']}"
        )

        return {
            "generated_text": result_text,
            "time_taken_ms": stats["latency_ms"],
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_jump": stats["avg_tokens_per_jump"],
            "status": "success",
        }

    except Exception as e:
        logger.error(f"❌ [RUNTIME] Inference failed:\n{traceback.format_exc()}")
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
