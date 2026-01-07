import time
import gc
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.engine import SpeculativeEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")


# --- THE WATCHDOG HACK ---
# This keeps the CPU slightly 'noisy' so Codespaces doesn't think the process is idle.
def keep_alive():
    while True:
        time.sleep(2)
        # Just a tiny heartbeat in the logs
        pass


threading.Thread(target=keep_alive, daemon=True).start()

# --- INITIALIZE OUTSIDE LIFESPAN ---
# This ensures the engine is part of the global state before Uvicorn even starts its loop.
logger.info("🤖 [BOOT] Global Engine Initialization...")
engine_instance = SpeculativeEngine(tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

app = FastAPI(title="Spec-Ops API")


class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    k_draft: int = Field(default=3, ge=1, le=5)


@app.get("/health")
async def health():
    return {"status": "online", "engine_ready": engine_instance is not None}


@app.post("/generate")
async def generate(query: Query):
    try:
        result, stats = engine_instance.generate(
            query.prompt, max_new_tokens=query.max_new_tokens, K=query.k_draft
        )
        return {
            "generated_text": result,
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_jump": stats["avg_tokens_per_jump"],
            "latency_ms": stats["latency_ms"],
        }
    except Exception as e:
        logger.error(f"❌ [RUNTIME] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
