import logging
import time
import traceback
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import SpeculativeEngine

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")

app = FastAPI(
    title="Spec-Ops Speculative Inference API",
    version="1.0.0"
)

# Global engine variable
engine = None

# --- Data Models for FastAPI ---
class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    temperature: float = 0.7

class PredictionResponse(BaseModel):
    generated_text: str
    time_taken_ms: float
    tokens_per_second: float
    status: str

# --- Endpoints ---

@app.get("/health")
async def health():
    """Verify the API and Model status."""
    return {
        "status": "online",
        "engine_loaded": engine is not None
    }

@app.post("/generate", response_model=PredictionResponse)
async def generate(query: Query):
    global engine
    
    # 2. Lazy Loading Logic (Initializes on first request)
    if engine is None:
        logger.info("🤖 [INIT] Loading ONNX Models into RAM (Target + Draft)...")
        try:
            start_load = time.time()
            engine = SpeculativeEngine(
                "/app/models/target/model_quantized.onnx",
                "/app/models/draft/model_quantized.onnx",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ) 
            logger.info(f"✅ [INIT] Models loaded in {time.time() - start_load:.2f}s")
        except Exception as e:
            logger.error(f"❌ [INIT] Failed: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Model engine failed to load.")

    # 3. Inference Logic
    try:
        start_time = time.time()
        
        # Safety cap on tokens to prevent OOM
        safe_limit = min(query.max_new_tokens, 25)
        
        # Execute speculative generation
        result_text = engine.generate(query.prompt, max_new_tokens=safe_limit)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Calculate performance metrics
        word_count = len(result_text.split())
        est_tokens = max(word_count * 1.3, 1) # Rough estimation for LLM tokens
        tps = est_tokens / max((end_time - start_time), 0.001)

        # Return matching the PredictionResponse schema
        return {
            "generated_text": result_text,
            "time_taken_ms": round(duration_ms, 2),
            "tokens_per_second": round(tps, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ [RUNTIME] Inference failed:\n{traceback.format_exc()}")
        gc.collect() # Clean up what we can
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)