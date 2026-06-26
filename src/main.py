import time
import os
import logging
import threading

# Added 'status' to the import list to fix the health check bug
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from src.engine import SpeculativeEngine
from src.metrics import SessionMetrics

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")

# --- PROMETHEUS METRICS ---
REQUEST_COUNT = Counter("specops_requests_total", "Total generation requests")
TOKEN_COUNT = Counter("specops_tokens_total", "Total tokens generated")
LATENCY_HIST = Histogram("specops_latency_seconds", "Total time spent generating text")
JUMP_GAUGE = Gauge("specops_avg_jump", "Average speculative jump per request")
ACCEPTANCE_RATE_GAUGE = Gauge(
    "specops_acceptance_rate", "Draft model token acceptance rate (0.0 - 1.0)"
)
DRAFT_LATENCY_HIST = Histogram(
    "specops_draft_latency_seconds", "Time spent executing the draft model"
)
TARGET_LATENCY_HIST = Histogram(
    "specops_target_latency_seconds", "Time spent verifying in the target model"
)

# --- ENGINE STATE ---
engine_instance = None


def load_engine_background():
    """Background task to download and initialize the engine."""
    global engine_instance
    try:
        logger.info("🤖 [BOOT] Starting background engine initialization...")
        engine_instance = SpeculativeEngine(
            tokenizer_id=os.environ.get(
                "MODEL_REPO", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
        )
        logger.info("✅ [BOOT] Engine is now ONLINE and ready for requests.")
    except Exception as e:
        logger.error(f"❌ [BOOT] Background initialization failed: {str(e)}")


# Start loading IMMEDIATELY in a separate thread
threading.Thread(target=load_engine_background, daemon=True).start()

# --- API SETUP ---
app = FastAPI(title="Spec-Ops API")


class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    k_draft: int = Field(default=3, ge=1, le=5)


@app.get("/health")
async def health(response: Response):
    """Endpoint for CI/CD and monitoring to check readiness."""
    if engine_instance is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "offline",
            "engine_ready": False,
            "mode": "Do not send traffic yet!",
        }

    return {
        "status": "online",
        "engine_ready": True,
        "mode": "heuristic-speculative",
    }


@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus to scrape."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate")
async def generate(query: Query):
    if engine_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Engine is still loading or downloading weights. Please wait.",
        )

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        result, stats = engine_instance.generate(
            query.prompt, max_new_tokens=query.max_new_tokens, K=query.k_draft
        )

        # Update Prometheus Metrics
        duration = time.time() - start_time
        LATENCY_HIST.observe(duration)
        TOKEN_COUNT.inc(stats["total_tokens"])
        JUMP_GAUGE.set(stats["avg_tokens_per_jump"])

        # Populate specialized speculative telemetry from engine stats
        if "acceptance_rate" in stats:
            ACCEPTANCE_RATE_GAUGE.set(stats["acceptance_rate"])
        if "draft_latency" in stats:
            DRAFT_LATENCY_HIST.observe(stats["draft_latency"])
        if "target_latency" in stats:
            TARGET_LATENCY_HIST.observe(stats["target_latency"])

        return {
            "generated_text": result,
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_jump": stats["avg_tokens_per_jump"],
            "latency_ms": stats["latency_ms"],
        }
    except Exception as e:
        logger.error(f"❌ [RUNTIME] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
