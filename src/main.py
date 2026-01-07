import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine import SpeculativeEngine

# 1. This dictionary will hold our "warm" models in memory
ml_models = {}

# 2. LIFESPAN: This is the "Pro" way to handle setup/teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🤖 [STARTUP] Initializing Speculative Engine...")
    start_time = time.time()
    
    # Load once and store in our global dictionary
    ml_models["engine"] = SpeculativeEngine(
        target_path="models/target/model_quantized.onnx",
        draft_path="models/draft/model_quantized.onnx",
        tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    end_time = time.time()
    print(f"✅ [STARTUP] Models loaded in {end_time - start_time:.2f} seconds.")
    
    yield  # The API is now "live" and accepting requests
    
    # --- SHUTDOWN ---
    print("♻️ [SHUTDOWN] Clearing models from RAM...")
    ml_models.clear()

# 3. Initialize FastAPI with the lifespan handler
app = FastAPI(
    title="Spec-Ops LLM API",
    description="Optimized Speculative Inference Service",
    lifespan=lifespan
)

# 4. Request Schema (Using Pydantic for validation)
class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="The secret to MLOps is")
    max_tokens: int = Field(default=40, ge=1, le=200)
    k: int = Field(default=3, ge=1, le=5) # K tokens to speculate

@app.get("/health")
def health():
    # Only returns healthy if the engine is actually in memory
    status = "ready" if "engine" in ml_models else "loading"
    return {"status": status, "engine": "TinyLlama-1.1B-Speculative"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if "engine" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Record inference time for "Ops" monitoring
        start_inf = time.time()
        
        response_text = ml_models["engine"].generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            K=request.k
        )
        
        duration = time.time() - start_inf
        return {
            "response": response_text,
            "latency_seconds": round(duration, 4),
            "tokens_requested": request.max_tokens
        }
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 so it's accessible from outside the Docker container
    uvicorn.run(app, host="0.0.0.0", port=8000)