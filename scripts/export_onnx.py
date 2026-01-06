import os
import json
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, AutoConfig

# Optimized for your project structure
MODELS = [
    ("target", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("draft", "jackaduma/Llama-160M"),
]

STATE_FILE = Path("models/.onnx_state.json")

def _load_state():
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def _is_done(state, folder, step):
    return state.get(folder, {}).get(step, False)

def _mark_done(state, folder, step):
    if folder not in state:
        state[folder] = {}
    state[folder][step] = True
    _save_state(state)

def export_and_quantize(folder_name, model_id):
    model_dir = Path(f"models/{folder_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    model_onnx = model_dir / "model.onnx"
    quantized_path = model_dir / "model_quantized.onnx"

    # ---- Export Step (Step 1) ----
    if model_onnx.exists() and _is_done(state, folder_name, "export"):
        print(f"[SKIP] {model_id} export: {model_onnx} exists and state says done.")
    else:
        print(f"[1/3] Downloading and Exporting {model_id} to ONNX...")
        model = ORTModelForCausalLM.from_pretrained(
            model_id, 
            export=True, 
            use_cache=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # Mark export step done
        _mark_done(state, folder_name, "export")
        print(f"[1/3] Export complete: {model_onnx}")

    # ---- Quantize Step (Step 2) ----
    if quantized_path.exists() and _is_done(state, folder_name, "quantize"):
        print(f"[SKIP] {model_id} quantize: {quantized_path} exists and state says done.")
    else:
        print(f"[2/3] Quantizing {model_id} (INT8/AVX2)...")
        quantizer = ORTQuantizer.from_pretrained(model_dir)
        dqconfig = AutoQuantizationConfig.avx2(is_static=False)
        quantizer.quantize(
            save_dir=model_dir,
            quantization_config=dqconfig,
        )
        # Mark quantize step done
        _mark_done(state, folder_name, "quantize")

    print(f"[3/3] SUCCESS: {model_id} is ready in {model_dir}")

if __name__ == "__main__":
    for folder, m_id in MODELS:
        export_and_quantize(folder, m_id)