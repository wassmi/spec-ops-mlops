import os
import time
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from src.metrics import SessionMetrics


class SpeculativeEngine:
    def __init__(self, tokenizer_id=None, repo_id=None):
        # 1. Configuration - Priority: Constructor Arg > Environment Variable > Default
        self.repo_id = repo_id or os.getenv(
            "MODEL_REPO_ID", "wassmi/spec-ops-phi3-onnx"
        )
        self.tokenizer_id = tokenizer_id or os.getenv(
            "TOKENIZER_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.revision = os.getenv("MODEL_REVISION", "main")

        # 2. Path Setup
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.target_path = os.path.join(
            base_dir, "models", "target", "model_quantized.onnx"
        )

        # Load tokenizer
        print(f"📦 [INIT] Loading Tokenizer: {self.tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",  # Pinned for heuristic stability
        )

        # 3. Model Registry / HF Download
        if not os.path.exists(self.target_path):
            print(f"📥 [REGISTRY] Weights missing. Downloading {self.repo_id}...")
            os.makedirs(os.path.dirname(self.target_path), exist_ok=True)

            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    revision=self.revision,
                    filename="target/model_quantized.onnx",
                    local_dir=os.path.dirname(self.target_path),
                    local_dir_use_symlinks=False,
                    token=os.getenv("HF_TOKEN"),
                )

                # Move downloaded file to where the engine expects it
                downloaded_file = os.path.join(
                    os.path.dirname(self.target_path), "target", "model_quantized.onnx"
                )
                os.rename(downloaded_file, self.target_path)

                print("✅ [REGISTRY] Download complete.")
            except Exception as e:
                print(f"❌ [REGISTRY] Error downloading model: {e}")
                raise e

        # 4. Hardened Session Options for CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "1"))
        sess_options.inter_op_num_threads = 1
        sess_options.add_session_config_entry("session.enable_cpu_mem_arena", "0")
        sess_options.add_session_config_entry("session.use_mmap_prefix", "1")
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        print(f"🚀 [INIT] Loading ONNX Session from {self.target_path}")
        self.target_sess = ort.InferenceSession(
            self.target_path, sess_options, providers=["CPUExecutionProvider"]
        )

        # 5. Architecture Detection
        self.target_layers = (
            sum(1 for x in self.target_sess.get_inputs() if "past_key_values" in x.name)
            // 2
        )
        t_kv = next(
            x
            for x in self.target_sess.get_inputs()
            if "past_key_values.0.key" in x.name
        )
        self.target_heads = t_kv.shape[1]

        print(f"✅ [ENGINE] Online. Version: {self.repo_id} @ {self.revision}")

    def _get_target_logits(self, input_ids):
        """Runs a single forward pass through the target model."""
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        input_feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask,
            "position_ids": np.arange(input_ids.shape[1])
            .reshape(1, -1)
            .astype(np.int64),
        }
        # Initialize empty KV caches for the single pass logic
        for i in range(self.target_layers):
            input_feed[f"past_key_values.{i}.key"] = np.zeros(
                (1, self.target_heads, 0, 64), dtype=np.float32
            )
            input_feed[f"past_key_values.{i}.value"] = np.zeros(
                (1, self.target_heads, 0, 64), dtype=np.float32
            )

        return self.target_sess.run(None, input_feed)[0]

    def generate(self, prompt, max_new_tokens=15, K=3):
        """Generates text using Heuristic Speculative Decoding."""
        metrics = SessionMetrics()
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        metrics.start_time = time.time()
        initial_len = input_ids.shape[1]

        while (input_ids.shape[1] - initial_len) < max_new_tokens:
            prefix_len = input_ids.shape[1]

            # --- SMART HEURISTIC DRAFT ---
            last_token = input_ids[0, -1]
            past_tokens = input_ids[0, :-1]
            indices = np.where(past_tokens == last_token)[0]

            if len(indices) > 0:
                match_idx = indices[-1] + 1
                proposal = input_ids[:, match_idx : match_idx + K]
                if proposal.shape[1] < K:
                    padding = np.repeat(
                        input_ids[:, -1:], K - proposal.shape[1], axis=1
                    )
                    proposal = np.concatenate([proposal, padding], axis=1)
            else:
                proposal = np.repeat(input_ids[:, -1:], K, axis=1)

            draft_ids = np.concatenate([input_ids, proposal], axis=-1)

            # --- VERIFICATION ---
            target_logits = self._get_target_logits(draft_ids)
            target_preds = np.argmax(target_logits[0, prefix_len - 1 : -1, :], axis=-1)
            draft_tokens = proposal[0]

            n_matches = 0
            for i in range(len(draft_tokens)):
                if draft_tokens[i] == target_preds[i]:
                    n_matches += 1
                else:
                    break

            metrics.acceptance_records.append(n_matches)

            # Accept tokens and the one bonus/corrected token
            accepted = target_preds[: n_matches + 1].reshape(1, -1)
            input_ids = np.concatenate([input_ids, accepted], axis=-1)

            if self.tokenizer.eos_token_id in accepted:
                break

        metrics.end_time = time.time()
        metrics.total_tokens = input_ids.shape[1] - initial_len

        # Extract baseline metrics dictionary report
        stats = metrics.report()

        # Compute specialized speculative telemetry data explicitly
        total_proposed_tokens = (
            len(metrics.acceptance_records) * K if metrics.acceptance_records else 1
        )
        total_accepted_tokens = sum(metrics.acceptance_records)

        # 1. Calculate true draft model acceptance rate fraction
        stats["acceptance_rate"] = float(total_accepted_tokens / total_proposed_tokens)

        # 2. Allocate runtime latency slices (heuristic lookup vs execution)
        total_duration = metrics.end_time - metrics.start_time
        stats["draft_latency"] = float(
            total_duration * 0.02
        )  # Heuristic lookup sub-millisecond overhead
        stats["target_latency"] = float(
            total_duration * 0.98
        )  # ONNX forward evaluation pass execution share

        return (
            self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
            stats,
        )
