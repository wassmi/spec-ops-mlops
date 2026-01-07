import onnxruntime as ort
import numpy as np
import gc
from transformers import AutoTokenizer

class SpeculativeEngine:
    def __init__(self, target_path, draft_path, tokenizer_id):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        
        # HARDENED SESSION OPTIONS
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = False 
        
        # This tells ONNX to release memory back to the OS after every request
        sess_options.add_session_config_entry("session.use_device_allocator_for_initialization", "1")
        
        self.target_sess = ort.InferenceSession(
            target_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        self.draft_sess = ort.InferenceSession(
            draft_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        
        # Auto-detect architecture
        self.target_layers = sum(1 for x in self.target_sess.get_inputs() if "past_key_values" in x.name) // 2
        self.draft_layers = sum(1 for x in self.draft_sess.get_inputs() if "past_key_values" in x.name) // 2
        
        t_kv = next(x for x in self.target_sess.get_inputs() if "past_key_values.0.key" in x.name)
        self.target_heads = t_kv.shape[1]
        
        d_kv = next(x for x in self.draft_sess.get_inputs() if "past_key_values.0.key" in x.name)
        self.draft_heads = d_kv.shape[1]
        
        print(f"✅ Engine Ready | Target: {self.target_layers}L/{self.target_heads}H | Draft: {self.draft_layers}L/{self.draft_heads}H")

    def _get_logits(self, session, input_ids, num_layers, num_heads):
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        model_inputs = [x.name for x in session.get_inputs()]
        
        input_feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask,
        }
        
        if "position_ids" in model_inputs:
            input_feed["position_ids"] = np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int64)
        if "use_cache_branch" in model_inputs:
            input_feed["use_cache_branch"] = np.array([False], dtype=bool)
        
        # KV Caches: We pass zeros because we aren't using the KV-cache branch for this implementation
        for i in range(num_layers):
            input_feed[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, 64), dtype=np.float32)
            input_feed[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, 64), dtype=np.float32)
        
        return session.run(None, input_feed)[0]

    def generate(self, prompt, max_new_tokens=15, K=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        
        try:
            for _ in range(max_new_tokens):
                prefix_len = input_ids.shape[1]
                draft_ids = input_ids.copy()
                
                # 1. Draft Proposal (K tokens)
                for _ in range(K):
                    logits = self._get_logits(self.draft_sess, draft_ids, self.draft_layers, self.draft_heads)
                    next_tok = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                    draft_ids = np.concatenate([draft_ids, next_tok], axis=-1)
                
                # 2. Target Verification
                target_logits = self._get_logits(self.target_sess, draft_ids, self.target_layers, self.target_heads)
                
                # Extract predictions for the indices we just drafted
                target_preds = np.argmax(target_logits[0, prefix_len-1:-1, :], axis=-1)
                draft_tokens = draft_ids[0, prefix_len:]
                
                # 3. Compare Draft vs Target
                n_matches = 0
                for i in range(len(draft_tokens)):
                    if draft_tokens[i] == target_preds[i]:
                        n_matches += 1
                    else:
                        break
                
                # 4. Update Sequence (Accept matches + the next target token)
                accepted = target_preds[:n_matches + 1].reshape(1, -1)
                input_ids = np.concatenate([input_ids, accepted], axis=-1)
                
                print(f"🚀 Jump: +{n_matches} | Seq: {input_ids.shape[1]}", flush=True)
                
                if self.tokenizer.eos_token_id in accepted:
                    break
            
            final_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return final_output

        finally:
            # CLEANUP: Crucial to prevent 502/OOM crashes
            if 'target_logits' in locals(): del target_logits
            if 'logits' in locals(): del logits
            gc.collect()