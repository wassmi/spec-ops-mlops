graph TD
    %% 1. The Development & Factory Phase
    subgraph "Phase 1: The Factory (CI/CD)"
        Code[Git Push: main.py + engine.py] --> GHA[GitHub Actions]
        GHA --> Build[Docker Build: spec-ops-serving]
        
        subgraph "Quality Gate"
            Build --> Script[scripts/ci_benchmark.py]
            Script --> Metric{TPJ > 1.0?}
            Metric -- Yes --> Push[Push to GHCR]
        end
    end

    %% 2. The Infrastructure Phase
    subgraph "Phase 2: The Hospital (Kubernetes)"
        Push --> K8s[K8s Cluster]
        K8s --> Service[k8s_service.yaml: Port 80]
        Service --> Pod[Pod: spec-ops-app]
    end

    %% 3. The Execution Phase (Based on your engine.py)
    subgraph "Phase 3: The Engine (Inference Logic)"
        User((User)) -->|Prompt| FastAPI[FastAPI: main.py]
        
        subgraph "Heuristic Speculative Loop"
            FastAPI --> Search[N-Gram Search: Look for last token in past_tokens]
            Search -->|Found Match| Guess[Propose next K tokens from History]
            Search -->|No Match| Static[Repeat last token K times]
            
            Guess --> Target[ONNX Target Model: Phi-3 Mini]
            Static --> Target
            
            Target -->|Parallel Logits| Verify{Verify Match?}
            Verify -->|Accept| Update[Update input_ids]
            Verify -->|Reject| Correct[Discard + Correct]
        end
    end

    %% 4. The Monitoring
    Update -->|specops_avg_jump| Prom[Prometheus/Grafana]