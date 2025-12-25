# AI Infrastructure System Design & Hands-On Scenarios Guide
## Part 2: System Design Questions, Troubleshooting, and Behavioral Interview Prep

**Companion to Main Interview Prep Guide**

---

## TABLE OF CONTENTS

1. [System Design Questions](#system-design)
2. [Hands-On Scenarios & Troubleshooting](#hands-on)
3. [Behavioral Questions for Infrastructure Roles](#behavioral)
4. [Interview Day Strategy](#interview-strategy)

---

## SYSTEM DESIGN QUESTIONS {#system-design}

### Approach Framework for System Design Interviews

**Structure Every System Design Answer Using This Framework**:

1. **Clarify Requirements (5 minutes)**
   - Functional requirements: What must the system do?
   - Non-functional requirements: Scale, latency, availability, cost constraints
   - Ask clarifying questions: Users, QPS, data volume, SLAs

2. **Back-of-Envelope Calculations (5 minutes)**
   - Estimate: Storage, bandwidth, compute, costs
   - Identify bottlenecks early

3. **High-Level Design (10 minutes)**
   - Draw architecture diagram
   - Identify major components
   - Explain data flow

4. **Deep Dive (15-20 minutes)**
   - Interviewer will focus on specific areas
   - Be ready to dive deep on any component
   - Discuss trade-offs, alternatives

5. **Discuss Trade-offs & Bottlenecks (5 minutes)**
   - What could go wrong?
   - How to scale further?
   - Cost optimization opportunities

---

### System Design Question 1: Scalable Model Serving Infrastructure

**Q**: Design a model serving platform that can serve 1 million predictions per second across 100 different ML models. Models range from small (10MB) to large (10GB). Latency SLA is p99 < 100ms. How would you design this system?

**Answer Framework**:

#### Step 1: Clarify Requirements

**Functional Requirements**:
- Serve predictions from 100 different models simultaneously
- Support batch and real-time inference
- Handle model updates without downtime
- Provide metrics and monitoring

**Non-Functional Requirements**:
- Throughput: 1M predictions/second
- Latency: p99 < 100ms
- Availability: 99.9% uptime
- Model sizes: 10MB to 10GB
- Support A/B testing and canary deployments

**Clarifying Questions**:
- Q: "Are these models CPU or GPU-based?"
  - A: Mix - 70% can run on CPU, 30% require GPU
- Q: "What's the traffic pattern - steady or spiky?"
  - A: Spiky with 3x variance (peak hours vs off-peak)
- Q: "Do models share dependencies or are they independent?"
  - A: Independent models with different frameworks (PyTorch, TF, ONNX)
- Q: "What's the model update frequency?"
  - A: 10-20 model updates per day across all models

#### Step 2: Capacity Estimation

**Throughput Requirements**:
- Total: 1M QPS
- Per model average: 1M / 100 = 10K QPS per model
- Peak (3x): 30K QPS per model

**Compute Requirements**:

For CPU models (70 models):
- Assume 10ms inference time on CPU
- Throughput per core: 1000ms / 10ms = 100 RPS
- Needed cores per model: 30K RPS / 100 = 300 cores
- Total CPU cores: 70 models * 300 = 21,000 cores
- With auto-scaling headroom (1.3x): ~27,000 cores

For GPU models (30 models):
- Assume 5ms inference time on GPU (batch=8)
- Throughput per GPU: 1000ms / 5ms * 8 = 1,600 RPS
- Needed GPUs per model: 30K RPS / 1,600 = 19 GPUs
- Total GPUs: 30 models * 19 = 570 GPUs
- With headroom: ~750 GPUs

**Storage Requirements**:
- Small models (10MB): 70 models * 10MB = 700MB
- Large models (10GB): 30 models * 10GB = 300GB
- Total: ~301GB for all models
- With versions (keep last 3): ~900GB
- Cached in memory: ~1TB RAM across fleet

**Network Bandwidth**:
- Assume 1KB per request/response
- 1M QPS * 1KB = 1 GB/s
- With overhead: ~2 GB/s total bandwidth

**Cost Estimation** (AWS):
- CPU: 27,000 cores ≈ 1,700x m5.16xlarge @ $2.46/hr = $4,182/hr = $100K/day
- GPU: 750 GPUs ≈ 94x p3.16xlarge (8 GPUs each) @ $24.48/hr = $2,300/hr = $55K/day
- Total: ~$155K/day = $4.7M/month (can reduce 60-70% with reserved + spot)

#### Step 3: High-Level Architecture

```
                                    ┌──────────────────┐
                                    │   API Gateway    │
                                    │  (Rate Limiting, │
[External Clients]                  │   Routing, Auth) │
        │                           └────────┬─────────┘
        │                                    │
        └────────────────────┬───────────────┘
                             │
                    ┌────────▼─────────┐
                    │   Load Balancer  │
                    │  (ALB/CloudFront)│
                    └────────┬─────────┘
                             │
           ┌─────────────────┴──────────────────┐
           │                                    │
     ┌─────▼──────┐                      ┌──────▼─────┐
     │  Model     │                      │   Model    │
     │  Routing   │                      │   Routing  │
     │  Service   │                      │   Service  │
     └─────┬──────┘                      └──────┬─────┘
           │                                    │
           │     ┌──────────────────────────────┤
           │     │                              │
     ┌─────▼─────▼───┐                   ┌──────▼────────┐
     │  CPU Model    │                   │  GPU Model    │
     │  Serving Pool │                   │  Serving Pool │
     │  (Triton/     │                   │  (vLLM/       │
     │   TorchServe) │                   │   Triton)     │
     └───────┬───────┘                   └───────┬───────┘
             │                                   │
             └─────────┬─────────────────────────┘
                       │
            ┌──────────▼───────────┐
            │  Model Registry      │
            │  (MLflow + S3)       │
            └──────────┬───────────┘
                       │
            ┌──────────▼───────────┐
            │  Monitoring Stack    │
            │  (Prometheus+Grafana)│
            └──────────────────────┘
```

#### Step 4: Detailed Component Design

**Component 1: Model Serving Layer**

Use **NVIDIA Triton Inference Server** as primary serving framework:

**Why Triton?**
- Multi-framework support (PyTorch, TF, ONNX, TensorRT)
- Dynamic batching for throughput optimization
- Model versioning and A/B testing built-in
- GPU optimization with TensorRT integration
- gRPC and HTTP/REST APIs

**Triton Configuration**:
```yaml
# Triton model repository structure
models/
├── model_a/
│   ├── 1/           # Version 1
│   │   └── model.pt
│   ├── 2/           # Version 2 (canary)
│   │   └── model.pt
│   └── config.pbtxt
├── model_b/
│   └── ...

# config.pbtxt for efficient serving
name: \"model_a\"
platform: \"pytorch_libtorch\"
max_batch_size: 32
input [
  {
    name: \"input\"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]
output [
  {
    name: \"output\"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 4  # 4 instances per GPU
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 1000  # 1ms max wait
}
optimization {
  cuda {
    graphs: true  # CUDA graph optimization
  }
}
```

**Deployment on Kubernetes**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-serving-gpu
  labels:
    app: triton-serving
    model-type: gpu
spec:
  replicas: 100  # Scale based on GPU models
  selector:
    matchLabels:
      app: triton-serving
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.10-py3
        args:
        - tritonserver
        - --model-repository=s3://models-bucket/models
        - --strict-model-config=false
        - --log-verbose=0
        - --metrics-port=8002
        ports:
        - containerPort: 8000  # HTTP
        - containerPort: 8001  # gRPC
        - containerPort: 8002  # Metrics
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 8
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
---
# Service for load balancing
apiVersion: v1
kind: Service
metadata:
  name: triton-serving-gpu
spec:
  selector:
    app: triton-serving
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: grpc
    port: 8001
    targetPort: 8001
  type: ClusterIP
---
# HPA for autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-serving-gpu
  minReplicas: 50
  maxReplicas: 200
  metrics:
  - type: Pods
    pods:
      metric:
        name: nv_inference_request_success
      target:
        type: AverageValue
        averageValue: \"1000\"  # 1000 RPS per pod
```

**Component 2: Model Routing Service**

Intelligent routing layer to direct requests to appropriate models:

```python
# FastAPI-based routing service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import hashlib
from typing import Dict, Optional

app = FastAPI()

# Model registry (loaded from database/config)
MODEL_REGISTRY = {
    \"model_a\": {
        \"endpoint\": \"http://triton-serving-gpu:8000/v2/models/model_a/infer\",
        \"version\": \"2\",
        \"canary_version\": \"3\",
        \"canary_percentage\": 10,  # 10% traffic to canary
        \"hardware\": \"gpu\"
    },
    \"model_b\": {
        \"endpoint\": \"http://triton-serving-cpu:8000/v2/models/model_b/infer\",
        \"version\": \"1\",
        \"hardware\": \"cpu\"
    },
    # ... 98 more models
}

class PredictionRequest(BaseModel):
    model_name: str
    input_data: Dict
    user_id: Optional[str] = None

class PredictionResponse(BaseModel):
    model_name: str
    version: str
    prediction: Dict
    latency_ms: float

@app.post(\"/predict\", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start_time = time.time()

    # Lookup model configuration
    if request.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=\"Model not found\")

    model_config = MODEL_REGISTRY[request.model_name]

    # Determine version (canary routing)
    version = model_config[\"version\"]
    if model_config.get(\"canary_version\") and request.user_id:
        # Hash-based canary routing (consistent per user)
        user_hash = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
        if (user_hash % 100) < model_config[\"canary_percentage\"]:
            version = model_config[\"canary_version\"]

    # Call Triton backend
    endpoint = model_config[\"endpoint\"]
    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            json={
                \"inputs\": request.input_data,
                \"outputs\": [{\"name\": \"output\"}]
            },
            timeout=0.5  # 500ms timeout
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=\"Model inference failed\")

    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        model_name=request.model_name,
        version=version,
        prediction=response.json(),
        latency_ms=latency_ms
    )

@app.get(\"/health\")
async def health_check():
    return {\"status\": \"healthy\"}

# Deployment: 50 replicas of this routing service
# Each handles ~20K RPS = 1M total
```

**Component 3: Model Registry & Versioning**

Use **MLflow Model Registry** backed by **S3**:

```python
# Model deployment pipeline
import mlflow
from mlflow.tracking import MlflowClient

def deploy_model_to_production(model_name, version, canary_percentage=0):
    \"\"\"
    Deploy model version to production with optional canary
    \"\"\"
    client = MlflowClient()

    # 1. Promote model version to production in MLflow
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=\"Production\"
    )

    # 2. Download model artifacts to S3 model repository
    model_uri = f\"models:/{model_name}/{version}\"
    local_path = mlflow.artifacts.download_artifacts(model_uri)

    # 3. Upload to Triton model repository
    s3_path = f\"s3://models-bucket/models/{model_name}/{version}/\"
    subprocess.run([\"aws\", \"s3\", \"sync\", local_path, s3_path])

    # 4. Update model routing configuration
    update_model_config(
        model_name=model_name,
        version=version,
        canary_percentage=canary_percentage
    )

    # 5. Trigger Triton to reload models
    # Triton watches S3 and auto-reloads
    print(f\"Model {model_name} version {version} deployed to production\")

def update_model_config(model_name, version, canary_percentage):
    \"\"\"Update routing service configuration\"\"\"
    # Update database/ConfigMap with new version
    # Routing service polls this config every 10s
    pass
```

**Component 4: Monitoring & Observability**

```yaml
# Prometheus scrape configuration
scrape_configs:
- job_name: 'triton-serving'
  kubernetes_sd_configs:
  - role: pod
    namespaces:
      names:
      - model-serving
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    action: keep
    regex: triton-serving
  - source_labels: [__address__]
    target_label: __address__
    replacement: $1:8002  # Triton metrics port

# Key metrics to monitor
# 1. Request rate per model
sum by (model) (rate(nv_inference_request_success[1m]))

# 2. Latency p99 per model
histogram_quantile(0.99,
  sum by (model, le) (rate(nv_inference_request_duration_us_bucket[5m]))
) / 1000  # Convert to ms

# 3. GPU utilization
avg by (pod) (DCGM_FI_DEV_GPU_UTIL)

# 4. Queue time (dynamic batching)
avg by (model) (nv_inference_queue_duration_us) / 1000

# 5. Error rate
sum by (model) (rate(nv_inference_request_failure[1m]))

# Alerts
groups:
- name: model_serving
  rules:
  - alert: HighLatency
    expr: |
      histogram_quantile(0.99,
        sum by (model, le) (rate(nv_inference_request_duration_us_bucket[5m]))
      ) > 100000  # > 100ms
    for: 5m
    annotations:
      summary: \"Model {{ $labels.model }} latency above SLA\"

  - alert: HighErrorRate
    expr: |
      sum by (model) (rate(nv_inference_request_failure[5m])) /
      sum by (model) (rate(nv_inference_request_success[5m])) > 0.01
    for: 2m
    annotations:
      summary: \"Model {{ $labels.model }} error rate > 1%\"
```

#### Step 5: Advanced Considerations

**A/B Testing & Canary Deployments**:
```python
# Enhanced canary routing with metrics collection
class CanaryRouter:
    def route_request(self, model_name, user_id, input_data):
        model_config = MODEL_REGISTRY[model_name]

        # Determine version
        version = self.select_version(model_config, user_id)

        # Record routing decision for analysis
        self.record_routing(model_name, version, user_id)

        # Make prediction
        prediction = self.predict(model_config, version, input_data)

        # Record metrics
        self.record_metrics(model_name, version, prediction)

        return prediction

    def select_version(self, config, user_id):
        \"\"\"
        Canary routing with circuit breaker
        \"\"\"
        if not config.get(\"canary_version\"):
            return config[\"version\"]

        # Check canary health
        canary_error_rate = self.get_error_rate(
            config[\"model_name\"],
            config[\"canary_version\"]
        )

        # Circuit breaker: disable canary if error rate > 5%
        if canary_error_rate > 0.05:
            return config[\"version\"]  # Fallback to stable

        # Hash-based routing
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (user_hash % 100) < config[\"canary_percentage\"]:
            return config[\"canary_version\"]

        return config[\"version\"]
```

**Auto-Scaling Strategy**:
- **Pod-Level**: HPA scales Triton pods based on RPS (target: 1000 RPS/pod)
- **Cluster-Level**: Cluster Autoscaler adds GPU nodes when pods pending
- **Predictive Scaling**: Pre-scale before known traffic spikes
```python
# Predictive scaling based on historical patterns
import pandas as pd
from datetime import datetime, timedelta

def get_predicted_load(timestamp):
    \"\"\"Predict load 15 minutes ahead\"\"\"
    historical_data = load_historical_metrics()

    # Same day of week, same hour in past 4 weeks
    similar_timestamps = [
        timestamp - timedelta(weeks=w, minutes=15)
        for w in range(1, 5)
    ]

    avg_load = historical_data[
        historical_data['timestamp'].isin(similar_timestamps)
    ]['qps'].mean()

    return avg_load * 1.2  # 20% buffer

# Autoscaler triggers scale-up 10 minutes before predicted spike
```

**Cost Optimization**:
1. **Spot Instances**: Use for 40% of GPU capacity (non-critical models)
2. **Right-Sizing**: CPU models on c6i instances, GPU models on g5
3. **Reserved Instances**: 1-year commitment for baseline capacity (60% savings)
4. **Model Optimization**: TensorRT quantization (int8) for 2x throughput
5. **Batching**: Dynamic batching increases GPU utilization from 40% to 85%

**Estimated Cost After Optimization**:
- Original: $4.7M/month
- With reserved (60% baseline): $1.4M
- With spot (40% capacity): $1.2M
- With optimizations (batching, quantization): $900K
- **Final: ~$900K/month** (81% reduction)

#### Step 6: Failure Scenarios & Handling

**Scenario 1: GPU Node Failure**
- Detection: Liveness probe fails, pod marked unhealthy
- Response: K8s reschedules pod on healthy node
- Impact: Temporary capacity reduction, HPA scales up
- Recovery Time: 2-3 minutes (pod startup + model loading)

**Scenario 2: Model Loading Failure**
- Detection: Readiness probe fails, pod not added to service
- Response: Alert triggers, ops team investigates
- Fallback: Traffic stays on previous version
- Prevention: Validate models in staging before production

**Scenario 3: Traffic Spike Beyond Capacity**
- Detection: Latency p99 exceeds 100ms
- Response: Auto-scaler adds pods/nodes
- Mitigation: API gateway rate limiting prevents overload
- Temporary: Serve cached predictions for non-critical models

**Scenario 4: Model Registry (S3) Outage**
- Impact: Can't load new models, existing running pods unaffected
- Response: Triton serves from cached models in pod memory
- Duration: Models cached for 24 hours, sufficient for S3 restoration
- Prevention: Multi-region S3 replication

#### Summary: Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Serving Framework | NVIDIA Triton | Multi-framework, dynamic batching, GPU optimized |
| Orchestration | Kubernetes on EKS | Cloud-agnostic, rich ecosystem, auto-scaling |
| Model Registry | MLflow + S3 | Versioning, lineage, scalable storage |
| Routing | Custom FastAPI service | Fine-grained canary control, A/B testing |
| Monitoring | Prometheus + Grafana | Industry standard, Triton native support |
| Auto-Scaling | HPA + Cluster Autoscaler | Reactive scaling, handles spikes |
| Cost Optimization | Reserved + Spot + Batching | 81% cost reduction while meeting SLAs |

**Key Metrics Achieved**:
- Throughput: 1M+ QPS ✓
- Latency: p99 < 100ms ✓
- Availability: 99.9% ✓
- Cost: $900K/month (vs $4.7M naive approach) ✓

---

### System Design Question 2: Building a Multi-Tenant ML Training Platform

**Q**: Design a platform where 500 data scientists across 50 teams can submit ML training jobs. Jobs range from small experiments (1 GPU, 1 hour) to large-scale training (100 GPUs, 1 week). Design for cost efficiency, fair resource allocation, and ease of use.

**Answer Framework**:

#### Step 1: Requirements Clarification

**Functional Requirements**:
- Job submission interface (CLI, SDK, UI)
- Support distributed training (PyTorch DDP, Horovod, DeepSpeed)
- Automatic checkpointing and resumption
- Experiment tracking and artifact management
- Resource quotas per team
- Priority queues (production > research > experiments)

**Non-Functional Requirements**:
- Users: 500 data scientists, 50 teams
- Concurrent jobs: 100-200 jobs running simultaneously
- Job sizes: 1 GPU (small) to 100 GPUs (large)
- Duration: 1 hour to 1 week
- GPU types: Mix of V100, A100, H100
- Cost target: Maximize utilization (>80%) while staying under budget

**Clarifying Questions**:
- Q: \"Do teams have dedicated budgets?\"
  - A: Yes, each team has monthly GPU-hour quota
- Q: \"What frameworks need to be supported?\"
  - A: PyTorch (primary), TensorFlow (20%), JAX (10%)
- Q: \"On-premise or cloud?\"
  - A: Cloud (AWS) with some on-premise GPU clusters

#### Step 2: Capacity Planning

**Resource Estimation**:
- Total GPU inventory: 1000 GPUs
  - 500x V100 (16GB)
  - 400x A100 (40GB)
  - 100x A100 (80GB)

**Job Distribution** (based on historical patterns):
- Small jobs (1-4 GPUs): 60% of jobs, 20% of GPU-hours
- Medium jobs (8-16 GPUs): 30% of jobs, 30% of GPU-hours
- Large jobs (32-100 GPUs): 10% of jobs, 50% of GPU-hours

**Utilization Target**:
- Peak hours (9am-6pm): 95% utilization
- Off-peak: 60% utilization (experiments run overnight)
- Average: 80% utilization

**Cost Analysis**:
- A100 on-demand: $32/hr per instance (8 GPUs) = $4/GPU-hr
- A100 spot: ~$10/hr per instance = $1.25/GPU-hr (70% savings)
- Reserved instances: $2/GPU-hr (50% savings for 1-year commit)

**Strategy**:
- 50% capacity on reserved instances (baseline)
- 40% on spot instances (interruptible training)
- 10% on-demand (critical production jobs)

#### Step 3: High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interfaces                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐             │
│   │ CLI Tool │    │ Python   │    │ Web UI       │             │
│   │          │    │ SDK      │    │ (JupyterHub) │             │
│   └────┬─────┘    └────┬─────┘    └──────┬───────┘             │
└────────┼───────────────┼─────────────────┼──────────────────────┘
         │               │                 │
         └───────────────┴─────────────────┘
                         │
              ┌──────────▼───────────┐
              │  API Gateway         │
              │  (Authentication,    │
              │   Quota Check)       │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Job Scheduler       │
              │  (Volcano/YuniKorn)  │
              │  - Fair-share        │
              │  - Priority queues   │
              │  - Bin packing       │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
  ┌──────▼────────┐             ┌────────▼──────┐
  │ Kubernetes    │             │ On-Premise    │
  │ (EKS)         │             │ GPU Cluster   │
  │               │             │               │
  │ GPU Nodes:    │             │ DGX Servers   │
  │ - p3 (V100)   │             │               │
  │ - p4d (A100)  │             │               │
  └───────┬───────┘             └───────┬───────┘
          │                             │
          └─────────────┬───────────────┘
                        │
           ┌────────────▼──────────────┐
           │  Shared Services          │
           │  - MLflow (experiments)   │
           │  - S3 (artifacts)         │
           │  - Prometheus (metrics)   │
           │  - Grafana (dashboards)   │
           └───────────────────────────┘
```

#### Step 4: Core Components Design

**Component 1: Job Submission Interface**

**Python SDK**:
```python
# training_platform SDK
from ml_platform import TrainingJob, resources

# Simple job submission
job = TrainingJob(
    name=\"bert-finetuning-exp42\",
    image=\"my-team/bert-training:v2.0\",
    command=\"python train.py --config config.yaml\",
    resources=resources.GPU(
        count=8,
        type=\"A100\",
        memory=\"40GB\"
    ),
    framework=\"pytorch\",
    distributed=\"DDP\",  # Automatic multi-node setup
    max_runtime=\"24h\",
    checkpoint_interval=\"1h\",
    priority=\"medium\",
    tags={\"experiment\": \"42\", \"team\": \"nlp\"}
)

# Submit job
job.submit()

# Monitor job
status = job.status()
print(f\"Status: {status.phase}\")  # Pending, Running, Succeeded, Failed
print(f\"GPUs allocated: {status.allocated_gpus}\")
print(f\"Runtime: {status.runtime}\")
print(f\"Cost so far: ${status.estimated_cost}\")

# Stream logs
for log_line in job.logs(follow=True):
    print(log_line)

# Advanced: Hyperparameter tuning
from ml_platform import HyperparameterSearch

search = HyperparameterSearch(
    base_job=job,
    params={
        \"learning_rate\": [1e-5, 5e-5, 1e-4],
        \"batch_size\": [16, 32, 64]
    },
    max_parallel_trials=9,
    metric=\"val_accuracy\",
    mode=\"maximize\"
)

best_trial = search.run()
```

**CLI Tool**:
```bash
# Submit job from YAML
mlplatform submit training_job.yaml

# List jobs
mlplatform list jobs --team nlp --status running

# Get job details
mlplatform describe job bert-finetuning-exp42

# Stream logs
mlplatform logs bert-finetuning-exp42 --follow

# Stop job
mlplatform stop bert-finetuning-exp42

# Job history
mlplatform history --team nlp --last 7d
```

**Component 2: Job Scheduler (Volcano)**

```yaml
# Volcano Queue for each team
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: team-nlp
spec:
  weight: 100           # Fair-share weight
  capability:           # Hard quota
    nvidia.com/gpu: 100
  reclaimable: true     # Can borrow unused capacity from other teams
  state: Open

# Job submission translates to Volcano Job
---
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: bert-finetuning-exp42
  namespace: team-nlp
spec:
  minAvailable: 8       # Need all 8 GPUs to start (gang scheduling)
  schedulerName: volcano
  queue: team-nlp
  priorityClassName: training-medium

  tasks:
  - replicas: 1
    name: master
    template:
      spec:
        containers:
        - name: pytorch-training
          image: my-team/bert-training:v2.0
          command: [\"python\", \"train.py\"]
          resources:
            limits:
              nvidia.com/gpu: 8
          env:
          - name: MASTER_ADDR
            value: \"bert-finetuning-exp42-master-0\"
          - name: WORLD_SIZE
            value: \"8\"
          - name: RANK
            value: \"0\"

  plugins:
    svc: []               # Automatic service creation for master
    env: []               # Inject environment variables
    ssh: []               # SSH between pods for MPI jobs

  policies:
  - event: PodEvicted
    action: RestartJob   # Restart if spot instance interrupted

  - event: PodFailed
    action: AbortJob     # Don't waste resources on failed jobs

  # Automatic checkpointing
  volumeMounts:
  - name: checkpoint-storage
    mountPath: /checkpoints
```

**Component 3: Fair-Share Scheduling**

```yaml
# Fair-share configuration
# Team quotas and priorities

teams:
- name: team-nlp
  guaranteed_gpus: 100      # Guaranteed minimum
  max_gpus: 200             # Can burst up to this
  weight: 20                # 20% share when all teams competing

- name: team-cv
  guaranteed_gpus: 150
  max_gpus: 300
  weight: 30

- name: team-recsys
  guaranteed_gpus: 80
  max_gpus: 150
  weight: 15

# Scheduling policy
scheduling_policy:
  # When cluster is underutilized, teams can exceed guaranteed
  allow_borrowing: true

  # When cluster is oversubscribed, enforce fair-share
  preemption:
    enabled: true
    priority_based: true     # Higher priority jobs can preempt lower
    fairness_threshold: 0.8  # Preempt if team using >80% above fair share

# Example scenario:
# Cluster: 1000 GPUs
# team-nlp guaranteed: 100 (actual weight: 20%)
# team-cv guaranteed: 150 (actual weight: 30%)
# Current usage:
#   team-nlp: 250 GPUs (using borrowed capacity)
#   team-cv: 100 GPUs (under guaranteed)
#   team-recsys: 50 GPUs
# If team-cv submits large job needing 200 GPUs:
#   - Scheduler preempts lowest priority jobs from team-nlp
#   - team-nlp drops from 250 to 150 GPUs (fair share)
#   - team-cv gets 250 GPUs (their fair share)
```

**Component 4: Automatic Checkpointing**

```python
# Checkpoint wrapper (injected into all training jobs)
import os
import signal
import boto3
import torch
from datetime import datetime

class AutoCheckpointer:
    def __init__(self, checkpoint_dir, interval_seconds=3600):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval_seconds
        self.last_checkpoint_time = datetime.now()
        self.s3_client = boto3.client('s3')

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGUSR1, self._handle_checkpoint_signal)

    def should_checkpoint(self):
        \"\"\"Check if enough time elapsed since last checkpoint\"\"\"
        elapsed = (datetime.now() - self.last_checkpoint_time).total_seconds()
        return elapsed >= self.interval

    def save(self, model, optimizer, step, metadata=None):
        \"\"\"Save checkpoint locally and to S3\"\"\"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Save locally
        local_path = f\"{self.checkpoint_dir}/checkpoint_step_{step}.pt\"
        torch.save(checkpoint, local_path)

        # Upload to S3 asynchronously
        s3_path = f\"s3://ml-checkpoints/{os.environ['JOB_NAME']}/checkpoint_step_{step}.pt\"
        self._upload_to_s3(local_path, s3_path)

        self.last_checkpoint_time = datetime.now()
        print(f\"Checkpoint saved: {s3_path}\")

    def load_latest(self, model, optimizer):
        \"\"\"Load latest checkpoint from S3\"\"\"
        job_name = os.environ['JOB_NAME']
        s3_prefix = f\"ml-checkpoints/{job_name}/\"

        # List all checkpoints
        response = self.s3_client.list_objects_v2(
            Bucket='ml-checkpoints',
            Prefix=s3_prefix
        )

        if 'Contents' not in response:
            return 0  # No checkpoint found

        # Get latest checkpoint
        latest = max(response['Contents'], key=lambda x: x['LastModified'])

        # Download and load
        local_path = '/tmp/checkpoint.pt'
        self.s3_client.download_file(
            'ml-checkpoints',
            latest['Key'],
            local_path
        )

        checkpoint = torch.load(local_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f\"Resumed from step {checkpoint['step']}\")
        return checkpoint['step']

    def _handle_sigterm(self, signum, frame):
        \"\"\"Handle spot instance termination\"\"\"
        print(\"Received SIGTERM, saving checkpoint before shutdown...\")
        # Trigger checkpoint save
        # (Training loop should check for this signal)
        with open('/tmp/checkpoint_requested', 'w') as f:
            f.write('1')

    def _upload_to_s3(self, local_path, s3_path):
        \"\"\"Upload checkpoint to S3\"\"\"
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        self.s3_client.upload_file(local_path, bucket, key)

# Usage in training script
checkpointer = AutoCheckpointer(
    checkpoint_dir='/checkpoints',
    interval_seconds=3600  # Every hour
)

# Load checkpoint if exists
start_step = checkpointer.load_latest(model, optimizer)

for step in range(start_step, total_steps):
    # Training step
    loss = train_step(model, optimizer, batch)

    # Checkpoint periodically
    if checkpointer.should_checkpoint():
        checkpointer.save(model, optimizer, step, metadata={'loss': loss})

    # Check for termination signal
    if os.path.exists('/tmp/checkpoint_requested'):
        checkpointer.save(model, optimizer, step, metadata={'interrupted': True})
        break
```

**Component 5: Experiment Tracking (MLflow Integration)**

```python
# Automatic MLflow logging wrapper
import mlflow
import os

class TrainingJobWrapper:
    def __init__(self):
        # MLflow setup
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        self.experiment_name = os.environ['JOB_NAME']
        mlflow.set_experiment(self.experiment_name)

    def run_training(self, config):
        with mlflow.start_run(run_name=self.experiment_name):
            # Log parameters
            mlflow.log_params(config)

            # Log system info
            mlflow.log_param(\"num_gpus\", os.environ['WORLD_SIZE'])
            mlflow.log_param(\"gpu_type\", os.environ['GPU_TYPE'])

            # Train model
            model, metrics = train(config)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.pytorch.log_model(model, \"model\")

            # Log artifacts
            mlflow.log_artifact(\"/checkpoints\")

            # Tag with job metadata
            mlflow.set_tags({
                \"team\": os.environ['TEAM'],
                \"job_id\": os.environ['JOB_ID'],
                \"priority\": os.environ['PRIORITY']
            })

        return model
```

#### Step 5: Cost Management & Optimization

**Cost Allocation per Team**:

```python
# Cost tracking service
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta

def calculate_team_costs(team_name, start_date, end_date):
    \"\"\"
    Calculate actual GPU-hours consumed by team
    \"\"\"
    prom = PrometheusConnect(url='http://prometheus:9090')

    # Query: Total GPU-hours for team
    query = f'''
        sum(
            avg_over_time(
                kube_pod_container_resource_requests{{
                    namespace=\"{team_name}\",
                    resource=\"nvidia_com_gpu\",
                    pod=~\".*-training-.*\"
                }}[{(end_date - start_date).total_seconds()}s]
            )
        ) / 3600
    '''

    result = prom.custom_query(query)
    gpu_hours = float(result[0]['value'][1])

    # Cost calculation
    # 50% on reserved ($2/GPU-hr), 40% on spot ($1.25/GPU-hr), 10% on-demand ($4/GPU-hr)
    avg_cost_per_gpu_hour = (0.5 * 2.0) + (0.4 * 1.25) + (0.1 * 4.0)  # $2.00
    total_cost = gpu_hours * avg_cost_per_gpu_hour

    return {
        'team': team_name,
        'gpu_hours': gpu_hours,
        'total_cost': total_cost,
        'period': f\"{start_date} to {end_date}\"
    }

# Generate monthly cost reports
teams = ['team-nlp', 'team-cv', 'team-recsys']
for team in teams:
    costs = calculate_team_costs(
        team,
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    print(f\"{team}: ${costs['total_cost']:.2f} ({costs['gpu_hours']:.1f} GPU-hours)\")
    send_team_cost_report(team, costs)
```

**Spot Instance Strategy**:

```yaml
# Mixed node group: on-demand + spot
# EKS node groups
resource \"aws_eks_node_group\" \"gpu_training_spot\" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = \"gpu-training-spot-a100\"

  instance_types = [\"p4d.24xlarge\"]  # 8x A100

  capacity_type = \"SPOT\"

  scaling_config {
    desired_size = 20   # 160 GPUs on spot
    max_size     = 50   # Can scale to 400 GPUs
    min_size     = 10   # Minimum 80 GPUs
  }

  labels = {
    \"capacity-type\" = \"spot\"
    \"gpu-type\"      = \"a100\"
  }

  taints = [
    {
      key    = \"spot-instance\"
      value  = \"true\"
      effect = \"NoSchedule\"
    }
  ]
}

resource \"aws_eks_node_group\" \"gpu_training_ondemand\" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = \"gpu-training-ondemand-a100\"

  instance_types = [\"p4d.24xlarge\"]

  capacity_type = \"ON_DEMAND\"

  scaling_config {
    desired_size = 30   # 240 GPUs on-demand (baseline)
    max_size     = 40
    min_size     = 25
  }

  labels = {
    \"capacity-type\" = \"on-demand\"
    \"gpu-type\"      = \"a100\"
  }
}

# Jobs tolerate spot instances by default
# High-priority jobs can request on-demand:
spec:
  priorityClassName: production-high
  nodeSelector:
    capacity-type: on-demand  # Force on-demand for critical jobs
```

#### Step 6: User Experience Enhancements

**Job Templates**:

```yaml
# templates/pytorch-distributed.yaml
# Pre-configured template for common use cases
apiVersion: ml.company.com/v1
kind: JobTemplate
metadata:
  name: pytorch-distributed-training
spec:
  description: \"Multi-node PyTorch DDP training\"
  parameters:
  - name: num_gpus
    type: integer
    default: 8
    description: \"Total GPUs to use\"

  - name: docker_image
    type: string
    description: \"Training container image\"

  - name: training_script
    type: string
    default: \"train.py\"

  - name: dataset_path
    type: string
    description: \"S3 path to training data\"

  job_spec:
    framework: pytorch
    distributed: DDP
    resources:
      gpus: \"{{ num_gpus }}\"
      memory: \"{{ num_gpus * 50 }}Gi\"
    image: \"{{ docker_image }}\"
    command: \"python {{ training_script }} --data {{ dataset_path }}\"
    checkpointing:
      enabled: true
      interval: 1h
    monitoring:
      mlflow: true
      tensorboard: true

# Use template
mlplatform submit --template pytorch-distributed-training \\
  --param num_gpus=32 \\
  --param docker_image=my-team/training:v1.0 \\
  --param dataset_path=s3://datasets/imagenet/
```

**JupyterHub Integration**:

```yaml
# Launch Jupyter notebook with GPU
apiVersion: v1
kind: Pod
metadata:
  name: jupyter-{{ username }}
  namespace: team-nlp
spec:
  containers:
  - name: jupyter
    image: jupyter/pytorch-notebook:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: MLFLOW_TRACKING_URI
      value: \"http://mlflow-server:5000\"
    volumeMounts:
    - name: home
      mountPath: /home/jovyan
  volumes:
  - name: home
    persistentVolumeClaim:
      claimName: jupyter-{{ username }}-home

# From Jupyter, submit training jobs:
from ml_platform import TrainingJob

job = TrainingJob.from_notebook(
    name=\"interactive-experiment\",
    notebook_path=\"experiment.ipynb\",
    gpus=4
)
job.submit()
```

#### Summary: Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Scheduler | Volcano | Gang scheduling, fair-share, queue management |
| Orchestration | Kubernetes (EKS) | Industry standard, rich ecosystem |
| Experiment Tracking | MLflow | Model registry, versioning, artifact storage |
| Checkpointing | S3 + Auto-wrapper | Fault tolerance, spot instance support |
| Job Submission | Python SDK + CLI | Low friction for data scientists |
| Resource Management | Quotas + Priority + Fair-share | Multi-tenancy, efficiency, fairness |
| Cost Optimization | 50% reserved + 40% spot | Balance cost (60% savings) with reliability |

**Platform Metrics**:
- Time to submit job: <1 minute (vs 30 min manual setup)
- GPU utilization: 85% (vs 40% before platform)
- Cost per GPU-hour: $2.00 (vs $4.00 on-demand)
- Jobs per day: 500+ (vs 50 manually)
- Spot interruption overhead: <5% (automatic checkpointing)

---

### System Design Question 3: Building LLM Inference Infrastructure at Scale

**Q**: Design an infrastructure to serve a 70B parameter LLM (like Llama 2 70B) for a product with 10M daily active users. Requirements: <2s latency for first token, 50 tokens/second generation speed, 99.9% availability. How would you architect this?

**Answer Framework**:

[Due to length, I'll provide abbreviated version - let me know if you want full detail]

#### Key Components:

1. **Model Serving**: vLLM with PagedAttention for efficient KV cache management
2. **Load Balancing**: Request routing based on sequence length and batch state
3. **Auto-Scaling**: Scale based on queue depth (pending requests)
4. **Multi-Region**: Deploy in 3 regions (US-East, US-West, EU) for global coverage
5. **Cost Optimization**: H100 GPUs (faster inference) vs A100 (cheaper) trade-off

**Capacity Calculation**:
- 10M DAU, assume 5 requests/user/day = 50M requests/day
- Peak QPS: 50M / (24 * 3600) * 3 (peak factor) ≈ 1,700 QPS
- Each H100 with vLLM: ~8 QPS (with continuous batching)
- GPUs needed: 1,700 / 8 = 213 H100s
- With headroom: ~280 H100s (35 nodes of 8x H100 each)

**Cost**: ~$2.5M/month (can optimize to $1.2M with reserved instances + 24/7 load balancing)

---

## HANDS-ON SCENARIOS & TROUBLESHOOTING {#hands-on}

### Scenario 1: Debugging OOMKilled Training Job

**Problem**: A PyTorch training job on Kubernetes keeps getting killed with \"OOMKilled\" status. The pod requests 8x A100 GPUs (40GB each) and the model is 7B parameters. Debug and fix.

**Diagnosis Steps**:

```bash
# 1. Check pod status
kubectl describe pod training-job-xyz

# Output shows:
#   State: Terminated
#   Reason: OOMKilled
#   Exit Code: 137

# 2. Check memory requests vs limits
kubectl get pod training-job-xyz -o yaml | grep -A 5 resources

# Output:
#   resources:
#     limits:
#       nvidia.com/gpu: 8
#       memory: 200Gi  # <<< Problem: underestimated
#     requests:
#       nvidia.com/gpu: 8
#       memory: 100Gi

# 3. Check actual GPU memory usage before crash
# (if we had monitoring)
kubectl logs training-job-xyz | grep -i memory

# 4. Calculate actual memory requirements
# 7B params * 2 bytes (fp16) = 14GB model
# + gradients (14GB)
# + optimizer states (28GB for Adam)
# + activation memory (~50GB for large batch)
# Total: ~106GB
# Across 8 GPUs: ~13GB per GPU (fits in 40GB)

# But CPU memory also needed:
# - DataLoader workers: 10GB
# - Data preprocessing: 20GB
# - PyTorch overhead: 10GB
# Total CPU memory: ~40GB
# Plus system overhead: 50-60GB

# Problem: memory limit too low!
```

**Root Cause**:
- Pod requests only 200Gi RAM, but actual need is ~300Gi
- Kubernetes OOM killer terminates pod when exceeding limit

**Fix**:

```yaml
# Updated job spec
resources:
  limits:
    nvidia.com/gpu: 8
    memory: 400Gi  # Increased from 200Gi
  requests:
    nvidia.com/gpu: 8
    memory: 350Gi  # Increased from 100Gi

# Also add shared memory for DataLoader
volumes:
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: 64Gi  # For multi-worker DataLoader

volumeMounts:
- name: shm
  mountPath: /dev/shm
```

**Prevention**:
- Always use memory profiling tools (PyTorch Profiler)
- Set memory limits 30-40% above estimated need
- Monitor actual usage and adjust

---

### Scenario 2: GPU Utilization at 20% During Training

**Problem**: Training job running but GPU utilization stuck at 20%. Training is much slower than expected.

**Diagnosis**:

```bash
# 1. Check GPU utilization in real-time
kubectl exec -it training-pod-xyz -- nvidia-smi dmon

# Output shows:
#   gpu   pwr  gtemp  mtemp    sm   mem   enc   dec  mclk  pclk
#     0    75     45     -    20    10     0     0  1215  1410
#     1    75     45     -    20    10     0     0  1215  1410
# sm (streaming multiprocessor) at 20% <<<

# 2. Check if CPU is bottleneck
kubectl exec -it training-pod-xyz -- top

# Output shows:
#   CPU: 95% - 98%  <<< CPU bottleneck!

# 3. Profile training script
kubectl exec -it training-pod-xyz -- python -m torch.utils.bottleneck train.py
```

**Root Causes**:
1. **Data Loading Bottleneck**: Too few DataLoader workers
2. **CPU Preprocessing**: Heavy augmentation on CPU instead of GPU
3. **Small Batch Size**: GPU not saturated

**Fix**:

```python
# Before (inefficient):
train_loader = DataLoader(
    dataset,
    batch_size=32,    # Too small for A100
    num_workers=4,    # Too few workers
    pin_memory=False
)

# Data augmentation on CPU
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# After (optimized):
train_loader = DataLoader(
    dataset,
    batch_size=256,        # Larger batch for A100
    num_workers=16,        # More workers to saturate CPU
    pin_memory=True,       # Faster CPU-GPU transfer
    prefetch_factor=4,     # Prefetch 4 batches per worker
    persistent_workers=True # Keep workers alive
)

# Move augmentation to GPU with NVIDIA DALI
import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_torch

# DALI pipeline for GPU augmentation
@dali.pipeline_def
def image_pipeline():
    images = dali.fn.readers.file(file_root=data_dir)
    images = dali.fn.decoders.image(images, device='mixed')  # Decode on GPU
    images = dali.fn.random_resized_crop(images, size=(224, 224), device='gpu')
    images = dali.fn.flip(images, horizontal=1, device='gpu')
    return images

# Additional: Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = criterion(output, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Result**: GPU utilization increases from 20% to 85-95%

---

### Scenario 3: Kubernetes Pod Stuck in Pending State

**Problem**: Submitted a training job 30 minutes ago, but pods still in \"Pending\" state.

**Diagnosis**:

```bash
# 1. Check pod status
kubectl get pods training-job-abc

# Output:
#   NAME                  READY   STATUS    RESTARTS   AGE
#   training-job-abc-0    0/1     Pending   0          30m

# 2. Describe pod for events
kubectl describe pod training-job-abc-0

# Output shows:
#   Events:
#     Warning  FailedScheduling  30m  default-scheduler
#       0/10 nodes are available: 10 Insufficient nvidia.com/gpu

# Problem: No nodes with available GPUs!

# 3. Check GPU availability across cluster
kubectl describe nodes | grep -A 5 \"nvidia.com/gpu\"

# Output:
#   Allocatable:
#     nvidia.com/gpu: 8
#   Allocated:
#     nvidia.com/gpu: 8  <<< All GPUs allocated

# 4. Check what's using GPUs
kubectl get pods --all-namespaces -o json | \\
  jq -r '.items[] | select(.spec.containers[].resources.limits.\"nvidia.com/gpu\" != null) | .metadata.name'

# Output shows many old completed jobs still holding GPUs!
```

**Root Causes**:
1. Completed jobs not cleaned up (pods still exist)
2. No cluster autoscaler configured
3. Resource quota reached

**Fix**:

```bash
# Immediate fix: Clean up completed jobs
kubectl delete pods --field-selector status.phase=Succeeded
kubectl delete pods --field-selector status.phase=Failed

# Long-term fix 1: Set TTL for jobs
```

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  ttlSecondsAfterFinished: 3600  # Delete 1 hour after completion
  template:
    spec:
      restartPolicy: Never
```

```bash
# Long-term fix 2: Enable cluster autoscaler
```

```yaml
# Cluster autoscaler configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        command:
        - ./cluster-autoscaler
        - --v=4
        - --cloud-provider=aws
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/ml-cluster
        - --scale-down-delay-after-add=10m
```

```bash
# Long-term fix 3: Implement resource quotas with warnings
kubectl create quota team-nlp-quota --hard=nvidia.com/gpu=40 --namespace=team-nlp

# Set up alerts when quota 80% utilized
```

---

### Scenario 4: Model Serving Latency Spike

**Problem**: Model serving API suddenly experiencing p99 latency of 800ms (SLA is 100ms). No recent deployments.

**Diagnosis**:

```bash
# 1. Check current latency distribution
curl http://triton-serving:8002/metrics | grep nv_inference_request_duration

# Output shows:
#   nv_inference_request_duration_us_count{model=\"resnet50\"} 1.5e6
#   nv_inference_request_duration_us_sum{model=\"resnet50\"} 1.2e9
#   Average: 800μs = 800ms <<< High!

# 2. Check request rate
curl http://triton-serving:8002/metrics | grep nv_inference_request_success

# Output:
#   rate(nv_inference_request_success[1m]): 2000 QPS
#   Normally: 500 QPS <<< 4x traffic spike!

# 3. Check GPU utilization
kubectl exec -it triton-pod-xyz -- nvidia-smi

# Output:
#   GPU Utilization: 100%
#   Memory Utilization: 95%

# 4. Check autoscaler status
kubectl get hpa triton-serving-hpa

# Output:
#   REFERENCE             TARGETS    MINPODS   MAXPODS   REPLICAS   AGE
#   Deployment/triton     2000/1000  10        50        10         30d
#                         ^^^^^ Current RPS per pod way above target

# Problem: Traffic spike, autoscaler hasn't scaled yet!

# 5. Check why autoscaler not scaling
kubectl describe hpa triton-serving-hpa

# Output:
#   Conditions:
#     AbleToScale: True
#     ScalingLimited: True  <<< Hit max replicas!
#   Events:
#     Unable to scale: maximum replicas reached (50)

# Root cause: Hit max replicas, need more capacity
```

**Immediate Fix**:

```bash
# Increase max replicas
kubectl patch hpa triton-serving-hpa --patch '{\"spec\":{\"maxReplicas\":100}}'

# Manually scale up immediately
kubectl scale deployment triton-serving --replicas=80
```

**Long-term Fixes**:

1. **Implement Request Queuing & Rate Limiting**:
```python
# API Gateway with rate limiting
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

@app.post(\"/predict\")
@limiter.limit(\"100/minute\")  # Max 100 req/min per client
async def predict(request: Request):
    # Queue request if backends overloaded
    if get_current_load() > 0.9:
        await request_queue.put(request)
        return {\"status\": \"queued\", \"position\": request_queue.qsize()}

    return await process_request(request)
```

2. **Predictive Autoscaling**:
```yaml
# Scheduled scaling for known traffic patterns
apiVersion: autoscaling.k8s.io/v1
kind: ScheduledScaler
metadata:
  name: triton-scheduled-scaling
spec:
  schedule:
  - name: morning-peak
    cron: \"0 9 * * 1-5\"  # 9 AM weekdays
    minReplicas: 80
    maxReplicas: 150

  - name: evening-off-peak
    cron: \"0 22 * * *\"   # 10 PM daily
    minReplicas: 20
    maxReplicas: 50
```

3. **Multi-Region Failover**:
```python
# Route 53 latency-based routing with health checks
# If one region overloaded, shift traffic to others
```

---

### Scenario 5: Distributed Training Hanging at Initialization

**Problem**: Multi-node PyTorch DDP training job stuck at initialization. Logs show \"Waiting for all processes to join...\" but never proceeds.

**Diagnosis**:

```bash
# 1. Check if all pods started
kubectl get pods -l job-name=distributed-training

# Output:
#   NAME                        READY   STATUS    RESTARTS   AGE
#   distributed-training-0      1/1     Running   0          10m
#   distributed-training-1      1/1     Running   0          10m
#   distributed-training-2      0/1     Pending   0          10m  <<< Not started!
#   distributed-training-3      1/1     Running   0          10m

# Problem: Pod 2 stuck in Pending, other pods waiting for it

# 2. Why is pod 2 pending?
kubectl describe pod distributed-training-2

# Output:
#   Events:
#     FailedScheduling: 0/20 nodes are available: 1 node had taint that pod didn't tolerate

# Pod 2 can't schedule, blocking entire job

# 3. Check if using gang scheduling
kubectl get job distributed-training -o yaml | grep minAvailable

# Output: (nothing) <<< Not using gang scheduling!

# Without gang scheduling:
# - Pods start independently
# - Some pods wait forever for others that can't start
# - Wastes resources
```

**Root Cause**: Not using gang scheduling (all-or-nothing scheduling)

**Fix**:

```yaml
# Use Volcano for gang scheduling
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: distributed-training
spec:
  minAvailable: 4  # Need all 4 pods to start job
  schedulerName: volcano
  tasks:
  - replicas: 4
    name: worker
    template:
      spec:
        containers:
        - name: pytorch
          image: pytorch/pytorch:2.0
          resources:
            limits:
              nvidia.com/gpu: 8

  plugins:
    ssh: []     # Enable SSH between pods
    svc: []     # Create service for master

  policies:
  - event: PodEvicted
    action: RestartJob

  - event: PodFailed
    action: AbortJob  # Don't waste resources if one fails
```

**Additional Debugging**:

```bash
# Check network connectivity between pods
kubectl exec -it distributed-training-0 -- ping distributed-training-1

# Check if NCCL can see all GPUs
kubectl exec -it distributed-training-0 -- python -c \"import torch; print(torch.cuda.device_count())\"

# Check environment variables
kubectl exec -it distributed-training-0 -- env | grep -E \"RANK|WORLD_SIZE|MASTER\"

# NCCL debug logs
kubectl exec -it distributed-training-0 -- bash
export NCCL_DEBUG=INFO
python train.py  # Will show NCCL initialization details
```

---

## BEHAVIORAL QUESTIONS FOR INFRASTRUCTURE ROLES {#behavioral}

### Framework: STAR Method

**S**ituation: Set the context
**T**ask: Describe your responsibility
**A**ction: Explain what you did (focus on YOUR actions)
**R**esult: Quantify the outcome

### Question 1: Describe a time you optimized infrastructure costs significantly

**Sample Answer**:

**Situation**: At my previous company, our ML training infrastructure costs were $400K/month on AWS, which was 40% over budget. Leadership asked our team to reduce costs by at least 25% without impacting data scientist productivity.

**Task**: I was assigned to lead the cost optimization initiative. I needed to identify cost drivers, propose optimizations, and implement them with minimal disruption.

**Action**:
1. **Analysis**: I analyzed 3 months of CloudWatch and Kubernetes metrics to identify waste:
   - 30% of GPU instances running idle (jobs completed but pods not terminated)
   - 60% of training jobs running on on-demand instances when spot would suffice
   - Many jobs over-provisioning resources (requesting 8 GPUs but using only 4)

2. **Quick Wins (Month 1)**:
   - Implemented Job TTL (auto-delete after 1 hour completion): Saved $40K/month
   - Created team dashboards showing real-time cost and GPU utilization
   - Set up Slack alerts when jobs run >24 hours idle

3. **Strategic Changes (Month 2-3)**:
   - Migrated 70% of training workloads to spot instances with automatic checkpointing: Saved $120K/month
   - Purchased 1-year reserved instances for baseline capacity: Saved $60K/month
   - Built resource recommendation tool using historical job data: Reduced over-provisioning by 40%

4. **Cultural Change**:
   - Presented cost dashboards in weekly team meetings
   - Introduced \"cost efficiency\" metric alongside model accuracy
   - Documented best practices guide for data scientists

**Result**:
- Reduced costs from $400K to $220K/month (45% reduction) in 3 months
- GPU utilization increased from 45% to 82%
- Zero complaints from data scientists (they didn't even notice the spot migration thanks to checkpointing)
- Leadership promoted the initiative company-wide, saving additional $500K across other teams

**What I Learned**: Cost optimization is as much about culture and visibility as it is about technical fixes. Making costs visible to users naturally drives better behavior.

---

### Question 2: Tell me about a time you had to debug a complex production issue

**Sample Answer**:

**Situation**: At 2 AM, our model serving API started returning 500 errors for 15% of requests. The on-call alert woke me up. This was our main revenue-generating service, and every minute of downtime cost thousands in lost business.

**Task**: As the on-call engineer, I needed to identify the root cause and restore service ASAP while minimizing impact to users.

**Action**:

1. **Immediate Triage** (5 minutes):
   - Checked Grafana dashboards: Error rate spiked 10 minutes ago
   - Verified recent deployments: None in last 24 hours (not a bad deploy)
   - Checked infrastructure metrics: No obvious resource exhaustion

2. **Hypothesis Formation** (10 minutes):
   - Sampled error logs: \"CUDA out of memory\" errors on 3 out of 20 pods
   - Those 3 pods were all on same Kubernetes node (node-xyz)
   - Hypothesis: GPU issue on specific node

3. **Investigation** (15 minutes):
   ```bash
   # SSH to node-xyz
   nvidia-smi

   # Output showed GPU 2 had ECC errors
   # GPU 2 was in unhealthy state but K8s didn't know
   ```

   - Root cause: GPU hardware failure, but Kubernetes device plugin doesn't detect bad GPUs, keeps scheduling pods to failed node

4. **Immediate Fix** (5 minutes):
   ```bash
   # Cordon node to prevent new pods
   kubectl cordon node-xyz

   # Drain existing pods to healthy nodes
   kubectl drain node-xyz --ignore-daemonsets --delete-emptydir-data
   ```

   - Error rate dropped to 0% within 2 minutes

5. **Long-term Fix** (Next Day):
   - Deployed custom DaemonSet to monitor GPU health:
   ```python
   # gpu-health-checker.py
   import pynvml
   import subprocess

   pynvml.nvmlInit()
   gpu_count = pynvml.nvmlDeviceGetCount()

   for i in range(gpu_count):
       handle = pynvml.nvmlDeviceGetHandleByIndex(i)
       ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(handle)

       if ecc_errors > 100:
           # Taint node to prevent scheduling
           subprocess.run([\"kubectl\", \"taint\", \"nodes\", NODE_NAME,
                          \"gpu-unhealthy=true:NoSchedule\"])
   ```

   - Configured alerting for GPU ECC errors
   - Worked with AWS support to replace faulty instance

**Result**:
- Restored service in 35 minutes (RTO target was 1 hour)
- Only 0.3% of users impacted (thanks to load balancing across healthy pods)
- Prevented future occurrences with proactive GPU health monitoring
- Post-incident report led to improved runbooks for GPU issues

**What I Learned**:
- Always verify infrastructure health, not just application health
- Importance of detailed observability (GPU-level metrics were missing)
- Having clear runbooks accelerates incident response

---

### Question 3: Describe a time you had to make a difficult technical trade-off decision

**Sample Answer**:

**Situation**: We were building a new ML training platform for the company. A key decision was whether to build a custom Kubernetes operator for ML jobs or use an existing solution like Kubeflow.

**Task**: As the tech lead, I needed to evaluate both options and make a recommendation that balanced short-term velocity with long-term maintainability.

**Action**:

1. **Gathered Requirements** (Week 1):
   - Interviewed 20 data scientists about pain points with current system
   - Key needs: Simple job submission, automatic checkpointing, GPU quota management
   - Nice-to-haves: Hyperparameter tuning, A/B testing, pipeline orchestration

2. **Evaluated Options**:

   **Option A: Kubeflow**
   - Pros:
     - Battle-tested, used by Google, Uber, etc.
     - Rich features (pipelines, katib, notebooks)
     - Active community, regular updates
   - Cons:
     - Heavy (10+ components to deploy)
     - Opinionated architecture (may not fit our needs)
     - Overhead to customize for our workflows
     - Team needs to learn Kubeflow abstractions

   **Option B: Custom Operator**
   - Pros:
     - Perfect fit for our exact needs
     - Lightweight, only what we need
     - Team retains full control
   - Cons:
     - 3-6 months development time
     - Ongoing maintenance burden
     - \"Not invented here\" syndrome
     - Risk of reinventing the wheel

3. **Analysis**:
   - Created comparison matrix with weighted criteria
   - Built POC with both approaches (2 weeks each)
   - Presented findings to team and stakeholders

4. **Decision**:
   - **Chose hybrid approach**:
     - Use Kubeflow for experiment tracking (notebooks, pipelines)
     - Build custom lightweight operator for job submission
     - Custom operator wraps Kubeflow's PyTorchJob under the hood

   **Rationale**:
   - Kubeflow too heavy for simple job submission (data scientists found it complex)
   - But Kubeflow pipelines valuable for ML workflows
   - Custom operator provides simple interface, Kubeflow provides advanced features
   - 80% of users use simple custom interface, 20% use full Kubeflow for complex workflows

**Result**:
- Launched platform in 4 months (vs 6 months custom, 2 months Kubeflow-only)
- 85% adoption rate among data scientists (vs 40% with previous system)
- Reduced average time to submit training job from 45 minutes to 2 minutes
- Custom operator was only 2,000 lines of code (manageable maintenance)
- Still benefited from Kubeflow community for pipeline features

**What I Learned**:
- Don't force binary choices (build vs buy) - hybrid approaches often best
- User experience matters more than technical elegance
- Leverage existing tools where they excel, customize where needed
- POCs are worth the investment for major architectural decisions

---

### Question 4: How do you handle disagreements with teammates on technical approaches?

**Sample Answer**:

**Situation**: On my last project, we were designing auto-scaling for our model serving infrastructure. I advocated for using Kubernetes HPA with custom metrics, while a senior engineer strongly preferred KEDA (Kubernetes Event-Driven Autoscaling).

**Task**: We needed to reach consensus to move forward, as the architecture decision would impact the entire team's work for the next 6 months.

**Action**:

1. **Understood Their Perspective**:
   - I scheduled a 1:1 to deeply understand their reasoning
   - They argued KEDA provides more event sources (Kafka, SQS) and simpler configuration
   - Acknowledged valid points: KEDA does have richer integrations

2. **Presented My Concerns**:
   - HPA is built into Kubernetes, one less dependency
   - We already have Prometheus metrics, don't need external event sources
   - Team familiar with HPA, KEDA would require training
   - But remained open: \"I might be wrong, let's validate together\"

3. **Data-Driven Comparison**:
   - Proposed: \"Let's build POCs of both approaches and measure objectively\"
   - Defined success criteria:
     - Scaling latency (time to scale up)
     - Complexity (lines of YAML)
     - Operational burden (monitoring, debugging)
   - Split work: I built HPA POC, they built KEDA POC
   - Tested both with realistic traffic patterns

4. **Results**:
   - HPA: 45 seconds scale-up latency, 80 lines YAML, standard Kubernetes debugging
   - KEDA: 30 seconds scale-up latency, 50 lines YAML, but required external scaler deployment

5. **Compromise**:
   - KEDA was technically superior for scale-up latency (our primary metric)
   - BUT, I raised operational concern: another component to manage
   - **Solution**: Agreed to use KEDA, but I took ownership of runbooks and monitoring for it
   - They agreed to pair with me on KEDA training for the team

**Result**:
- Implemented KEDA successfully, met all performance goals
- Learned a new technology (KEDA) that expanded my skillset
- Strengthened relationship with senior engineer through collaborative problem-solving
- Team benefited from both our perspectives

**What I Learned**:
- **Disagree and commit**: Once we had data, I committed fully even though HPA was \"my\" preference
- Technical disagreements are healthy if handled professionally
- POCs are worth the time investment for major decisions
- Ego has no place in engineering - the best idea should win, not the loudest voice

**Follow-up**: I've since used this \"POC-driven decision making\" approach many times. It removes emotion from technical debates.

---

### Question 5: Tell me about a time you improved team productivity through tooling or automation

**Sample Answer**:

**Situation**: Data scientists on my team spent ~30% of their time on infrastructure tasks: setting up training jobs, debugging Kubernetes issues, managing dependencies. This was frustrating for them and inefficient for the company.

**Task**: I wanted to reduce this overhead and let data scientists focus on modeling. My goal was to cut infrastructure time from 30% to <10%.

**Action**:

1. **Identified Pain Points** (Week 1):
   - Shadowed 5 data scientists for a day each
   - Biggest time sinks:
     - Writing K8s YAML for training jobs (30 min avg)
     - Debugging pod failures (1-2 hours when it happens)
     - Environment/dependency management (\"it works on my laptop\")
     - Waiting for cluster access approvals (2-3 days)

2. **Built Self-Service Platform** (Month 1-2):
   - **CLI tool** for job submission:
     ```bash
     # Old way: Write 100 lines of K8s YAML
     # New way:
     ml train --gpus 8 --image my-team/training:v1 --script train.py
     ```
   - **Pre-built Docker images** with common ML frameworks
   - **Template library** for common job patterns
   - **Auto-provisioned namespaces** for each user (no approval needed)

3. **Improved Debugging Experience** (Month 3):
   - Created troubleshooting dashboard showing common issues
   - Built `ml debug` command that automatically diagnoses issues:
     ```bash
     ml debug my-training-job

     # Output:
     # ❌ Job failed: OOMKilled
     # 💡 Suggestion: Your job used 245Gi memory but requested only 100Gi
     # 📝 Fix: Add --memory 300Gi to your command
     # 🔗 Documentation: https://wiki/oom-guide
     ```

4. **Documentation & Training**:
   - Wrote 10-minute quickstart guide
   - Recorded 5-minute demo video
   - Held weekly office hours for questions
   - Created #ml-platform Slack channel for support

5. **Measured Impact**:
   - Tracked time spent on infra tasks via survey (before/after)
   - Monitored tool adoption rate
   - Collected feedback every 2 weeks

**Result**:
- Reduced infra time from 30% to 8% (measured via monthly survey)
- Job submission time: 30 min → 2 min (93% faster)
- 95% of data scientists using platform within 3 months
- **Productivity gain**: 500 data scientists * 22% time saved = 110 FTE-equivalent capacity unlocked
- Reduced support tickets to infra team by 60%
- Promoted to Senior Engineer partly due to this impact

**What I Learned**:
- Best tools are invisible - data scientists don't think about infra anymore
- User research (shadowing) >>> assumptions about user needs
- Incremental rollout builds trust (we didn't force migration)
- Support and documentation as important as the tool itself

---

### Additional Behavioral Questions to Prepare

1. **Tell me about your most challenging technical project**
2. **Describe a time you failed. What did you learn?**
3. **How do you stay current with rapidly evolving AI/ML infrastructure technologies?**
4. **Give an example of when you had to balance technical debt vs new features**
5. **Describe a time you mentored a junior engineer**
6. **Tell me about a time you had to work with a difficult stakeholder**
7. **How do you prioritize when you have multiple urgent tasks?**
8. **Describe a time you made a mistake in production. How did you handle it?**
9. **Tell me about a time you advocated for better engineering practices**
10. **How do you approach on-call responsibilities and incident response?**

---

## INTERVIEW DAY STRATEGY {#interview-strategy}

### Before the Interview

**1 Week Before**:
- Research company's ML/AI products and infrastructure scale
- Review their engineering blog for infrastructure topics
- Identify technologies they use (from job description, LinkedIn, Glassdoor)
- Prepare 5-7 questions to ask interviewers

**1 Day Before**:
- Review key concepts from this guide
- Practice 2-3 system design questions out loud
- Prepare your \"tell me about yourself\" 2-minute pitch
- Test video/audio setup if remote

**Morning Of**:
- Light review of notes (don't cram)
- Eat well, stay hydrated
- Arrive 10 minutes early (virtual or in-person)

### During the Interview

**Structure**:
- Typical AI Infrastructure interview loop (4-6 hours):
  1. Behavioral/Experience (30-45 min)
  2. Technical Deep Dive (45-60 min)
  3. System Design (60 min)
  4. Coding/Hands-On (45-60 min)
  5. Cultural Fit/Team Match (30 min)
  6. Q&A with Hiring Manager (30 min)

**For Each Round**:

1. **Clarify Questions**:
   - Don't jump straight to solutions
   - Ask clarifying questions for 2-3 minutes
   - Confirm your understanding before proceeding

2. **Think Out Loud**:
   - Verbalize your thought process
   - Interviewers want to understand how you think
   - \"I'm considering A vs B because...\"

3. **Structure Your Answer**:
   - High-level approach first
   - Then dive into details
   - Use diagrams for system design

4. **Acknowledge Trade-offs**:
   - Every technical decision has trade-offs
   - \"We could do X for simplicity, or Y for performance\"
   - Shows mature engineering judgment

5. **Be Honest**:
   - If you don't know something: \"I haven't used X, but I'd approach it by...\"
   - Better than making stuff up

### Red Flags to Avoid

- **Don't**:
  - Badmouth previous employers
  - Claim expertise in technologies you've only read about
  - Ignore hints from interviewer (they're trying to help!)
  - Get defensive when challenged
  - Rush through questions without clarifying
  - Forget to ask questions at the end

- **Do**:
  - Show enthusiasm for ML infrastructure
  - Demonstrate curiosity and learning mindset
  - Discuss trade-offs and alternatives
  - Ask thoughtful questions about their infrastructure
  - Follow up with thank-you email within 24 hours

### Questions to Ask Interviewers

**About the Team**:
- \"What's the biggest infrastructure challenge your team is facing?\"
- \"How does the ML infrastructure team collaborate with data scientists/researchers?\"
- \"What's the on-call rotation like? How do you handle incidents?\"

**About Technology**:
- \"What's your ML infrastructure stack? (K8s, cloud provider, serving framework)\"
- \"Are you currently migrating to any new technologies?\"
- \"What percentage of your models are in production vs experimental?\"

**About Growth**:
- \"What would success look like for this role in the first 6 months?\"
- \"How does the team approach professional development?\"
- \"What's a recent project the team is proud of?\"

**About Culture**:
- \"How do you balance innovation vs stability in infrastructure?\"
- \"What's the process for proposing new technologies or approaches?\"
- \"How are technical decisions made on the team?\"

---

## FINAL TIPS FOR SUCCESS

### What Interviewers Are Really Looking For

1. **Technical Depth**: Deep knowledge in core areas (K8s, cloud, ML)
2. **Breadth**: Awareness of adjacent technologies
3. **Problem-Solving**: How you approach unfamiliar challenges
4. **Communication**: Can you explain complex topics clearly?
5. **Ownership**: Do you take initiative and see things through?
6. **Collaboration**: Work well with diverse teams (ML, SRE, product)
7. **Growth Mindset**: Continuously learning and improving

### Common Pitfalls & How to Avoid Them

| Pitfall | How to Avoid |
|---------|-------------|
| Over-engineering solutions | Start simple, then add complexity only if needed |
| Not asking clarifying questions | Always spend first 3-5 min understanding requirements |
| Memorizing answers | Understand concepts deeply, not surface-level |
| Not practicing out loud | Do mock interviews with friends/peers |
| Ignoring soft skills | Practice STAR method for behavioral questions |
| Not researching company | Spend 1-2 hours researching before interview |

### Study Resources

**Kubernetes**:
- \"Kubernetes in Action\" by Marko Lukša
- Official K8s documentation (especially on GPUs, Jobs)
- CNCF Kubeflow tutorials

**Cloud Platforms**:
- AWS Solutions Architect course (A Cloud Guru)
- GCP Professional Cloud Architect cert prep
- Hands-on labs on Qwiklabs

**ML Infrastructure**:
- \"Designing Machine Learning Systems\" by Chip Huyen
- Kubeflow documentation
- NVIDIA Triton documentation
- MLOps community resources

**System Design**:
- \"System Design Interview\" by Alex Xu
- ML-specific: Check out Eugene Yan's blog
- Practice on Pramp, Interviewing.io

---

**Remember**: Interviews are a two-way street. You're evaluating them as much as they're evaluating you. Show confidence, be authentic, and demonstrate your passion for ML infrastructure. Good luck!

---

*This guide is based on real job postings and interview experiences. For the main technical interview prep guide, see AI_Infrastructure_Interview_Prep_Guide.md.*
