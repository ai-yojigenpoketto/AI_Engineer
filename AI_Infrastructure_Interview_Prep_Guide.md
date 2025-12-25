# AI Infrastructure & AIOps Interview Preparation Guide
## Complete Interview Preparation for AI Infrastructure Engineer, AIOps Engineer, ML Infrastructure Engineer, and AI Platform Engineer Roles

**Based on Analysis of 42 Job Postings (December 2025)**

---

## TABLE OF CONTENTS

1. [Executive Summary & Study Approach](#executive-summary)
2. [Technical Interview Questions by Skill Domain](#technical-questions)
   - Cloud Platforms (AWS/GCP/Azure) for AI/ML
   - Kubernetes & Container Orchestration
   - Python for Infrastructure Automation
   - Infrastructure as Code (Terraform)
   - Model Serving Platforms
   - MLOps Platforms
   - ML Frameworks (Infrastructure Perspective)
   - Monitoring & Observability
   - CI/CD for ML Pipelines
   - Distributed Training Infrastructure
3. [System Design Questions](#system-design)
4. [Hands-On Scenarios & Troubleshooting](#hands-on)
5. [Behavioral Questions](#behavioral)
6. [Study Timeline & Preparation Strategy](#study-timeline)
7. [Interview Tips & Best Practices](#interview-tips)

---

## EXECUTIVE SUMMARY & STUDY APPROACH {#executive-summary}

### What This Guide Covers

This comprehensive guide prepares you for technical interviews at companies like:
- **FAANG Companies**: Apple, Google, Meta, Netflix, Amazon
- **AI-First Companies**: Anthropic, OpenAI, Cohere, Scale AI
- **Tech Unicorns**: Uber, Airbnb, LinkedIn
- **Enterprises**: Disney, Microsoft, Financial Services

### Key Interview Focus Areas (By Frequency)

Based on the job analysis, interviews will heavily focus on:

1. **Cloud Platforms (90% of roles)** - Deep expertise in AWS, GCP, or Azure for ML workloads
2. **Kubernetes (86% of roles)** - Advanced orchestration, operators, GPU scheduling
3. **Python (81% of roles)** - Infrastructure automation, ML frameworks integration
4. **Containerization (79% of roles)** - Docker, optimization, security
5. **Infrastructure as Code (64% of roles)** - Terraform, CloudFormation
6. **Model Serving (45% of roles)** - vLLM, Triton, TensorRT-LLM (HOT SKILL)
7. **MLOps Platforms (43% of roles)** - MLflow, Kubeflow, Ray

### How to Use This Guide

**For Different Experience Levels:**

- **Entry-Level (0-2 years)**: Focus on Sections 2.1-2.6, especially foundational questions
- **Mid-Level (3-5 years)**: Cover all of Section 2, 50% of Section 3, all of Section 4
- **Senior+ (5+ years)**: Master all sections, especially system design and architectural questions

**Recommended Study Approach:**

1. **Week 1-2**: Core infrastructure skills (Cloud, K8s, Docker, Python)
2. **Week 3-4**: MLOps tools and model serving platforms
3. **Week 5-6**: System design and hands-on scenarios
4. **Week 7-8**: Advanced topics, behavioral prep, mock interviews

---

## TECHNICAL INTERVIEW QUESTIONS BY SKILL DOMAIN {#technical-questions}

## 2.1 CLOUD PLATFORMS (AWS/GCP/Azure) FOR AI/ML WORKLOADS

**Market Context**: 90% of jobs require cloud expertise. AWS (67%) leads, followed by GCP (57%) and Azure (43%).

### Conceptual Foundation

**Definition**: Cloud platforms provide on-demand, scalable infrastructure for AI/ML workloads, including specialized services for model training, deployment, data processing, and GPU/TPU access. Understanding cloud architecture for ML means knowing how to optimize cost, performance, and reliability for compute-intensive, data-heavy AI systems.

**Why This Matters**:
- AI workloads require massive scale (thousands of GPUs, petabytes of data)
- Cloud elasticity enables cost-effective experimentation and production serving
- Managed ML services accelerate development but require deep understanding to optimize
- Multi-cloud and hybrid strategies are increasingly common (40% of enterprise roles)

**Key Prerequisites**:
- Understanding of cloud computing fundamentals (IaaS, PaaS, SaaS)
- Networking basics (VPCs, subnets, security groups)
- Storage concepts (object storage, block storage, data lakes)
- Basic ML workflow understanding (train, evaluate, deploy, monitor)

**Critical Terminology**:
- **GPU Instances**: EC2 P4/P5 (AWS), A2/A3 (GCP), NC-series (Azure)
- **Managed ML Services**: SageMaker, Vertex AI, Azure ML
- **Data Services**: S3, BigQuery, ADLS (Azure Data Lake Storage)
- **Container Services**: EKS, GKE, AKS
- **Spot Instances**: Discounted preemptible compute for training

### Industry Relevance & Rationale

**Real-World Applications**:
1. **Training Infrastructure**: Large-scale distributed training on GPU clusters
2. **Model Serving**: Auto-scaling inference endpoints for variable traffic
3. **Data Pipelines**: ETL workflows processing training data at scale
4. **Feature Engineering**: Real-time feature computation and serving
5. **Cost Optimization**: Spot instances for training, reserved for production

**Company Examples**:
- **Netflix**: Uses AWS extensively for feature infrastructure and model serving
- **Apple**: Hybrid cloud + on-premise for privacy-sensitive ML workloads
- **Uber**: Multi-region deployment on AWS for low-latency predictions
- **Cohere**: Multi-cloud (GCP/Azure/AWS/OCI) for LLM training flexibility

**Career Impact**:
- Cloud expertise commands 20-30% salary premium
- Multi-cloud skills are rare and highly valued
- Deep cloud ML service knowledge separates mid from senior engineers

### Pros & Cons Analysis

**Advantages**:
- **Elasticity**: Scale from 1 to 10,000 GPUs instantly
- **Managed Services**: Reduce operational burden (SageMaker, Vertex AI)
- **Global Reach**: Low-latency serving across regions
- **Cost Efficiency**: Pay-per-use, spot instances for training
- **Innovation**: Access to latest accelerators (H100s, TPUs)

**Limitations & Trade-offs**:
- **Vendor Lock-in**: Tight coupling to proprietary services
- **Cost Complexity**: Unpredictable bills without governance
- **Data Egress Costs**: Moving large datasets between clouds/on-prem expensive
- **Compliance**: Data residency requirements limit cloud options
- **Cold Start**: Serverless ML endpoints have latency penalties

**When to Use What**:
- **AWS SageMaker**: Comprehensive ML platform, best for AWS-native stacks
- **GCP Vertex AI**: Superior for TPU workloads, BigQuery integration
- **Azure ML**: Enterprise integration, strong compliance features
- **Multi-cloud**: Risk mitigation, cost optimization, avoiding lock-in

**Common Pitfalls**:
- Over-relying on managed services without understanding underlying infrastructure
- Not using spot instances for interruptible training workloads
- Ignoring data transfer costs between services/regions
- Poor resource tagging leading to cost allocation nightmares

### Interview Questions

#### Question 1: Foundational - Cloud ML Service Understanding

**Q**: Explain the difference between AWS SageMaker, EC2 with custom ML setup, and Lambda for ML inference. When would you choose each?

**Comprehensive Answer**:

"I'll compare these three approaches for running ML workloads on AWS:

**AWS SageMaker** is a fully managed ML platform that abstracts infrastructure:
- Provides managed Jupyter notebooks, training jobs, and hosted endpoints
- Handles auto-scaling, model versioning, and deployment
- Best for: Teams wanting rapid experimentation without infrastructure management, organizations standardizing on AWS ML tooling
- Limitations: Higher cost than DIY, less flexibility for custom architectures, potential vendor lock-in

**EC2 with Custom ML Setup** gives complete control:
- Spin up GPU instances (P4d, P5) and configure everything yourself
- Install PyTorch, CUDA, custom networking, etc.
- Best for: Custom distributed training setups, cost-sensitive workloads using spot instances, when you need specific hardware configurations
- Limitations: Higher operational burden, requires infrastructure expertise, slower to iterate

**Lambda for ML Inference** is serverless compute:
- Pay-per-invocation, auto-scales to zero
- Limited to 10GB memory, 15-minute timeout
- Best for: Infrequent inference requests, CPU-based small model inference, prototyping
- Limitations: No GPU support, cold start latency (can be 1-3 seconds), not suitable for real-time or batch inference

**My Decision Framework**:
- **Prototyping/Research**: SageMaker notebooks for fast iteration
- **Training Large Models**: EC2 spot instances with custom distributed training
- **Production Serving (high QPS)**: SageMaker endpoints or EKS with GPU nodes
- **Production Serving (sporadic)**: Lambda with CPU models or SageMaker Serverless
- **Cost-Sensitive Production**: EC2 reserved instances with auto-scaling"

**Key Points to Cover**:
- Managed vs. self-managed trade-offs
- Cost considerations for each approach
- Scalability and latency characteristics
- Appropriate use cases for each

**Code Example**: Setting up SageMaker vs. EC2
```python
# SageMaker Training Job (Managed)
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=sagemaker_role,
    instance_type='ml.p4d.24xlarge',
    instance_count=4,
    framework_version='2.0',
    py_version='py310',
    distribution={'pytorchddp': {'enabled': True}}
)
estimator.fit({'training': s3_training_path})

# EC2 Custom Setup (More Control)
# 1. Launch EC2 instances with GPU
# 2. Install CUDA, PyTorch manually
# 3. Set up distributed training with torchrun
# 4. Manage checkpointing to S3
# More work, but full control over configuration
```

**Common Mistakes**:
- Using Lambda for model inference without considering cold starts
- Not leveraging spot instances for training to save 70%+ costs
- Over-engineering with EC2 when SageMaker would suffice
- Not considering data transfer costs between services

**Excellence Indicators**:
- Mentioning specific instance types (P4d vs P5, A100 vs H100)
- Discussing cost optimization strategies (spot, reserved, savings plans)
- Bringing up real-world constraints (compliance, latency SLAs)
- Comparing with GCP/Azure equivalents

**Follow-up Discussion**:
- "How would you implement training checkpointing for spot instance interruptions?"
- "What metrics would you monitor for SageMaker endpoint auto-scaling?"
- "How do you handle model artifacts that exceed Lambda's deployment package size?"

---

#### Question 2: Intermediate - Multi-Region ML Deployment

**Q**: Design a multi-region deployment strategy for a real-time ML model serving system. The model needs to serve predictions with <100ms latency globally while minimizing costs.

**Comprehensive Answer**:

"I'll design a multi-region ML serving architecture balancing latency, availability, and cost:

**Architecture Components**:

1. **Regional Model Endpoints**:
   - Deploy model replicas in 4-5 AWS regions covering major geographies (us-east-1, eu-west-1, ap-southeast-1, us-west-2, sa-east-1)
   - Each region runs EKS cluster with GPU nodes for model serving
   - Use vLLM or Triton for efficient inference serving

2. **Global Traffic Routing**:
   - AWS Route 53 with latency-based routing to direct users to nearest region
   - CloudFront CDN for caching prediction results where applicable (for deterministic models with repeated queries)
   - Global Accelerator for TCP-level optimization when CDN caching not possible

3. **Model Synchronization**:
   - Store model artifacts in S3 with cross-region replication
   - Use ECR (Elastic Container Registry) replication for container images
   - Implement blue-green deployments per region with canary testing

4. **Data Locality & Compliance**:
   - Feature data stored regionally to comply with GDPR/data residency
   - Replicate critical features to multiple regions with DynamoDB Global Tables
   - Use Redis Cluster in each region for hot feature caching

5. **Cost Optimization**:
   - Primary regions on reserved instances, secondary on spot where acceptable
   - Auto-scaling based on regional traffic patterns (scale down APAC during US peak hours)
   - Share base infrastructure (EKS control plane) across multiple models

**Latency Optimization**:
- Keep feature data co-located with model endpoints
- Use model optimization (TensorRT, quantization) to reduce inference time
- Implement request batching with <50ms wait time
- Monitor p99 latency per region, trigger alerts >100ms

**Failure Handling**:
- Health checks route traffic away from degraded regions
- Fallback to nearest healthy region if local region fails
- Circuit breakers prevent cascading failures

**Monitoring**:
- Per-region latency dashboards (p50, p99, p99.9)
- Model accuracy drift monitoring per region
- Cost breakdown by region and model version"

**Key Points to Cover**:
- Geographic distribution strategy
- Traffic routing mechanisms
- Model consistency across regions
- Cost vs. latency trade-offs
- Failure scenarios and handling

**Code Example**: EKS Multi-Region Deployment with Terraform
```hcl
# Terraform for multi-region EKS model serving
module "ml_serving_us_east_1" {
  source = "./modules/ml-serving-cluster"
  region = "us-east-1"
  cluster_name = "ml-serving-use1"
  node_groups = {
    gpu_inference = {
      instance_types = ["g5.2xlarge"]
      desired_size = 4
      max_size = 20
      min_size = 2
    }
  }
  model_s3_bucket = "models-global"
}

module "ml_serving_eu_west_1" {
  source = "./modules/ml-serving-cluster"
  region = "eu-west-1"
  cluster_name = "ml-serving-euw1"
  # ... similar configuration
}

# Route 53 latency-based routing
resource "aws_route53_record" "ml_api_latency" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.mlservice.com"
  type    = "A"

  set_identifier = "us-east-1"
  latency_routing_policy {
    region = "us-east-1"
  }
  alias {
    name = module.ml_serving_us_east_1.lb_dns_name
    zone_id = module.ml_serving_us_east_1.lb_zone_id
    evaluate_target_health = true
  }
}
```

**Common Mistakes**:
- Not considering data residency and compliance requirements
- Underestimating cross-region data transfer costs
- Synchronous model updates causing downtime
- No fallback strategy when primary region fails

**Excellence Indicators**:
- Discussing specific AWS services (Route 53 latency routing, Global Accelerator)
- Mentioning cost optimization strategies (reserved + spot mix)
- Considering edge cases (regional failures, model version skew)
- Quantifying latency targets and trade-offs

**Follow-up Discussion**:
- "How would you handle a model rollback scenario across all regions?"
- "What if 80% of traffic comes from one region - how would you optimize?"
- "How do you ensure model version consistency during deployments?"

---

#### Question 3: Intermediate - GPU Instance Selection & Cost Optimization

**Q**: Your team needs to train a 7B parameter LLM. You have $50,000 budget for training. Walk me through how you'd select AWS instances, estimate training time, and optimize costs.

**Comprehensive Answer**:

"I'll approach this as a cost-constrained GPU infrastructure planning problem:

**Step 1: Understand Training Requirements**

For a 7B parameter LLM:
- Model size: ~14GB in fp16 (2 bytes per param)
- With gradients and optimizer states: ~56GB memory (4x model size with Adam)
- Typical training corpus: 1-2 trillion tokens
- Estimated FLOPs: ~6 * 7B * 2T tokens = 84e21 FLOPs

**Step 2: GPU Instance Analysis**

AWS GPU options for training:
- **p4d.24xlarge**: 8x A100 40GB, $32.77/hr on-demand, ~$9.83/hr spot (70% savings)
- **p4de.24xlarge**: 8x A100 80GB, $40.97/hr on-demand, ~$12.29/hr spot
- **p5.48xlarge**: 8x H100, $98.32/hr on-demand, limited spot availability

**Step 3: Configuration Decision**

For 7B model with 56GB requirements:
- A100 40GB: Would need FSDP or pipeline parallelism across GPUs
- A100 80GB: Can fit model + optimizer on single GPU, easier scaling
- H100: Faster but 3x cost, only worthwhile if time-critical

**I'd choose p4de.24xlarge with spot instances**:
- Each instance has 8x A100 80GB GPUs
- Can fit full model on single GPU, use remaining for data parallelism
- Spot price ~$12.29/hr vs. $40.97 on-demand

**Step 4: Scaling & Time Estimation**

Using 4x p4de.24xlarge instances (32 GPUs total):
- Aggregate throughput: ~32 * 80 TFLOPs (A100 fp16) = 2,560 TFLOPs/s
- Training time: 84e21 FLOPs / (2560e12 FLOPs/s) / 0.5 (efficiency) ≈ 18 days
- Spot cost: 4 instances * $12.29/hr * 24hr * 18 days = $21,230

**Step 5: Cost Optimization Strategies**

1. **Spot Instances**: 70% savings, critical for budget
   - Implement checkpointing every 1 hour
   - Use `boto3` to handle spot interruptions gracefully
   - Spread across multiple AZs to reduce interruption risk

2. **Mixed Precision Training**: Use bf16 instead of fp32
   - Reduces memory, increases throughput by 2x
   - Already factored into calculations above

3. **Gradient Accumulation**: If memory tight
   - Accumulate over 4-8 micro-batches before weight update
   - Maintain effective batch size while reducing per-GPU memory

4. **Checkpointing Strategy**:
   - Save to S3 every hour (~$0.023/GB/month storage)
   - Keep last 3 checkpoints, delete older ones
   - Checkpoint size ~56GB, minimal cost

5. **Data Loading Optimization**:
   - Stream data from S3 with prefetching
   - Use FSx for Lustre for high-throughput shared filesystem across instances
   - Cost: ~$600/month for 1.2TB Lustre

**Revised Budget**:
- Compute (spot): $21,230
- Storage (S3 for data + checkpoints): ~$500
- Networking (inter-AZ transfer): ~$1,000
- FSx Lustre: ~$300 (for 18 days)
- **Total: ~$23,030** - well under $50K budget

**Contingency Planning**:
- Reserve $5,000 for spot interruption overhead (might need on-demand failover)
- Remaining $22K could accelerate training with more instances or H100s if needed

**Monitoring & Iteration**:
- Track GPU utilization (should be >90%)
- Monitor tokens/second/GPU
- If spot interruptions >20%, blend in some on-demand instances"

**Key Points to Cover**:
- GPU instance types and characteristics
- Cost calculation methodology
- Spot vs. on-demand trade-offs
- Training time estimation
- Checkpointing and interruption handling

**Code Example**: Spot Instance Management
```python
# Spot instance request with interruption handling
import boto3
from sagemaker.pytorch import PyTorch

# Configure training with spot instances
estimator = PyTorch(
    entry_point='train_llm.py',
    role=role,
    instance_type='ml.p4de.24xlarge',
    instance_count=4,
    max_run=86400 * 20,  # 20 days max
    use_spot_instances=True,
    max_wait=86400 * 25,  # Wait up to 25 days for spot
    checkpoint_s3_uri='s3://my-bucket/checkpoints/',
    checkpoint_local_path='/opt/ml/checkpoints',
)

# Training script with checkpointing
# train_llm.py
def train():
    # Load checkpoint if exists
    checkpoint_path = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
    if os.path.exists(f'{checkpoint_path}/latest.pt'):
        checkpoint = torch.load(f'{checkpoint_path}/latest.pt')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']

    for step in range(start_step, total_steps):
        # Training loop

        # Save checkpoint every hour
        if step % checkpoint_interval == 0:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'step': step
            }, f'{checkpoint_path}/latest.pt')
```

**Common Mistakes**:
- Using on-demand instances without considering spot
- Not accounting for optimizer states in memory calculations
- Underestimating data transfer costs
- No checkpointing strategy for spot interruptions
- Choosing H100s without cost-benefit analysis

**Excellence Indicators**:
- Detailed cost calculations with actual AWS pricing
- Understanding memory requirements (model + gradients + optimizer states)
- Mentioning specific optimization techniques (FSDP, mixed precision)
- Spot instance best practices (checkpointing, multi-AZ)
- Considering alternatives (H100 vs A100, 40GB vs 80GB)

**Follow-up Discussion**:
- "What if training needs to finish in 5 days instead of 18 - how would you adjust?"
- "How would you monitor training efficiency to detect issues early?"
- "What if spot availability is poor - what's your fallback plan?"

---

#### Question 4: Advanced - Hybrid Cloud ML Infrastructure

**Q**: A financial services company needs to deploy ML models but has regulatory requirements to keep sensitive data on-premise. Design a hybrid cloud/on-premise architecture for model training (on-prem) and serving (cloud).

**Comprehensive Answer**:

"This is a classic hybrid cloud architecture challenge balancing compliance with cloud benefits:

**Requirements Analysis**:
- **Training**: Must be on-premise (data residency, compliance)
- **Serving**: Can be in cloud (predictions don't expose PII)
- **Challenges**: Model transfer, data pipeline orchestration, unified monitoring

**Architecture Design**:

**1. On-Premise Training Infrastructure**:
- **Compute**: GPU cluster (NVIDIA DGX systems or custom servers)
- **Storage**: High-performance NFS or distributed filesystem (Ceph, Lustre)
- **Orchestration**: Self-managed Kubernetes or OpenShift
- **MLOps**: MLflow for experiment tracking, DVC for data versioning
- **Security**: Air-gapped or limited egress through secure gateway

**2. Cloud Serving Infrastructure**:
- **Platform**: AWS EKS or GCP GKE for model serving
- **Endpoints**: Auto-scaling inference endpoints with vLLM/Triton
- **CDN**: CloudFront for global low-latency access
- **Monitoring**: CloudWatch, Datadog for observability

**3. Hybrid Connectivity & Model Transfer**:
- **Network**: AWS Direct Connect or Azure ExpressRoute (dedicated 10Gbps+ link)
- **Security**: VPN overlay with encryption, network segmentation
- **Model Registry**: Hybrid model registry (MLflow with S3 backend, synced copy on-prem)
- **Artifact Transfer**:
  - Automated pipeline: On-prem CI/CD pushes model artifacts to cloud S3
  - Transfer through secure bastion/gateway with audit logging
  - Model scanning for security before cloud deployment

**4. Data Handling Strategy**:
- **Training Data**: Stays entirely on-premise, never touches cloud
- **Feature Engineering**: Run on-premise, only export aggregated features if needed
- **Inference Data**: Cloud-based applications send requests to cloud endpoints
- **Prediction Storage**: Can store in cloud (no PII linkage)

**5. Unified Workflow Orchestration**:
- **Control Plane**: Apache Airflow running on-premise
- **Workflow**:
  1. Data ingestion (on-prem)
  2. Feature engineering (on-prem)
  3. Model training (on-prem K8s)
  4. Model validation (on-prem)
  5. Model artifact push to cloud (secure gateway)
  6. Cloud deployment via CI/CD (GitHub Actions + ArgoCD)
  7. Canary testing in cloud
  8. Production rollout

**6. Monitoring & Observability**:
- **Metrics Collection**: Prometheus on-prem and in cloud
- **Centralized Dashboards**: Grafana with federated query across environments
- **Log Aggregation**: Splunk or ELK with secure log forwarding
- **Alerting**: PagerDuty integrated with both environments

**Security & Compliance**:
- **Access Control**: Separate IAM/RBAC for on-prem vs. cloud
- **Data Classification**: Clear labels (sensitive vs. non-sensitive)
- **Audit Logging**: All model transfers logged and reviewable
- **Encryption**: TLS for transfer, encryption at rest both sides
- **Compliance Validation**: Regular audits, model lineage tracking

**Cost Considerations**:
- **On-Prem Capex**: High upfront for GPU servers ($100K-500K per DGX)
- **Cloud Opex**: Variable based on serving load ($5K-50K/month)
- **Network**: Direct Connect ~$2K/month + data transfer fees
- **Trade-off**: Higher on-prem costs justified by compliance requirements

**Failure Scenarios**:
- **Network Failure**: Model serving continues in cloud, training queued on-prem
- **On-Prem Outage**: No new models deployed, existing cloud models serve
- **Cloud Outage**: Switch to backup cloud region or temporary on-prem serving

**Diagram**:
```
[On-Premise]                    [Secure Gateway]           [AWS Cloud]
┌─────────────────┐            ┌──────────────┐           ┌────────────────┐
│ Training Data   │            │              │           │                │
│ (Sensitive PII) │            │  Encrypted   │           │ Model Registry │
└────────┬────────┘            │  Transfer    │           │ (S3 + MLflow)  │
         │                     │  Audit Logs  │           └───────┬────────┘
         ↓                     │              │                   │
┌─────────────────┐            └───────┬──────┘                   ↓
│ Feature Eng.    │                    │                  ┌────────────────┐
│ (Spark/Airflow) │                    │                  │ EKS Cluster    │
└────────┬────────┘                    │                  │ (vLLM/Triton)  │
         │                             │                  └───────┬────────┘
         ↓                             │                          │
┌─────────────────┐                    │                          ↓
│ K8s Training    │                    │                  ┌────────────────┐
│ (PyTorch DDP)   │                    │                  │ API Gateway    │
└────────┬────────┘                    │                  │ + CloudFront   │
         │                             │                  └───────┬────────┘
         ↓                             │                          │
┌─────────────────┐                    │                          ↓
│ Model Artifacts │───────────────────►│─────────────────►   [External Users]
│ (Validated)     │                    │
└─────────────────┘                    │
         │                             │
         ↓                             │
┌─────────────────┐                    │
│ MLflow Registry │◄───────────────────┤ (Sync)
│ (On-Prem)       │                    │
└─────────────────┘                    │
```

**Implementation Phases**:
1. **Phase 1**: Set up Direct Connect, establish secure model transfer pipeline
2. **Phase 2**: Deploy cloud serving infrastructure, test with dummy models
3. **Phase 3**: Integrate on-prem training with cloud deployment
4. **Phase 4**: Implement unified monitoring and alerting
5. **Phase 5**: Automate entire workflow, document runbooks"

**Key Points to Cover**:
- Clear separation of sensitive data (on-prem) from serving (cloud)
- Secure model artifact transfer mechanisms
- Compliance and audit requirements
- Network connectivity options
- Unified orchestration and monitoring

**Code Example**: Secure Model Transfer Pipeline
```python
# Airflow DAG for hybrid model deployment
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator
import boto3
import hashlib

def validate_model():
    """Run security scans and validation before transfer"""
    model_path = '/nfs/models/latest/model.pt'

    # Scan for embedded secrets or PII
    with open(model_path, 'rb') as f:
        content = f.read()
        # Run security scans

    # Validate model structure
    import torch
    model = torch.load(model_path)
    assert 'state_dict' in model

    # Compute checksum for integrity
    sha256 = hashlib.sha256(content).hexdigest()
    with open(f'{model_path}.sha256', 'w') as f:
        f.write(sha256)

    return True

def trigger_cloud_deployment():
    """Trigger cloud CI/CD pipeline after model upload"""
    # Call GitHub Actions workflow or Jenkins job
    import requests
    response = requests.post(
        'https://api.github.com/repos/myorg/ml-serving/actions/workflows/deploy.yml/dispatches',
        headers={'Authorization': f'token {GITHUB_TOKEN}'},
        json={'ref': 'main', 'inputs': {'model_version': 'v1.2.3'}}
    )
    return response.status_code == 204

with DAG('hybrid_model_deployment', schedule_interval='@daily') as dag:
    validate = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model
    )

    transfer = LocalFilesystemToS3Operator(
        task_id='transfer_to_s3',
        filename='/nfs/models/latest/model.pt',
        dest_key='models/production/{{ ds }}/model.pt',
        dest_bucket='ml-models-prod',
        aws_conn_id='aws_secure_gateway'
    )

    deploy = PythonOperator(
        task_id='trigger_cloud_deployment',
        python_callable=trigger_cloud_deployment
    )

    validate >> transfer >> deploy
```

**Common Mistakes**:
- Assuming all data can move to cloud without compliance review
- Underestimating network bandwidth requirements for large models
- No audit trail for model artifacts crossing security boundary
- Not having fallback when cloud is unavailable
- Ignoring latency of cross-boundary operations

**Excellence Indicators**:
- Specific network connectivity options (Direct Connect, ExpressRoute)
- Discussing compliance frameworks (SOC2, HIPAA, GDPR)
- Detailed security controls (encryption, audit logging, access control)
- Cost breakdown for hybrid architecture
- Mentioning real-world constraints (network latency, data transfer limits)

**Follow-up Discussion**:
- "How would you handle model rollback in this hybrid setup?"
- "What if regulators require ability to audit all predictions - how would you architect this?"
- "How do you ensure on-prem and cloud environments stay in sync (libraries, configs)?"

---

#### Question 5: Advanced - Cloud Cost Optimization for ML Workloads

**Q**: Your company's AWS ML infrastructure costs $200K/month. Leadership wants 40% cost reduction without impacting model quality or latency. How would you approach this?

**Comprehensive Answer**:

"This is a cost optimization challenge requiring systematic analysis and strategic changes:

**Step 1: Cost Analysis & Baseline**

First, I'd break down the $200K into categories:
```
Assumed breakdown (typical for ML workloads):
- Compute (EC2/EKS GPU instances): $120K (60%)
  - Training: $80K
  - Inference: $40K
- Storage (S3, EBS, EFS): $30K (15%)
- Data Transfer: $25K (12.5%)
- Managed Services (SageMaker, RDS, etc.): $20K (10%)
- Other (NAT gateways, load balancers): $5K (2.5%)
```

Target: Reduce by $80K (40%) to $120K/month

**Step 2: Quick Wins (Month 1) - Target $30K savings**

1. **Compute Reservations & Savings Plans** ($15K savings):
   - Analyze steady-state compute usage over past 90 days
   - Purchase 1-year Compute Savings Plans for predictable workloads
   - For training GPU instances used 24/7, savings plans give ~40% discount
   - Example: $50K/month stable compute → $20K/month after commitment

2. **Spot Instances for Training** ($10K savings):
   - Move all training workloads to spot instances (70% savings)
   - Implement robust checkpointing (already discussed)
   - Current: $80K on-demand training → $24K with spot + $6K interruption overhead

3. **Right-Sizing Instances** ($5K savings):
   - Audit instance utilization (CloudWatch metrics)
   - Downsize over-provisioned instances (e.g., CPU instances running at 15% util)
   - Example: Move from m5.4xlarge to m5.2xlarge where appropriate

**Step 3: Medium-Term Optimizations (Month 2-3) - Target $30K savings**

4. **Inference Optimization** ($12K savings):
   - Implement model quantization (int8) for 2x throughput on same hardware
   - Batch inference requests with dynamic batching (vLLM, Triton)
   - Example: 10x g5.2xlarge → 5x g5.2xlarge with batching
   - Savings: $40K → $20K inference costs, net $8K after optimization work

5. **Auto-Scaling Tuning** ($8K savings):
   - Review current auto-scaling policies
   - Implement predictive scaling based on historical patterns
   - Scale down non-production environments outside business hours
   - Use Fargate Spot for non-critical inference workloads

6. **Storage Optimization** ($10K savings):
   - Implement S3 Intelligent-Tiering for training data
   - Move old model artifacts to Glacier ($0.004/GB vs $0.023/GB for S3 Standard)
   - Clean up unused EBS volumes and snapshots (often 20-30% waste)
   - Compress datasets where possible
   - Example: $30K storage → $20K after lifecycle policies

**Step 4: Architectural Changes (Month 4-6) - Target $20K savings**

7. **Data Transfer Reduction** ($12K savings):
   - Identify cross-region, cross-AZ, and egress traffic
   - Co-locate compute with data (same AZ when possible)
   - Use VPC endpoints for S3 (free vs NAT gateway egress)
   - Cache frequent requests at edge with CloudFront
   - Example: $25K data transfer → $13K after optimization

8. **Managed Service Alternatives** ($8K savings):
   - Evaluate if SageMaker is necessary for all workloads
   - For simple inference, migrate to EKS with Triton (30-40% cheaper)
   - Use self-managed MLflow instead of SageMaker Feature Store where applicable
   - Keep SageMaker only for critical production workloads

**Step 5: Monitoring & Governance**

Implement ongoing cost controls:
- Tag all resources (team, project, environment) for accountability
- Set up AWS Budgets with alerts at 80%, 90%, 100% thresholds
- Weekly cost review meetings with team leads
- Automate shutdown of idle dev/test environments
- Implement cost dashboards in Grafana

**Implementation Plan**:

```
Month 1: Quick Wins
- Week 1: Analyze costs, identify top 10 cost drivers
- Week 2: Purchase savings plans, migrate training to spot
- Week 3: Right-size instances, clean up waste
- Week 4: Validate savings, document changes
Expected savings: $30K/month

Month 2-3: Medium Optimizations
- Implement model quantization and batching
- Tune auto-scaling policies
- Set up S3 lifecycle policies
Expected savings: $30K/month (cumulative $60K)

Month 4-6: Architectural
- Re-architect data flows to reduce transfer
- Migrate from managed services where appropriate
- Implement governance and automation
Expected savings: $20K/month (cumulative $80K)

Total: $80K/month savings = 40% reduction
```

**Risk Mitigation**:
- **Spot Interruptions**: Keep 10% on-demand capacity as failover
- **Performance Regression**: A/B test all optimizations before full rollout
- **Latency Impact**: Monitor p99 latency, rollback if degrades >10%
- **Model Quality**: Re-validate quantized models, ensure accuracy loss <1%

**Metrics to Track**:
- Cost per inference ($/prediction)
- Cost per training job ($/model)
- GPU utilization percentage (target >85%)
- Spot interruption rate (<5% acceptable)
- Latency p99 (maintain <100ms SLA)

**Tools**:
- AWS Cost Explorer for trend analysis
- CloudHealth or Kubecost for multi-dimensional cost allocation
- Grafana dashboards for real-time cost tracking
- Terraform for infrastructure-as-code changes"

**Key Points to Cover**:
- Systematic cost analysis methodology
- Multiple optimization strategies (quick wins + long-term)
- Quantified savings for each initiative
- Risk considerations and mitigation
- Implementation timeline

**Code Example**: Cost Monitoring & Auto-Shutdown
```python
# Lambda function to auto-shutdown idle ML instances
import boto3
from datetime import datetime, timedelta

ec2 = boto3.client('ec2')
cloudwatch = boto3.client('cloudwatch')

def lambda_handler(event, context):
    """Shut down EC2 instances with low GPU utilization"""

    instances = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Environment', 'Values': ['dev', 'staging']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )

    for reservation in instances['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']

            # Check GPU utilization over last 24 hours
            response = cloudwatch.get_metric_statistics(
                Namespace='CWAgent',
                MetricName='nvidia_gpu_utilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=datetime.utcnow() - timedelta(days=1),
                EndTime=datetime.utcnow(),
                Period=3600,
                Statistics=['Average']
            )

            if response['Datapoints']:
                avg_util = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])

                if avg_util < 10:  # Less than 10% GPU utilization
                    print(f"Stopping idle instance {instance_id} (avg GPU util: {avg_util:.1f}%)")
                    ec2.stop_instances(InstanceIds=[instance_id])

                    # Notify team
                    sns = boto3.client('sns')
                    sns.publish(
                        TopicArn='arn:aws:sns:us-east-1:123456789:ml-cost-alerts',
                        Subject=f'Stopped idle GPU instance {instance_id}',
                        Message=f'Instance {instance_id} stopped due to low utilization ({avg_util:.1f}%)'
                    )

# Terraform for budget alerts
resource "aws_budgets_budget" "ml_infrastructure" {
  name              = "ml-infrastructure-monthly"
  budget_type       = "COST"
  limit_amount      = "120000"
  limit_unit        = "USD"
  time_period_start = "2025-01-01_00:00"
  time_unit         = "MONTHLY"

  cost_filter {
    name = "TagKeyValue"
    values = ["team$ml-platform"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["ml-team@company.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = ["cto@company.com"]
  }
}
```

**Common Mistakes**:
- Cutting costs without measuring impact on latency/quality
- Not having rollback plan when optimizations cause issues
- Focusing only on compute, ignoring storage and network costs
- Making ad-hoc changes without governance framework
- Over-relying on spot instances without proper interruption handling

**Excellence Indicators**:
- Specific cost breakdown and quantified savings per initiative
- Phased implementation plan with milestones
- Risk analysis and mitigation strategies
- Mentioning specific AWS cost optimization tools
- Understanding trade-offs (cost vs performance vs reliability)

**Follow-up Discussion**:
- "How would you convince engineering teams to adopt spot instances if they're resistant?"
- "What if your optimizations cause a production incident - how do you balance cost vs reliability?"
- "How would you measure the success of your cost optimization program?"

---

### Key Takeaways: Cloud Platforms for AI/ML

**Must-Know Concepts**:
1. GPU instance types and selection criteria (A100 vs H100, spot vs on-demand)
2. Multi-region deployment strategies for ML serving
3. Cost optimization techniques (spot, reserved, savings plans)
4. Hybrid cloud architectures for compliance
5. Managed ML services (SageMaker, Vertex AI) vs DIY on EC2/GKE

**Interview Success Factors**:
- Quantify everything (costs, latency, throughput)
- Discuss trade-offs explicitly (cost vs performance vs complexity)
- Show awareness of real-world constraints (compliance, budget, time)
- Demonstrate hands-on experience with specific services

**Common Interview Red Flags**:
- Generic cloud knowledge without ML-specific context
- No awareness of GPU instance options
- Can't estimate costs for realistic scenarios
- Doesn't mention spot instances for training
- No consideration of compliance/security

---

## 2.2 KUBERNETES & CONTAINER ORCHESTRATION FOR ML SYSTEMS

**Market Context**: 86% of jobs require Kubernetes expertise. This is the single most critical infrastructure skill for AI/ML roles.

### Conceptual Foundation

**Definition**: Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications. For ML workloads, K8s provides GPU scheduling, distributed training coordination, auto-scaling inference endpoints, and resource isolation across teams and projects.

**Why This Matters for ML**:
- ML workloads have unique requirements: GPU access, long-running training jobs, variable compute needs
- Multi-tenancy: Data scientists, researchers, and production systems share same infrastructure
- Kubernetes operators enable declarative ML pipelines (Kubeflow, Ray, TorchServe)
- Cloud-agnostic: Same K8s YAML works on EKS, GKE, AKS, on-premise

**Key Prerequisites**:
- Container fundamentals (Docker images, registries, layers)
- Linux basics (processes, namespaces, cgroups)
- Networking concepts (DNS, load balancing, ingress)
- YAML configuration syntax

**Critical Terminology for ML on Kubernetes**:
- **GPU Scheduling**: Allocating GPUs to pods via device plugins
- **Job/CronJob**: Batch workloads for training
- **Deployment/StatefulSet**: Serving workloads
- **HPA (Horizontal Pod Autoscaler)**: Scale pods based on CPU/memory/custom metrics
- **Operators**: Custom controllers for ML frameworks (TorchX, KubeFlow)
- **Resource Quotas**: Limit GPU/CPU/memory per namespace (team isolation)
- **Taints & Tolerations**: Reserve GPU nodes for specific workloads
- **NVIDIA Device Plugin**: Exposes GPUs to K8s scheduler

### Industry Relevance & Rationale

**Real-World Applications**:
1. **Model Training**: Launch distributed PyTorch DDP jobs across multiple GPU nodes
2. **Model Serving**: Deploy auto-scaling inference endpoints with load balancing
3. **Batch Inference**: Process large datasets with parallel jobs
4. **Experiment Tracking**: Run Jupyter notebooks for data scientists
5. **Resource Management**: Fair-share GPU access across teams

**Company Examples**:
- **Uber**: Michelangelo platform runs on K8s for model training and serving
- **Airbnb**: Bighead ML platform uses K8s with Spark and Ray
- **Apple**: Custom K8s clusters with Apache YuniKorn for ML compute scheduling
- **Cohere**: Self-managed K8s on GCP for training 100B+ parameter LLMs

**Career Impact**:
- K8s expertise is non-negotiable for 86% of roles
- Advanced K8s skills (operators, schedulers) command senior-level comp
- Kubernetes troubleshooting is a key interview differentiator

### Pros & Cons Analysis

**Advantages**:
- **Standardization**: Same APIs across clouds and on-premise
- **Scalability**: Easily scale from 1 to 1000s of nodes
- **Resource Efficiency**: Bin-packing, auto-scaling, spot instance integration
- **Ecosystem**: Rich ML tools (Kubeflow, MLflow, Ray, Seldon) built on K8s
- **Multi-Tenancy**: Namespace isolation, resource quotas, RBAC

**Limitations & Trade-offs**:
- **Complexity**: Steep learning curve, many moving parts
- **Overhead**: Control plane resource costs, operational burden
- **GPU Scheduling**: Less mature than CPU, limited topologies (no MIG support in vanilla K8s)
- **Stateful Workloads**: Training jobs with checkpointing require careful design
- **Debugging**: Troubleshooting pod issues harder than VMs

**When to Use Kubernetes for ML**:
- Multi-team environments needing resource isolation
- Both training and serving workloads on same infrastructure
- Scaling from 10s to 100s of experiments simultaneously
- When cloud portability matters

**When NOT to Use K8s**:
- Single-user research environments (VMs simpler)
- Extremely large-scale single-tenant training (bare metal may be better)
- Org lacks K8s expertise and can't hire/train

**Common Pitfalls**:
- Not reserving GPU nodes for ML workloads (CPU pods steal GPUs)
- Insufficient resource requests/limits causing OOM kills
- No persistent storage for checkpoints (loses hours of training)
- Not understanding pod eviction and preemption policies

### Interview Questions

#### Question 6: Foundational - K8s Basics for ML Workloads

**Q**: Explain the difference between a Kubernetes Job, Deployment, and StatefulSet. When would you use each for ML workloads?

**Comprehensive Answer**:

"These are three core Kubernetes workload types, each suited for different ML use cases:

**Job**:
- **Purpose**: Runs a task to completion, then terminates
- **Characteristics**:
  - Pods run until success (exit code 0)
  - Can configure retries, parallelism, completions
  - Pods are not restarted after successful completion
- **ML Use Cases**:
  - **Model Training**: PyTorch training job that runs for 24 hours then completes
  - **Batch Inference**: Process 1M images, then finish
  - **Data Processing**: ETL pipeline, feature engineering job
  - **Hyperparameter Tuning**: Run 100 parallel experiments
- **Example**: Train a model with PyTorch DDP across 4 GPUs
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-resnet50
spec:
  parallelism: 4  # 4 pods (one per GPU)
  completions: 4
  template:
    spec:
      containers:
      - name: pytorch-training
        image: pytorch/pytorch:2.0-cuda11.8
        command: [\"torchrun\", \"--nproc_per_node=1\", \"train.py\"]
        resources:
          limits:
            nvidia.com/gpu: 1
      restartPolicy: OnFailure
```

**Deployment**:
- **Purpose**: Manages long-running, stateless services
- **Characteristics**:
  - Maintains desired number of pod replicas
  - Supports rolling updates and rollbacks
  - Pods are interchangeable (any pod can serve any request)
  - Automatically reschedules pods if nodes fail
- **ML Use Cases**:
  - **Model Serving**: REST API endpoint serving predictions
  - **Stateless Inference**: Each request is independent
  - **API Gateways**: Routing layer for multiple models
- **Example**: Deploy a model serving API with 5 replicas
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: model-serving
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.04-py3
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: \"16Gi\"
            cpu: \"4\"
```

**StatefulSet**:
- **Purpose**: Manages stateful applications with stable identities
- **Characteristics**:
  - Pods have stable, unique network identifiers
  - Ordered deployment and scaling (pod-0, then pod-1, etc.)
  - Each pod gets dedicated persistent storage
  - Pods are NOT interchangeable
- **ML Use Cases**:
  - **Distributed Training Coordinators**: When nodes need specific roles (rank 0, rank 1)
  - **Feature Stores**: Feast online serving with persistent state
  - **Model Registries**: MLflow with persistent database
  - **Vector Databases**: Milvus, Weaviate with persistent indexes
- **Example**: Deploy a distributed feature store
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: feature-store
spec:
  serviceName: \"feature-store\"
  replicas: 3
  selector:
    matchLabels:
      app: feast
  template:
    spec:
      containers:
      - name: feast-server
        image: feast/feature-server:latest
        volumeMounts:
        - name: feast-data
          mountPath: /feast/data
  volumeClaimTemplates:
  - metadata:
      name: feast-data
    spec:
      accessModes: [ \"ReadWriteOnce\" ]
      resources:
        requests:
          storage: 100Gi
```

**Decision Matrix**:

| Workload | Job | Deployment | StatefulSet |
|----------|-----|------------|-------------|
| Model Training | ✓ (batch) | ✗ | △ (if needs stable identity) |
| Model Serving | ✗ | ✓ (stateless) | △ (if stateful like A/B testing) |
| Batch Inference | ✓ | ✗ | ✗ |
| Feature Store | ✗ | ✗ | ✓ (needs persistence) |
| API Gateway | ✗ | ✓ | ✗ |
| Jupyter Notebooks | ✗ | ✓ or StatefulSet | ✓ (if user state matters) |

**Key Differences**:
- **Lifecycle**: Job (finite), Deployment/StatefulSet (continuous)
- **Identity**: Job/Deployment (ephemeral), StatefulSet (stable)
- **Storage**: Job/Deployment (optional), StatefulSet (persistent per pod)
- **Order**: Job/Deployment (unordered), StatefulSet (ordered scaling)"

**Key Points to Cover**:
- Purpose and characteristics of each resource type
- ML-specific use cases for each
- When to choose one over another
- YAML examples demonstrating key differences

**Common Mistakes**:
- Using Deployment for training jobs (pods restart on completion wastefully)
- Using Job for serving (doesn't provide continuous availability)
- Not understanding StatefulSet ordering implications
- Confusing DaemonSet with other types

**Excellence Indicators**:
- Mentioning specific ML use cases for each type
- Understanding pod identity and lifecycle differences
- Discussing persistent storage implications
- Showing YAML examples with GPU requests

**Follow-up Discussion**:
- "What happens if a training Job pod fails midway - how would you handle checkpointing?"
- "For StatefulSet, how do you perform rolling updates without downtime?"
- "How would you choose between Job with parallelism vs. Deployment for batch inference?"

---

#### Question 7: Intermediate - GPU Scheduling & Node Affinity

**Q**: You have a Kubernetes cluster with 10 GPU nodes (8x A100 GPUs each) and 50 CPU nodes. Design a strategy to ensure ML training jobs get GPU access while preventing CPU-only workloads from blocking GPU nodes.

**Comprehensive Answer**:

"This is a critical resource management problem in multi-tenant ML clusters. I'll design a strategy using taints, tolerations, node labels, and resource quotas:

**Step 1: Node Labeling & Organization**

First, label GPU nodes to identify their capabilities:
```bash
# Label GPU nodes
kubectl label nodes gpu-node-{1..10} \
  node-type=gpu \
  gpu-model=a100 \
  gpu-count=8 \
  zone=us-east-1a

# Label CPU nodes
kubectl label nodes cpu-node-{1..50} \
  node-type=cpu \
  zone=us-east-1a
```

**Step 2: Taint GPU Nodes**

Apply taints to prevent non-GPU workloads from scheduling on GPU nodes:
```bash
# Taint all GPU nodes
kubectl taint nodes -l node-type=gpu \
  nvidia.com/gpu=present:NoSchedule
```

This means: Only pods with matching toleration can schedule on these nodes.

**Step 3: Configure GPU Device Plugin**

Ensure NVIDIA device plugin is running to expose GPUs:
```yaml
# NVIDIA Device Plugin DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      nodeSelector:
        node-type: gpu
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.14.0
        name: nvidia-device-plugin-ctr
        env:
        - name: FAIL_ON_INIT_ERROR
          value: \"false\"
```

**Step 4: Training Job Configuration**

ML training jobs request GPUs and tolerate the taint:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training
spec:
  template:
    spec:
      # Tolerate GPU node taint
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      # Prefer nodes with specific GPU
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: gpu-model
                operator: In
                values: [\"a100\"]

      containers:
      - name: training
        image: pytorch/pytorch:2.0-cuda11.8
        resources:
          limits:
            nvidia.com/gpu: 8  # Request all 8 GPUs on node
          requests:
            nvidia.com/gpu: 8
            memory: \"400Gi\"
            cpu: \"64\"

      # Ensure all 8 GPUs are on same node
      nodeSelector:
        gpu-count: \"8\"
```

**Step 5: Resource Quotas per Team**

Implement namespace-based resource quotas to prevent one team from monopolizing GPUs:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-nlp
spec:
  hard:
    requests.nvidia.com/gpu: \"32\"  # Max 32 GPUs (4 nodes)
    limits.nvidia.com/gpu: \"32\"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-cv
spec:
  hard:
    requests.nvidia.com/gpu: \"24\"  # Max 24 GPUs (3 nodes)
    limits.nvidia.com/gpu: \"24\"
```

**Step 6: Priority Classes**

Create priority classes for different workload types:
```yaml
# High priority for production inference
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: production-high
value: 1000000
globalDefault: false
description: \"Production model serving\"
---
# Medium priority for training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-medium
value: 500000
description: \"ML training jobs\"
---
# Low priority for experiments
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: experiment-low
value: 100000
description: \"Experimental workloads\"
```

Use in pods:
```yaml
spec:
  priorityClassName: training-medium
```

**Step 7: Node Affinity for CPU Workloads**

Ensure CPU workloads prefer CPU nodes:
```yaml
# CPU-only deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-api
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node-type
                operator: In
                values: [\"cpu\"]
      containers:
      - name: api
        resources:
          requests:
            cpu: \"2\"
            memory: \"4Gi\"
```

**Step 8: Monitoring & Alerting**

Monitor GPU utilization and scheduling:
```yaml
# Prometheus queries
# GPU utilization per node
avg by (node) (DCGM_FI_DEV_GPU_UTIL)

# Pending pods waiting for GPUs
sum(kube_pod_status_phase{phase=\"Pending\"} * on(pod) group_left() kube_pod_container_resource_requests{resource=\"nvidia_com_gpu\"})

# GPU allocation by namespace
sum by (namespace) (kube_pod_container_resource_requests{resource=\"nvidia_com_gpu\"})
```

**Alternative: GPU Sharing**

For smaller models, enable GPU sharing with MIG (Multi-Instance GPU):
```yaml
# Enable MIG on A100 nodes (7 slices per GPU)
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-mig-config
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.5gb:
        - devices: all
          mig-devices:
            \"1g.5gb\": 7  # 7 small slices per A100
```

**Complete Strategy Summary**:

1. **Taints on GPU nodes** → Block non-GPU workloads
2. **Tolerations in ML jobs** → Explicitly allow GPU access
3. **Resource quotas** → Fair share per team
4. **Priority classes** → Production > Training > Experiments
5. **Node affinity** → Guide CPU workloads to CPU nodes
6. **Monitoring** → Track utilization and alert on underutilization

**Results**:
- GPU nodes: 95%+ GPU utilization, 0% CPU-only pods
- CPU nodes: Host all non-GPU workloads
- Fair GPU access across teams via quotas
- Production workloads preempt experiments when needed"

**Key Points to Cover**:
- Taints and tolerations mechanism
- Node labeling for organization
- Resource quotas for multi-tenancy
- Priority classes for workload prioritization
- Monitoring GPU utilization

**Code Example**: Complete GPU Training Job
```python
# Python script to generate and submit GPU job
from kubernetes import client, config

def create_gpu_training_job(name, image, gpus=8):
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    job = client.V1Job(
        api_version=\"batch/v1\",
        kind=\"Job\",
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    tolerations=[
                        client.V1Toleration(
                            key=\"nvidia.com/gpu\",
                            operator=\"Exists\",
                            effect=\"NoSchedule\"
                        )
                    ],
                    affinity=client.V1Affinity(
                        node_affinity=client.V1NodeAffinity(
                            required_during_scheduling_ignored_during_execution=
                                client.V1NodeSelector(
                                    node_selector_terms=[
                                        client.V1NodeSelectorTerm(
                                            match_expressions=[
                                                client.V1NodeSelectorRequirement(
                                                    key=\"gpu-count\",
                                                    operator=\"In\",
                                                    values=[str(gpus)]
                                                )
                                            ]
                                        )
                                    ]
                                )
                        )
                    ),
                    containers=[
                        client.V1Container(
                            name=\"training\",
                            image=image,
                            resources=client.V1ResourceRequirements(
                                limits={\"nvidia.com/gpu\": gpus},
                                requests={\"nvidia.com/gpu\": gpus, \"memory\": \"400Gi\"}
                            )
                        )
                    ],
                    restart_policy=\"Never\",
                    priority_class_name=\"training-medium\"
                )
            )
        )
    )

    batch_v1.create_namespaced_job(namespace=\"team-nlp\", body=job)
    print(f\"Job {name} created with {gpus} GPUs\")

create_gpu_training_job(\"llm-training-001\", \"my-training-image:v1.0\", gpus=8)
```

**Common Mistakes**:
- Forgetting to apply taints → CPU pods steal GPU nodes
- Not setting resource requests → Scheduler doesn't know GPU requirements
- No resource quotas → One team uses all GPUs
- Missing tolerations → GPU jobs can't schedule
- No priority classes → Critical jobs wait behind experiments

**Excellence Indicators**:
- Discussing taints/tolerations AND affinity (layered strategy)
- Mentioning GPU sharing options (MIG for A100/H100)
- Understanding priority and preemption
- Suggesting monitoring and alerts
- Considering multi-tenancy with quotas

**Follow-up Discussion**:
- "What if a high-priority job needs to evict a running low-priority training job?"
- "How would you handle fractional GPU requests (e.g., 0.5 GPU for small models)?"
- "What metrics would you monitor to detect inefficient GPU usage?"

---

#### Question 8: Intermediate - Kubernetes Autoscaling for Model Serving

**Q**: Design an autoscaling strategy for a model serving deployment that handles variable traffic (1000 QPS peak, 100 QPS off-peak). The model inference takes 50ms per request on GPU.

**Comprehensive Answer**:

"I'll design a comprehensive autoscaling strategy using Horizontal Pod Autoscaler (HPA) and Cluster Autoscaler:

**Step 1: Understand Capacity Requirements**

**Peak Load**:
- Traffic: 1000 QPS
- Latency per request: 50ms on GPU
- Throughput per GPU: 1000ms / 50ms = 20 requests/second per GPU
- GPUs needed: 1000 QPS / 20 RPS = 50 GPUs minimum
- Add 30% buffer for spikes: 65 GPUs

**Off-Peak Load**:
- Traffic: 100 QPS
- GPUs needed: 100 / 20 = 5 GPUs
- With buffer: 7 GPUs

**Step 2: Deployment Configuration**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  namespace: production
spec:
  replicas: 7  # Start with off-peak capacity
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.04
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        - containerPort: 8002
          name: metrics
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: \"16Gi\"
            cpu: \"4\"
          limits:
            nvidia.com/gpu: 1
            memory: \"16Gi\"
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**Step 3: Horizontal Pod Autoscaler (HPA)**

Use custom metrics (QPS) for autoscaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 7   # Off-peak capacity
  maxReplicas: 70  # Peak + buffer
  metrics:
  # Scale based on custom metric: requests per second per pod
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: \"15\"  # Target 15 RPS per pod (75% of capacity)

  # Fallback: CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # GPU utilization (via custom metrics)
  - type: Pods
    pods:
      metric:
        name: nvidia_gpu_utilization_percentage
      target:
        type: AverageValue
        averageValue: \"75\"

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Wait 60s before scaling up
      policies:
      - type: Percent
        value: 50  # Scale up by 50% of current pods
        periodSeconds: 60
      - type: Pods
        value: 10  # Or add 10 pods
        periodSeconds: 60
      selectPolicy: Max  # Use whichever adds more pods

    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5min before scaling down
      policies:
      - type: Percent
        value: 10  # Scale down by 10% at a time
        periodSeconds: 60
```

**Step 4: Expose Custom Metrics**

Deploy Prometheus adapter to expose custom metrics:

```yaml
# ServiceMonitor for Triton metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-serving-metrics
spec:
  selector:
    matchLabels:
      app: model-serving
  endpoints:
  - port: metrics
    interval: 15s
---
# Prometheus rules to compute RPS
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-serving-rps
spec:
  groups:
  - name: model_serving
    interval: 15s
    rules:
    - record: http_requests_per_second
      expr: |
        rate(nv_inference_request_success[1m])
```

**Step 5: Cluster Autoscaler for Nodes**

Configure cluster autoscaler to add GPU nodes when pods are pending:

```yaml
# Cluster Autoscaler deployment (EKS example)
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
        - --cloud-provider=aws
        - --namespace=kube-system
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/ml-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        - --scale-down-delay-after-add=5m
        - --scale-down-unneeded-time=5m
```

**For EKS, configure node group with autoscaling**:
```hcl
# Terraform for EKS node group
resource \"aws_eks_node_group\" \"gpu_inference\" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = \"gpu-inference-g5-2xlarge\"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = [\"g5.2xlarge\"]

  scaling_config {
    desired_size = 7   # Start with off-peak
    max_size     = 70  # Scale to peak
    min_size     = 5   # Absolute minimum
  }

  labels = {
    \"node-type\" = \"gpu-inference\"
    \"gpu-model\" = \"a10g\"
  }

  tags = {
    \"k8s.io/cluster-autoscaler/enabled\" = \"true\"
    \"k8s.io/cluster-autoscaler/ml-cluster\" = \"owned\"
  }
}
```

**Step 6: Cost Optimization with Spot Instances**

For less critical replicas, use spot instances:

```yaml
# Spot instance node group (separate from on-demand)
# Configure 50% capacity on spot, 50% on on-demand

# Deployment with pod topology spread
spec:
  topologySpreadConstraints:
  - maxSkew: 2
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        app: model-serving
  - maxSkew: 1
    topologyKey: node.kubernetes.io/instance-type
    whenUnsatisfiable: ScheduleAnyway
```

**Step 7: PodDisruptionBudget**

Ensure high availability during scale-down:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: model-serving-pdb
spec:
  minAvailable: 70%  # Always keep 70% of pods running
  selector:
    matchLabels:
      app: model-serving
```

**Step 8: Monitoring Dashboard**

Key metrics to monitor:

```yaml
# Grafana dashboard queries
# 1. Current QPS
sum(rate(nv_inference_request_success[1m]))

# 2. Latency p99
histogram_quantile(0.99, sum(rate(nv_inference_request_duration_bucket[5m])) by (le))

# 3. Pod count over time
count(kube_pod_info{pod=~\"model-serving-.*\"})

# 4. GPU utilization per pod
avg by (pod) (DCGM_FI_DEV_GPU_UTIL{pod=~\"model-serving-.*\"})

# 5. Cost per request
(sum(kube_pod_container_resource_requests{pod=~\"model-serving-.*\", resource=\"nvidia_com_gpu\"}) * 0.50) / sum(rate(nv_inference_request_success[1h]))
```

**Step 9: Alerting**

```yaml
# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-serving-alerts
spec:
  groups:
  - name: model_serving
    rules:
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99,
          sum(rate(nv_inference_request_duration_bucket[5m])) by (le)
        ) > 0.1  # p99 > 100ms
      for: 5m
      annotations:
        summary: \"Model serving latency above SLA\"

    - alert: InsufficientCapacity
      expr: |
        sum(rate(nv_inference_request_success[1m])) /
        (count(kube_pod_info{pod=~\"model-serving-.*\"}) * 15) > 0.9
      for: 2m
      annotations:
        summary: \"Need more pods - approaching capacity\"
```

**Complete Autoscaling Strategy**:

```
Traffic Pattern:
Off-Peak (100 QPS) ──► HPA maintains 7 pods (7 GPUs)
                       Cluster Autoscaler: 7 nodes

Peak (1000 QPS)    ──► HPA scales to 65 pods (65 GPUs)
                       Cluster Autoscaler adds nodes
                       Target: 2-3 min scale-up time

Scale Down         ──► After 5min stable low traffic
                       HPA gradually removes pods
                       Cluster Autoscaler removes nodes after 5min
```

**Expected Behavior**:
- **Scale-up time**: 2-3 minutes (new pods + node provisioning)
- **Scale-down time**: 5-10 minutes (stabilization windows)
- **Cost savings**: 85% fewer GPUs during off-peak (7 vs 65)
- **SLA maintained**: p99 latency < 100ms at all traffic levels"

**Key Points to Cover**:
- HPA with custom metrics (not just CPU)
- Cluster autoscaler for node scaling
- Stabilization windows to prevent flapping
- PodDisruptionBudget for availability
- Cost optimization with spot instances

**Code Example**: Custom Metrics Server
```python
# Flask app exposing custom metrics for Prometheus
from flask import Flask, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

app = Flask(__name__)

# Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
gpu_utilization = Gauge('nvidia_gpu_utilization_percentage', 'GPU utilization')

@app.route('/predict', methods=['POST'])
@request_duration.time()
def predict():
    request_count.inc()
    # Model inference logic
    result = model.predict(request.json)
    return jsonify(result)

@app.route('/metrics')
def metrics():
    # Update GPU metrics
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_utilization.set(util.gpu)

    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**Common Mistakes**:
- Scaling only on CPU (ignores GPU utilization)
- Too short stabilization windows (pods flap up/down)
- No PodDisruptionBudget (downtime during scale-down)
- Not considering node provisioning time in scale-up
- Missing readiness probes (traffic sent to unready pods)

**Excellence Indicators**:
- Multi-metric HPA (custom + resource metrics)
- Understanding scale-up vs scale-down policies
- Configuring cluster autoscaler alongside HPA
- Discussing cost optimization (spot instances)
- Mentioning SLIs/SLOs and alerting

**Follow-up Discussion**:
- "What if scale-up takes 5 minutes but traffic spikes in 30 seconds - how do you handle?"
- "How would you implement predictive autoscaling based on historical traffic patterns?"
- "What's your strategy for blue-green deployments with autoscaling?"

---

#### Question 9: Advanced - Building a Kubernetes Operator for ML Training

**Q**: Your company runs hundreds of PyTorch distributed training jobs per week. Each job requires complex setup (multi-node, checkpointing, spot handling). Design a custom Kubernetes Operator to simplify job submission for data scientists.

**Comprehensive Answer**:

"I'll design a custom Kubernetes Operator that abstracts the complexity of distributed training. This is similar to what companies like Uber (Fiber), Kubeflow (PyTorchJob), and Ray (RayJob) have built:

**Step 1: Define Custom Resource (CRD)**

Create a `TrainingJob` CRD that data scientists can use:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.ml.company.com
spec:
  group: ml.company.com
  versions:
  - name: v1alpha1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              # Training configuration
              image:
                type: string
                description: \"Docker image with training code\"
              command:
                type: array
                items:
                  type: string

              # Resource requirements
              workers:
                type: integer
                description: \"Number of worker nodes\"
                minimum: 1
              gpusPerWorker:
                type: integer
                description: \"GPUs per worker\"

              # Checkpointing
              checkpointPath:
                type: string
                description: \"S3 path for checkpoints\"
              checkpointInterval:
                type: integer
                description: \"Checkpoint every N minutes\"

              # Spot instance handling
              useSpot:
                type: boolean
                default: false

              # Environment
              env:
                type: array
                items:
                  type: object
                  properties:
                    name:
                      type: string
                    value:
                      type: string

          status:
            type: object
            properties:
              phase:
                type: string
                enum: [\"Pending\", \"Running\", \"Succeeded\", \"Failed\"]
              startTime:
                type: string
              completionTime:
                type: string
              workerStatuses:
                type: array
                items:
                  type: object
  scope: Namespaced
  names:
    plural: trainingjobs
    singular: trainingjob
    kind: TrainingJob
    shortNames:
    - tj
```

**Step 2: Data Scientist User Experience**

Data scientists submit simple YAML:

```yaml
apiVersion: ml.company.com/v1alpha1
kind: TrainingJob
metadata:
  name: llm-training-experiment-42
  namespace: team-nlp
spec:
  image: my-registry/pytorch-training:v2.0
  command: [\"python\", \"train.py\", \"--config\", \"config.yaml\"]

  # Simple resource specification
  workers: 4          # 4 worker nodes
  gpusPerWorker: 8    # 8 GPUs each = 32 GPUs total

  # Automatic checkpointing
  checkpointPath: s3://my-bucket/experiments/exp-42/checkpoints
  checkpointInterval: 60  # Every 60 minutes

  # Cost savings
  useSpot: true

  # Environment variables
  env:
  - name: MASTER_PORT
    value: \"29500\"
  - name: NCCL_DEBUG
    value: \"INFO\"
```

**Step 3: Operator Controller Logic**

The operator watches `TrainingJob` resources and creates underlying K8s resources:

```python
# Simplified operator controller (using Kopf framework)
import kopf
import kubernetes
from kubernetes import client, config

@kopf.on.create('ml.company.com', 'v1alpha1', 'trainingjobs')
def create_training_job(spec, name, namespace, **kwargs):
    \"\"\"
    When TrainingJob is created, generate:
    1. ConfigMap for distributed training setup
    2. Service for worker communication
    3. Job for master node (rank 0)
    4. Jobs for worker nodes (rank 1-N)
    \"\"\"

    workers = spec['workers']
    gpus_per_worker = spec['gpusPerWorker']
    image = spec['image']
    command = spec['command']
    use_spot = spec.get('useSpot', False)

    # 1. Create Service for worker discovery
    service = create_training_service(name, namespace)

    # 2. Create ConfigMap with distributed training config
    config_map = create_distributed_config(
        name, namespace, workers,
        master_addr=f\"{name}-master-0.{name}.{namespace}.svc.cluster.local\"
    )

    # 3. Create master Job (rank 0)
    master_job = create_worker_job(
        name=f\"{name}-master\",
        namespace=namespace,
        image=image,
        command=command,
        rank=0,
        world_size=workers,
        gpus=gpus_per_worker,
        use_spot=use_spot,
        checkpoint_path=spec.get('checkpointPath'),
        checkpoint_interval=spec.get('checkpointInterval', 60)
    )

    # 4. Create worker Jobs (rank 1 to N-1)
    worker_jobs = []
    for rank in range(1, workers):
        worker_job = create_worker_job(
            name=f\"{name}-worker-{rank}\",
            namespace=namespace,
            image=image,
            command=command,
            rank=rank,
            world_size=workers,
            gpus=gpus_per_worker,
            use_spot=use_spot,
            checkpoint_path=spec.get('checkpointPath'),
            checkpoint_interval=spec.get('checkpointInterval', 60)
        )
        worker_jobs.append(worker_job)

    # Apply all resources
    api = client.CoreV1Api()
    batch_api = client.BatchV1Api()

    api.create_namespaced_service(namespace, service)
    api.create_namespaced_config_map(namespace, config_map)
    batch_api.create_namespaced_job(namespace, master_job)
    for job in worker_jobs:
        batch_api.create_namespaced_job(namespace, job)

    return {'message': f'Training job {name} created with {workers} workers'}

def create_worker_job(name, namespace, image, command, rank, world_size, gpus, use_spot, checkpoint_path, checkpoint_interval):
    \"\"\"Generate K8s Job for a training worker\"\"\"

    env_vars = [
        client.V1EnvVar(name=\"RANK\", value=str(rank)),
        client.V1EnvVar(name=\"WORLD_SIZE\", value=str(world_size)),
        client.V1EnvVar(name=\"MASTER_ADDR\", value=f\"{name.split('-')[0]}-master-0\"),
        client.V1EnvVar(name=\"MASTER_PORT\", value=\"29500\"),
        client.V1EnvVar(name=\"NCCL_DEBUG\", value=\"INFO\"),
        client.V1EnvVar(name=\"CHECKPOINT_PATH\", value=checkpoint_path),
        client.V1EnvVar(name=\"CHECKPOINT_INTERVAL\", value=str(checkpoint_interval)),
    ]

    container = client.V1Container(
        name=\"pytorch-training\",
        image=image,
        command=[\"/bin/bash\", \"-c\"],
        args=[
            # Wrapper script that handles checkpointing and spot interruptions
            f\"\"\"
            # Setup distributed training
            export RANK={rank}
            export WORLD_SIZE={world_size}

            # Handle spot instance interruptions
            (while true; do
                if curl -s http://169.254.169.254/latest/meta-data/spot/instance-action | grep -q action; then
                    echo \"Spot interruption detected, saving checkpoint...\"
                    touch /tmp/checkpoint_now
                    sleep 60
                fi
                sleep 5
            done) &

            # Run training with checkpointing
            python checkpoint_wrapper.py -- {' '.join(command)}
            \"\"\"
        ],
        env=env_vars,
        resources=client.V1ResourceRequirements(
            requests={\"nvidia.com/gpu\": gpus, \"memory\": \"400Gi\"},
            limits={\"nvidia.com/gpu\": gpus}
        ),
        volume_mounts=[
            client.V1VolumeMount(name=\"shm\", mount_path=\"/dev/shm\")
        ]
    )

    # Tolerations for GPU nodes and spot instances
    tolerations = [
        client.V1Toleration(key=\"nvidia.com/gpu\", operator=\"Exists\", effect=\"NoSchedule\")
    ]
    if use_spot:
        tolerations.append(
            client.V1Toleration(key=\"spot-instance\", operator=\"Exists\", effect=\"NoSchedule\")
        )

    pod_template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={\"app\": name, \"rank\": str(rank)}
        ),
        spec=client.V1PodSpec(
            containers=[container],
            restart_policy=\"OnFailure\",
            tolerations=tolerations,
            node_selector={\"node-type\": \"gpu\"},
            volumes=[
                client.V1Volume(
                    name=\"shm\",
                    empty_dir=client.V1EmptyDirVolumeSource(medium=\"Memory\", size_limit=\"64Gi\")
                )
            ]
        )
    )

    job = client.V1Job(
        api_version=\"batch/v1\",
        kind=\"Job\",
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=client.V1JobSpec(
            template=pod_template,
            backoff_limit=3
        )
    )

    return job

@kopf.on.update('ml.company.com', 'v1alpha1', 'trainingjobs')
def update_training_job(spec, status, name, namespace, **kwargs):
    \"\"\"Handle job updates (e.g., cancel, resize)\"\"\"
    # Implementation for job updates
    pass

@kopf.on.delete('ml.company.com', 'v1alpha1', 'trainingjobs')
def delete_training_job(spec, name, namespace, **kwargs):
    \"\"\"Clean up resources when TrainingJob is deleted\"\"\"
    batch_api = client.BatchV1Api()
    api = client.CoreV1Api()

    # Delete all worker jobs
    batch_api.delete_collection_namespaced_job(
        namespace=namespace,
        label_selector=f\"app={name}\"
    )

    # Delete service and configmap
    api.delete_namespaced_service(name, namespace)
    api.delete_namespaced_config_map(f\"{name}-config\", namespace)

    return {'message': f'Training job {name} deleted'}

@kopf.on.timer('ml.company.com', 'v1alpha1', 'trainingjobs', interval=30.0)
def monitor_training_job(spec, status, name, namespace, **kwargs):
    \"\"\"Periodically check job status and update TrainingJob status\"\"\"
    batch_api = client.BatchV1Api()

    jobs = batch_api.list_namespaced_job(
        namespace=namespace,
        label_selector=f\"app={name}\"
    )

    # Check if all workers are complete
    all_succeeded = all(job.status.succeeded == 1 for job in jobs.items)
    any_failed = any(job.status.failed and job.status.failed > 0 for job in jobs.items)

    if all_succeeded:
        return {'phase': 'Succeeded', 'completionTime': str(datetime.now())}
    elif any_failed:
        return {'phase': 'Failed'}
    else:
        return {'phase': 'Running'}
```

**Step 4: Automatic Checkpointing Wrapper**

Include checkpoint wrapper in Docker image:

```python
# checkpoint_wrapper.py - injected into training containers
import os
import sys
import signal
import boto3
import torch
from datetime import datetime

checkpoint_path = os.environ['CHECKPOINT_PATH']
checkpoint_interval = int(os.environ['CHECKPOINT_INTERVAL']) * 60  # Convert to seconds
s3 = boto3.client('s3')

def save_checkpoint(model, optimizer, step):
    \"\"\"Save checkpoint to S3\"\"\"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'timestamp': datetime.now().isoformat()
    }

    local_path = '/tmp/checkpoint.pt'
    torch.save(checkpoint, local_path)

    # Upload to S3
    s3_path = f\"{checkpoint_path}/checkpoint_step_{step}.pt\"
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    s3.upload_file(local_path, bucket, key)
    print(f\"Checkpoint saved to {s3_path}\")

def load_latest_checkpoint(model, optimizer):
    \"\"\"Load latest checkpoint from S3\"\"\"
    bucket, prefix = checkpoint_path.replace('s3://', '').split('/', 1)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if 'Contents' not in response:
        return 0  # No checkpoint

    # Find latest checkpoint
    latest = max(response['Contents'], key=lambda x: x['LastModified'])
    local_path = '/tmp/checkpoint.pt'
    s3.download_file(bucket, latest['Key'], local_path)

    checkpoint = torch.load(local_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f\"Resumed from step {checkpoint['step']}\")
    return checkpoint['step']

# Signal handler for spot interruptions
def handle_interruption(signum, frame):
    print(\"Spot interruption signal received, saving checkpoint...\")
    save_checkpoint(model, optimizer, current_step)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_interruption)

# Continue with normal training script
# This wrapper provides checkpoint save/load for free
```

**Step 5: Deployment & Usage**

Deploy the operator:

```bash
# Deploy CRD
kubectl apply -f trainingjob-crd.yaml

# Deploy operator
kubectl apply -f operator-deployment.yaml
```

**Operator Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trainingjob-operator
  namespace: ml-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trainingjob-operator
  template:
    spec:
      serviceAccountName: trainingjob-operator
      containers:
      - name: operator
        image: my-registry/trainingjob-operator:v1.0
        env:
        - name: PYTHONUNBUFFERED
          value: \"1\"
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: trainingjob-operator
  namespace: ml-platform
---
# RBAC permissions for operator
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: trainingjob-operator
rules:
- apiGroups: [\"ml.company.com\"]
  resources: [\"trainingjobs\"]
  verbs: [\"get\", \"list\", \"watch\", \"update\", \"patch\"]
- apiGroups: [\"batch\"]
  resources: [\"jobs\"]
  verbs: [\"get\", \"list\", \"create\", \"delete\"]
- apiGroups: [\"\"]
  resources: [\"services\", \"configmaps\"]
  verbs: [\"get\", \"list\", \"create\", \"delete\"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: trainingjob-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: trainingjob-operator
subjects:
- kind: ServiceAccount
  name: trainingjob-operator
  namespace: ml-platform
```

**Step 6: Advanced Features**

Additional capabilities to implement:

1. **Auto-resume on Spot Interruption**:
```python
@kopf.on.event('batch', 'v1', 'jobs')
def handle_job_failure(event, **kwargs):
    job = event['object']
    if job.status.failed and 'spot-instance' in job.spec.template.spec.tolerations:
        # Auto-restart job
        recreate_job(job)
```

2. **Resource Request Estimation**:
```python
def estimate_resources(model_size_gb):
    \"\"\"Estimate GPU memory needed based on model size\"\"\"
    # Rule of thumb: 4x model size (model + gradients + optimizer states)
    memory_needed = model_size_gb * 4

    if memory_needed <= 40:
        return 'a100-40gb', 1
    elif memory_needed <= 80:
        return 'a100-80gb', 1
    else:
        return 'a100-80gb', (memory_needed // 80) + 1
```

3. **Cost Tracking**:
```python
@kopf.on.timer('ml.company.com', 'v1alpha1', 'trainingjobs', interval=300.0)
def track_costs(spec, status, name, namespace, **kwargs):
    \"\"\"Calculate job cost based on runtime and resources\"\"\"
    if status.get('phase') == 'Running':
        start_time = datetime.fromisoformat(status['startTime'])
        runtime_hours = (datetime.now() - start_time).total_seconds() / 3600

        # Calculate cost
        gpus = spec['workers'] * spec['gpusPerWorker']
        cost_per_gpu_hour = 1.00 if spec.get('useSpot') else 3.00
        total_cost = runtime_hours * gpus * cost_per_gpu_hour

        # Update status with cost
        return {'estimatedCost': f\"${total_cost:.2f}\"}
```

**Benefits of This Operator**:

1. **Simplified UX**: Data scientists submit 15-line YAML instead of 200+ lines
2. **Best Practices**: Automatic checkpointing, spot handling, distributed setup
3. **Resource Optimization**: Smart GPU allocation, spot instance usage
4. **Observability**: Centralized status, cost tracking, logs
5. **Governance**: Resource quotas, cost limits enforced at operator level

**Real-World Usage**:
```bash
# Data scientist workflow
kubectl apply -f my-training-job.yaml
kubectl get trainingjobs
kubectl logs trainingjob/llm-training-experiment-42
kubectl describe trainingjob llm-training-experiment-42

# Output:
# Name:         llm-training-experiment-42
# Namespace:    team-nlp
# Phase:        Running
# Workers:      4
# GPUs:         32 (8 per worker)
# Start Time:   2025-01-15 10:30:00
# Estimated Cost: $124.50
# Checkpoints:  s3://my-bucket/experiments/exp-42/checkpoints/
```"

**Key Points to Cover**:
- CRD design and schema
- Operator controller logic (create, update, delete, monitor)
- Resource generation (Jobs, Services, ConfigMaps)
- Automatic checkpointing and spot handling
- RBAC and permissions
- User experience improvement

**Code Example**: Complete CRD Submission
```bash
# Submit training job
cat <<EOF | kubectl apply -f -
apiVersion: ml.company.com/v1alpha1
kind: TrainingJob
metadata:
  name: bert-finetuning
spec:
  image: my-registry/bert-training:v1.0
  command: [\"python\", \"finetune.py\"]
  workers: 2
  gpusPerWorker: 4
  checkpointPath: s3://ml-checkpoints/bert-finetuning
  checkpointInterval: 30
  useSpot: true
  env:
  - name: BATCH_SIZE
    value: \"64\"
  - name: LEARNING_RATE
    value: \"5e-5\"
EOF

# Monitor status
kubectl get tj bert-finetuning -w
```

**Common Mistakes**:
- Not handling finalizers properly (resources leak on delete)
- Missing RBAC permissions for operator
- Not validating CRD spec (invalid configs cause crashes)
- No status updates (users don't know job state)
- Ignoring idempotency (operator creates duplicate resources)

**Excellence Indicators**:
- Understanding CRD design patterns
- Implementing proper reconciliation loops
- Handling edge cases (spot interruptions, node failures)
- Designing for user experience (simple API, good defaults)
- Mentioning observability and cost tracking

**Follow-up Discussion**:
- "How would you implement job preemption for low-priority jobs?"
- "What if a user deletes a TrainingJob while training is running - how do you handle graceful shutdown?"
- "How would you extend this operator to support TensorFlow in addition to PyTorch?"

---

#### Question 10: Advanced - Kubernetes Multi-Tenancy for ML Platform

**Q**: Design a multi-tenant Kubernetes architecture for an ML platform serving 20 teams. Each team should have isolated resources, but you need to maximize GPU utilization across teams. How do you prevent one team from monopolizing resources while ensuring fair access?

**Comprehensive Answer**:

"This is a complex multi-tenancy challenge requiring namespace isolation, resource quotas, fair-share scheduling, and cost allocation. Here's a comprehensive design:

**Step 1: Namespace Architecture**

Create isolated namespaces per team:

```bash
# Create namespaces for each team
for team in team-nlp team-cv team-recsys team-search; do
  kubectl create namespace $team
  kubectl label namespace $team team=$team
done

# Create shared namespace for common services
kubectl create namespace ml-platform-shared
```

**Step 2: Resource Quotas per Team**

Implement hard quotas to prevent resource hogging:

```yaml
# Resource quota for team-nlp
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-nlp-quota
  namespace: team-nlp
spec:
  hard:
    # GPU limits
    requests.nvidia.com/gpu: \"40\"    # Max 40 GPUs
    limits.nvidia.com/gpu: \"40\"

    # CPU and memory
    requests.cpu: \"200\"               # Max 200 CPU cores
    requests.memory: \"800Gi\"          # Max 800GB RAM
    limits.cpu: \"400\"
    limits.memory: \"1600Gi\"

    # Object counts
    pods: \"200\"                       # Max 200 pods
    persistentvolumeclaims: \"50\"     # Max 50 PVCs
    services: \"20\"

    # Storage
    requests.storage: \"10Ti\"          # Max 10TB persistent storage

---
# Separate quota for different priority classes
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-nlp-prod-quota
  namespace: team-nlp
spec:
  hard:
    requests.nvidia.com/gpu: \"20\"    # Reserve 20 GPUs for production
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values: [\"production-high\"]
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-nlp-dev-quota
  namespace: team-nlp
spec:
  hard:
    requests.nvidia.com/gpu: \"20\"    # 20 GPUs for dev/experiments
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values: [\"experiment-low\", \"training-medium\"]
```

**Step 3: Fair-Share Scheduling with Priority & Preemption**

Implement priority classes for workload prioritization:

```yaml
# Production workloads (highest priority)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: production-high
value: 1000000
preemptionPolicy: PreemptLowerPriority
globalDefault: false
description: \"Production model serving - can preempt lower priority\"

---
# Training workloads (medium priority)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-medium
value: 500000
preemptionPolicy: PreemptLowerPriority
description: \"Model training - can be preempted by production\"

---
# Experimental workloads (lowest priority)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: experiment-low
value: 100000
preemptionPolicy: Never  # Don't preempt others
description: \"Experiments - gets preempted if resources needed\"
```

**Step 4: Fair-Share GPU Scheduling**

Deploy custom scheduler or configure default scheduler for fair-share:

**Option A: Use Volcano Scheduler (Fair-Share Plugin)**

```yaml
# Volcano scheduler configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: volcano-scheduler-configmap
  namespace: volcano-system
data:
  volcano-scheduler.conf: |
    actions: \"allocate, backfill, preempt\"
    tiers:
    - plugins:
      - name: priority
      - name: gang
      - name: conformance
    - plugins:
      - name: drf        # Dominant Resource Fairness
      - name: predicates
      - name: proportion # Fair-share across namespaces
      - name: nodeorder

# Queue for each team (fair-share unit)
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: team-nlp-queue
spec:
  weight: 20        # This team gets 20% of total resources
  capability:
    nvidia.com/gpu: 40
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: team-cv-queue
spec:
  weight: 30        # This team gets 30%
  capability:
    nvidia.com/gpu: 60
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: team-recsys-queue
spec:
  weight: 25
  capability:
    nvidia.com/gpu: 50
```

**Option B: Use Apache YuniKorn (used by Apple)**

```yaml
# YuniKorn configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: yunikorn-configs
  namespace: yunikorn
data:
  queues.yaml: |
    partitions:
    - name: default
      queues:
      - name: root
        submitacl: \"*\"
        queues:
        - name: team-nlp
          resources:
            guaranteed:
              nvidia.com/gpu: 40
            max:
              nvidia.com/gpu: 80
          properties:
            priority.policy: fifo
        - name: team-cv
          resources:
            guaranteed:
              nvidia.com/gpu: 60
            max:
              nvidia.com/gpu: 100
        - name: team-recsys
          resources:
            guaranteed:
              nvidia.com/gpu: 50
            max:
              nvidia.com/gpu: 90
```

**Step 5: RBAC & Access Control**

Implement Role-Based Access Control:

```yaml
# Role for data scientists in each namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: data-scientist
  namespace: team-nlp
rules:
# Can manage training jobs
- apiGroups: [\"batch\", \"apps\", \"ml.company.com\"]
  resources: [\"jobs\", \"deployments\", \"trainingjobs\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]

# Can view logs and exec into pods
- apiGroups: [\"\"]
  resources: [\"pods\", \"pods/log\", \"pods/exec\"]
  verbs: [\"get\", \"list\", \"create\"]

# Can create services for model serving
- apiGroups: [\"\"]
  resources: [\"services\", \"configmaps\", \"secrets\"]
  verbs: [\"get\", \"list\", \"create\", \"update\", \"delete\"]

# CANNOT modify quotas or RBAC
- apiGroups: [\"\"]
  resources: [\"resourcequotas\"]
  verbs: [\"get\", \"list\"]

---
# Role binding for team members
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-nlp-data-scientists
  namespace: team-nlp
subjects:
- kind: Group
  name: \"team-nlp@company.com\"  # From OIDC/LDAP
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: data-scientist
  apiGroup: rbac.authorization.k8s.io
```

**Step 6: Network Isolation**

Implement NetworkPolicies to isolate team traffic:

```yaml
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: team-nlp
spec:
  podSelector: {}
  policyTypes:
  - Ingress

---
# Allow ingress from same namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: team-nlp
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          team: team-nlp

---
# Allow ingress from shared services (MLflow, monitoring)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-shared-services
  namespace: team-nlp
spec:
  podSelector: {}
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ml-platform-shared
```

**Step 7: Cost Allocation & Showback**

Track and report costs per team:

```python
# Cost allocation script
import prometheus_api_client
from datetime import datetime, timedelta

def calculate_team_costs(namespace, start_time, end_time):
    \"\"\"Calculate costs for a team based on resource usage\"\"\"
    prom = prometheus_api_client.PrometheusConnect(url=\"http://prometheus:9090\")

    # GPU hours
    gpu_query = f'''
        sum(
            avg_over_time(
                kube_pod_container_resource_requests{{
                    namespace=\"{namespace}\",
                    resource=\"nvidia_com_gpu\"
                }}[{end_time - start_time}]
            )
        ) / 3600
    '''
    gpu_hours = prom.custom_query(gpu_query)[0]['value'][1]
    gpu_cost = float(gpu_hours) * 3.00  # $3/GPU-hour

    # CPU hours
    cpu_query = f'''
        sum(
            avg_over_time(
                kube_pod_container_resource_requests{{
                    namespace=\"{namespace}\",
                    resource=\"cpu\"
                }}[{end_time - start_time}]
            )
        ) / 3600
    '''
    cpu_hours = prom.custom_query(cpu_query)[0]['value'][1]
    cpu_cost = float(cpu_hours) * 0.05  # $0.05/CPU-hour

    # Storage
    storage_query = f'''
        sum(
            kube_persistentvolumeclaim_resource_requests_storage_bytes{{
                namespace=\"{namespace}\"
            }}
        ) / (1024^4)  # Convert to TB
    '''
    storage_tb = prom.custom_query(storage_query)[0]['value'][1]
    storage_cost = float(storage_tb) * 23  # $23/TB/month

    total_cost = gpu_cost + cpu_cost + storage_cost

    return {
        'namespace': namespace,
        'period': f\"{start_time} to {end_time}\",
        'gpu_hours': gpu_hours,
        'gpu_cost': gpu_cost,
        'cpu_hours': cpu_hours,
        'cpu_cost': cpu_cost,
        'storage_tb': storage_tb,
        'storage_cost': storage_cost,
        'total_cost': total_cost
    }

# Generate monthly reports
for team_namespace in ['team-nlp', 'team-cv', 'team-recsys']:
    costs = calculate_team_costs(
        team_namespace,
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    print(f\"Team: {team_namespace}\")
    print(f\"Total Cost: ${costs['total_cost']:.2f}\")
    print(f\"GPU Cost: ${costs['gpu_cost']:.2f} ({costs['gpu_hours']:.1f} hours)\")
    print(\"---\")
```

**Step 8: Resource Borrowing & Bursting**

Allow teams to temporarily exceed quotas when cluster has spare capacity:

```yaml
# LimitRange to set defaults and enforce maximum limits
apiVersion: v1
kind: LimitRange
metadata:
  name: team-nlp-limits
  namespace: team-nlp
spec:
  limits:
  # Default requests if not specified
  - default:
      nvidia.com/gpu: \"1\"
      memory: \"16Gi\"
      cpu: \"4\"
    defaultRequest:
      nvidia.com/gpu: \"1\"
      memory: \"8Gi\"
      cpu: \"2\"
    type: Container

  # Maximum per container
  - max:
      nvidia.com/gpu: \"8\"   # No single container can request >8 GPUs
      memory: \"400Gi\"
      cpu: \"64\"
    type: Container

  # Maximum per pod
  - max:
      nvidia.com/gpu: \"8\"
      memory: \"400Gi\"
    type: Pod
```

**Implement elasticity with Cluster Autoscaler**:
- Teams can burst beyond guaranteed quotas when spare capacity exists
- When total cluster demand exceeds capacity, fair-share scheduler allocates based on weights
- Lowest priority jobs (experiments) get preempted first

**Step 9: Observability Dashboard**

Deploy team-specific Grafana dashboards:

```yaml
# Prometheus queries for team dashboard
# 1. Current GPU usage vs quota
(
  sum(kube_pod_container_resource_requests{namespace=\"team-nlp\", resource=\"nvidia_com_gpu\"})
  /
  kube_resourcequota{namespace=\"team-nlp\", resource=\"requests.nvidia.com/gpu\", type=\"hard\"}
) * 100

# 2. Pod distribution by priority
sum by (priority_class) (kube_pod_info{namespace=\"team-nlp\"})

# 3. Cost trend over time
sum(
  rate(
    kube_pod_container_resource_requests{namespace=\"team-nlp\", resource=\"nvidia_com_gpu\"}[1h]
  )
) * 3.00  # $3/GPU-hour

# 4. Queue wait time (if using Volcano/YuniKorn)
avg(volcano_queue_pod_pending_time{queue=\"team-nlp-queue\"})
```

**Step 10: Policy Enforcement**

Use OPA (Open Policy Agent) or Kyverno for policy enforcement:

```yaml
# Kyverno policy: Require resource requests
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-gpu-requests
spec:
  validationFailureAction: enforce
  rules:
  - name: require-gpu-resource-requests
    match:
      any:
      - resources:
          kinds:
          - Pod
        namespaceSelector:
          matchLabels:
            enforce-quotas: \"true\"
    validate:
      message: \"GPU requests must be specified\"
      pattern:
        spec:
          containers:
          - resources:
              requests:
                nvidia.com/gpu: \"?*\"

---
# Policy: Prevent using latest tag
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-latest-tag
spec:
  validationFailureAction: enforce
  rules:
  - name: require-image-tag
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: \"Using 'latest' tag is not allowed\"
      pattern:
        spec:
          containers:
          - image: \"!*:latest\"
```

**Complete Multi-Tenancy Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ team-nlp    │  │ team-cv     │  │ team-recsys │        │
│  │ Namespace   │  │ Namespace   │  │ Namespace   │        │
│  │             │  │             │  │             │        │
│  │ Quota:      │  │ Quota:      │  │ Quota:      │        │
│  │ - 40 GPUs   │  │ - 60 GPUs   │  │ - 50 GPUs   │        │
│  │ - 200 CPUs  │  │ - 300 CPUs  │  │ - 250 CPUs  │        │
│  │             │  │             │  │             │        │
│  │ Network     │  │ Network     │  │ Network     │        │
│  │ Isolation   │  │ Isolation   │  │ Isolation   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          │                                  │
│                ┌─────────▼──────────┐                       │
│                │  Fair-Share        │                       │
│                │  Scheduler         │                       │
│                │  (Volcano/YuniKorn)│                       │
│                └─────────┬──────────┘                       │
│                          │                                  │
│          ┌───────────────┴───────────────┐                  │
│          │                               │                  │
│  ┌───────▼───────┐              ┌────────▼────────┐         │
│  │  GPU Nodes    │              │  CPU Nodes      │         │
│  │  (200 GPUs)   │              │  (1000 CPUs)    │         │
│  └───────────────┘              └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Results**:
- **Isolation**: Teams cannot access each other's resources or data
- **Fair Access**: Fair-share scheduler ensures equitable GPU distribution
- **Flexibility**: Teams can burst when spare capacity available
- **Cost Transparency**: Detailed cost reports per team
- **Efficiency**: 85%+ cluster GPU utilization vs 60% without fair-share
- **Governance**: Quotas prevent runaway resource usage"

**Key Points to Cover**:
- Namespace isolation per team
- Resource quotas (hard limits)
- Fair-share scheduling (Volcano, YuniKorn, or custom)
- Priority classes and preemption
- RBAC for access control
- Network isolation
- Cost allocation and showback
- Policy enforcement

**Common Mistakes**:
- Only using quotas without fair-share scheduler (teams race for resources)
- No network isolation (security risk)
- Missing RBAC (teams can interfere with each other)
- No cost tracking (teams overconsume)
- Quotas too restrictive (low utilization) or too loose (monopolization)

**Excellence Indicators**:
- Discussing multiple isolation mechanisms (quota, RBAC, network)
- Mentioning advanced schedulers (Volcano, YuniKorn)
- Understanding priority and preemption
- Implementing cost allocation
- Considering both guaranteed and burst capacity
- Policy enforcement (OPA, Kyverno)

**Follow-up Discussion**:
- "How would you handle a situation where one team's production workload needs to preempt another team's training?"
- "What if teams want to share models or datasets - how do you enable collaboration while maintaining isolation?"
- "How would you implement chargeback (teams actually pay for resources) instead of showback?"

---

### Key Takeaways: Kubernetes for ML

**Must-Know Concepts**:
1. GPU scheduling with device plugins and node taints
2. Resource types for ML workloads (Job, Deployment, StatefulSet)
3. Autoscaling (HPA with custom metrics, Cluster Autoscaler)
4. Building operators for custom ML workflows
5. Multi-tenancy with quotas, fair-share, RBAC

**Interview Success Factors**:
- Demonstrate hands-on K8s experience (mention specific kubectl commands, YAML)
- Discuss ML-specific challenges (GPU scheduling, distributed training, checkpointing)
- Show understanding of production concerns (availability, cost, security)
- Mention specific tools (Kubeflow, Volcano, YuniKorn, Triton)

**Common Interview Red Flags**:
- Only theoretical knowledge, no hands-on K8s experience
- Can't explain difference between Job and Deployment
- No awareness of GPU scheduling challenges
- Doesn't understand multi-tenancy requirements
- Never mentions monitoring or observability

---

*[Sections 2.3 through 2.10 and remaining sections will continue in the same detailed format covering Python, Terraform, Model Serving, MLOps Platforms, Monitoring, CI/CD, Distributed Training, System Design, Hands-On Scenarios, Behavioral Questions, and Study Timeline]*

---

## STUDY TIMELINE & PREPARATION ROADMAP {#study-timeline}

### 8-Week Intensive Preparation Plan

**Week 1-2: Core Infrastructure Foundations**
- Cloud Platforms: AWS or GCP deep dive (choose primary platform)
  - Complete: AWS Solutions Architect or GCP Professional Cloud Architect course
  - Hands-on: Deploy ML model on SageMaker/Vertex AI
  - Practice: Estimate costs for training scenarios
- Kubernetes Fundamentals
  - Complete: Kubernetes for Beginners course
  - Hands-on: Deploy multi-pod application locally (Minikube)
  - Practice: Write Deployments, Services, Ingress YAML

**Week 3-4: ML-Specific Infrastructure**
- Kubernetes for ML
  - Study: GPU scheduling, Jobs, StatefulSets
  - Hands-on: Deploy PyTorch training job on K8s
  - Practice: Write HPA with custom metrics
- Model Serving
  - Study: vLLM, Triton, TensorRT-LLM documentation
  - Hands-on: Deploy model with Triton on K8s
  - Practice: Optimize inference latency

**Week 5-6: MLOps & Advanced Topics**
- MLOps Platforms
  - Study: MLflow, Kubeflow, Ray
  - Hands-on: Build end-to-end pipeline with MLflow
  - Practice: Implement experiment tracking
- Infrastructure as Code
  - Study: Terraform for AWS/GCP
  - Hands-on: Write Terraform for EKS cluster
  - Practice: Manage state, modules, workspaces

**Week 7: System Design & Hands-On**
- System Design Practice
  - Study: Example solutions in this guide
  - Practice: Design 3-5 systems end-to-end
  - Mock: Explain designs to friend/peer
- Troubleshooting Scenarios
  - Study: Common Kubernetes issues
  - Practice: Debug failing pods, OOMKilled containers
  - Hands-on: Performance tuning exercises

**Week 8: Mock Interviews & Final Prep**
- Technical Mock Interviews
  - Schedule: 2-3 mock interviews with peers
  - Practice: All question types (technical, system design, behavioral)
  - Review: Analyze gaps, iterate
- Behavioral Preparation
  - Draft: STAR stories for key experiences
  - Practice: Deliver concisely (2-3 minutes each)
- Final Review
  - Review: All questions in this guide
  - Refresh: Key commands, configurations
  - Prepare: Questions to ask interviewers

### Daily Study Schedule (Week 1-2 Example)

**Monday-Friday (3-4 hours/day)**:
- Morning (1 hour): Study theory (courses, documentation)
- Evening (2-3 hours): Hands-on practice (build, deploy, troubleshoot)

**Saturday (6-8 hours)**:
- Deep-dive project: Build something substantial (e.g., deploy multi-node training)

**Sunday (2-3 hours)**:
- Review week, document learnings, plan next week

---

*This is a comprehensive starting framework. The complete guide would continue with detailed sections for all remaining topics. Would you like me to continue with specific sections?*
