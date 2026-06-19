<div align="center">

<h1>From Debate to Equilibrium: Belief‑Driven Multi‑Agent LLM Reasoning via Bayesian Nash Equilibrium</h1>
<h3>Efficient Coordination via Nash Equilibrium for Multi-Agent LLM Framework</h3>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Together AI](https://img.shields.io/badge/Together%20AI-Integration-green.svg)](https://www.together.ai/)

<div align="center">
    <figure>
        <br>
        <p><em>A multi-agent reinforcement learning framework that combines Large Language Models with coordinated decision-making for complex reasoning tasks</em></p>
    </figure>
</div>

</div>

## Motivation

Existing multi-agent frameworks face significant limitations when applied to Large Language Models (LLMs). Traditional approaches struggle with the high-dimensional nature of language models and lack proper coordination mechanisms for complex reasoning tasks.

<div align="center">
    <figure>
        <img src="assets/compare.jpg" alt="ECON vs Traditional MAD Comparison" width="800">
        <br>
        <p><em>Comparison between ECON and traditional Multi-Agent Debate (MAD) approaches</em></p>
    </figure>
</div>

Current multi-agent LLM systems suffer from:

- **Prohibitive Communication Costs**: Traditional multi-agent debate relies on explicit message passing, incurring substantial token costs and computational overhead
- **No Convergence Guarantees**: Current approaches lack theoretical assurances of converging to stable, effective solutions
- **Scalability Challenges**: Information exchange often exceeds LLM context limits, severely impeding scalability in large agent ensembles

### Our Solution: ECON Framework

<div align="center">
    <figure>
        <img src="assets/framework.jpg" alt="ECON Framework Architecture" width="800">
        <br>
        <p><em>ECON's two-stage coordination architecture with Bayesian Nash Equilibrium</em></p>
    </figure>
</div>

To address these critical challenges, we introduce **ECON** - a multi-agent LLM framework that implements efficient coordination via Bayesian Nash Equilibrium, enabling scalable and theoretically grounded multi-agent reasoning.

- **Implicit Belief-Driven Coordination**: Replaces costly message passing with belief-based coordination, dramatically reducing communication overhead
- **Guaranteed Convergence to Equilibrium**: Establishes a rigorous Bayesian Nash Equilibrium (BNE) framework with theoretical convergence guarantees  
- **Hierarchical & Scalable Architecture**: Enables effective coordination in large ensembles via a local-to-global approach that respects LLM context limits

## Minimal Usage

## Installation

We provide two installation methods:

### Package Installation (Recommended)

Install the ECON framework dependencies:

```bash
pip install -r requirements.txt
```

### Development Installation

For development or customization, clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/ECON.git
cd ECON

# Create and activate conda environment  
conda create -n econ python=3.8
conda activate econ

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

Before running the framework, you need to set up the Together AI API key:

```bash
export TOGETHER_API_KEY="your_together_ai_api_key"
```

## Usage

### Quick Start with Command Line Interface

Set your API key once:

```bash
export TOGETHER_API_KEY="your_together_ai_api_key"
```

### One-line Math sanity run (train 1 ep, test 5 eps)

```bash
python scripts/run_math_test.py \
  --train-eps 1 \
  --test-eps 5 \
  --log-dir logs_exp1 \
  --model-dir models_exp1
```

### Default P0 training + BNE testing

```bash
python scripts/run_p0_test.py \
  --train-eps 100 \
  --test-eps 30 \
  --log-dir logs_exp1 \
  --model-dir models_exp1
```

Notes:
- `max_rounds` defaults to 1 (single decision) in `scripts/config_p0.yaml` and `scripts/config_math.yaml`; increase if you need multi-round episodes.
- Reward weights in env reuse the α weights learned during training; override via `reward.initial_weights` in config.
- All schemes expect `agent_memory` so runner/learner can consume short-term trajectories.

## Configuration

### Key Parameters

- `n_agents`: Number of executor agents. The default hierarchy uses 6 executors.
- `hierarchy`: Coordinator layout. The default is 1 global coordinator, 2 sub-coordinators, and 3 executors per sub-coordinator.
- `coordinator_model`: Coordinator LLM model name  
- `executor_model`: Executor LLM model name
- `update_interval`: Gradient update frequency (default: 10 steps)
- `bne_max_iterations`: Maximum BNE coordination iterations
- `belief_dim`: Dimension of agent belief states
- `sampling.temperature_min/max`: Bounds for temperature
- `sampling.p_min/max`: Bounds for repetition penalty (second action dimension)
- `sampling.top_p_default`: Fixed top_p used for generation (default 0.9)

### Supported Models

The framework supports any open-source language model accessible via Together AI API. Models can be hosted using:

- **Together AI**: For remote model serving with API access
- **Local APIs**: Compatible with OpenAI-style APIs

#### Example: Using Llama-3.3-70B-Instruct-Turbo

```bash
./run_econ.sh \
  --api-key YOUR_API_KEY \
  --config src/config/config.yaml \
  --agents 3 \
  --experiment-name llama-coordination-test
```



#### Custom Datasets

Create your own datasets following the Hugging Face format with `question` and `answer` fields:

```yaml
env_args:
  hf_dataset_path: "your_custom_dataset"
  dataset_split: "train"
  question_field_name: "question"
  answer_field_name: "answer" 
  max_question_length: 1024
  max_answer_length: 512
```

## Testing & Evaluation

### Available Testing Methods

The framework provides **multiple testing approaches** for comprehensive model validation:

#### 1. **Integrated Training + Testing** ([scripts/run_p0_test.py](scripts/run_p0_test.py))

Train and test in one command (recommended for quick experiments):

```bash
# Quick test (5 train episodes, 3 test episodes)
python scripts/run_p0_test.py \
  --train-eps 5 \
  --test-eps 3 \
  --log-dir logs_quick \
  --model-dir models_quick

# Full training (100 train episodes, 30 test episodes)
python scripts/run_p0_test.py \
  --train-eps 100 \
  --test-eps 30 \
  --log-dir logs_exp1 \
  --model-dir models_exp1
```

**Features:**
- Automatically runs training followed by BNE testing (3 rounds)
- Saves model checkpoints to `--model-dir`
- Logs test traces to `--log-dir/llm_traces_test_bne_3rounds.json`
- Reports accuracy and P0 metadata (JSON parsing rate)

#### 2. **Testing Pre-trained Models** ([scripts/test_p0.py](scripts/test_p0.py))

Test existing trained models without retraining:

```bash
export TOGETHER_API_KEY="your_api_key"
python scripts/test_p0.py
```

**Configuration:**
- Edit `MODEL_DIR` variable in [test_p0.py](scripts/test_p0.py:11) to point to your trained model directory (e.g., `./models_exp1/final`)
- Runs both baseline (no BNE) and P0 BNE (3 rounds) tests
- Outputs: `logs_p0_test_baseline.json` and `logs_p0_test_p0.json`

**Example Output:**
```
Baseline (no BNE):    10/10 = 100.0%
P0 BNE (3 rounds):    10/10 = 100.0%
P0 Metadata:         JSON=100%
```

> Note: `src/eval.py` is not part of the current workflow; rely on the above test scripts for evaluation.

#### 3. **Dataset-Specific Testing**

Test on MATH or SVAMP datasets:

```bash
# MATH dataset
python scripts/run_math_test.py \
  --train-eps 5 \
  --test-eps 10 \
  --log-dir logs_math \
  --model-dir models_math

# SVAMP dataset
python scripts/run_svamp_test.py \
  --train-eps 5 \
  --test-eps 10 \
  --log-dir logs_svamp \
  --model-dir models_svamp
```

Already-trained checkpoints can be evaluated directly with `scripts/test_math.py` and `scripts/test_svamp.py` (baseline vs BNE, 10 episodes each by default).

No-BNE baseline evaluation can also run multiple plain discussion rounds without BNE networks:

```bash
python scripts/run_baseline_no_bne.py --baseline-rounds 3
```

For the baseline controller, `baseline_rounds` is the total number of executor/coordinator discussion rounds. Each round runs all executors once, then the coordinator aggregates once. Later rounds feed the previous executor outputs and coordinator commitment back into the next executor prompts.

### Episode Structure Explanation

**Important:** Episodes in ECON have a unique structure that differs from traditional RL environments.

**Default Setup (Single-Decision Episodes)**:
```
Episode = One math problem
├─ t=0: Decision step (includes K internal BNE refinement rounds)
│   └─ BNE coordination: belief updates, response generation, convergence
└─ t=1: Terminal state (reward computation)
```

**Key Points:**
- One episode = one math problem (not multiple attempts)
- 2 RL timesteps = 1 decision + 1 terminal (standard episodic RL)
- BNE refinement (K rounds) happens **internally** at t=0
- Multi-round debate occurs **inside** the decision step via belief coordination

**Internal BNE Process (at t=0)**:
```python
# Within single timestep t=0:
Round 0: e_init → LLM outputs → Commitment_0
Round 1: e_refined_1 → LLM outputs → Commitment_1
Round 2: e_refined_2 → LLM outputs → Commitment_2
Final: Submit Commitment_2 as answer
```

**Configuration:**
```yaml
bne:
  max_iterations_train: 1    # BNE rounds during training
  max_iterations_infer: 3    # BNE rounds during testing
```

**Optional Multi-Round Episodes:**

For environments requiring multiple answer revision steps:

```yaml
env_args:
  max_rounds: 3  # Enable multi-round attempts (default is 1)
```

This creates true multi-step episodes:
- **Single-step (default)**: 2 timesteps (decision + terminal)
- **Multi-round (optional)**: (N+1) timesteps (N decisions + terminal)

## Advanced Features

### Architecture Components

- **Coordinator LLM**: Generates strategies (≤50 tokens) and final commitments without revealing answers
- **Executor LLMs**: Multiple agents that process strategies and generate individual responses
- **BeliefNetwork**: Individual agent belief state management with Q-value computation
- **BeliefEncoder**: Group representation aggregation using attention mechanisms
- **Mixer**: Attention-based agent interaction layer that aggregates local Q values with commitment-alignment and consistency regularization

### Bayesian Nash Equilibrium Framework

Our approach implements a rigorous two-stage BNE framework:

1. **Stage 1 - Individual Belief Formation**: Each agent develops independent belief states and generates initial responses
2. **Stage 2 - BNE Coordination**: Agents iteratively update beliefs through equilibrium computation until convergence

### Reward System

Three reward components are implemented in `src/envs/huggingface_dataset_env.py` using embeddings:

- **Task-Specific (TS)**: Binary correctness from numeric match with the ground truth.
- **Action Likelihood (AL)**: Cosine similarity between executor outputs and the coordinator commitment embedding (mapped to [0,1]).
- **Collaborative Contribution (CC)**: Combination of fidelity to the commitment and novelty vs. peer responses (embedding-based).

Dynamic α-weights are learned via the learner’s `alpha_logits` (see `src/learners/q_learner.py`), and the runner passes the live weights to the env so reward composition is consistent between train/test. AL/CC require the commitment/output embedding model (`BAAI/bge-large-en-v1.5`, installed via `sentence-transformers`) to be available locally.

### Loss Functions

- **TD Loss**: Local Q TD error (per agent).
- **Mixer Loss**: Global TD + consistency loss + commitment alignment from the mixer.
- **BNE Losses**: For BNE mode, equilibrium and commitment-improvement terms are added (see learner).
- **α Update**: Optional dynamic reward-weight update (when enabled in config).

### Early Stopping

Our practical early-stopping hooks are:

1) **BNE convergence guard (train & infer)** — implemented in `src/controllers/basic_mac.py`. Refinement halts when
   the coordinator commitment matches the previous round, the prompt-parameter delta falls below the configured
   `bne.convergence_threshold`, or an oscillation pattern is detected.
2) **Best-checkpoint saving** — `src/train.py` saves periodic checkpoints and always writes a final snapshot.
3) **Theoretical three-criterion stopper (optional)** — enabled when `early_stopping.enabled: true` and thresholds are provided in config: `commitment_threshold`, `reward_threshold`, `loss_threshold`, plus `patience`/`warmup`. Training stops when all three conditions hold for `patience` episodes. If thresholds are absent, a single-metric stopper (default `reward_mean`) is used.

### Target Network Updates

Soft update mechanism (paper Section 4.4):
```
φ' ← τφ + (1-τ)φ'
```

```yaml
target_update_tau: 0.01          # Soft update parameter τ
target_update_interval: 8        # Update frequency
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{
yi2025from,
title={From Debate to Equilibrium: Belief\nobreakdash-Driven Multi\nobreakdash-Agent {LLM} Reasoning via Bayesian Nash Equilibrium},
author={Yi Xie and Zhanke Zhou and Chentao Cao and Qiyu Niu and Tongliang Liu and Bo Han},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=RQwexjUCxm}
}
```

## Frequently Asked Questions

**Q1: How do I test?**  
Use [scripts/run_p0_test.py](scripts/run_p0_test.py) for train+test in one run, or [scripts/test_p0.py](scripts/test_p0.py) to evaluate a saved checkpoint (baseline vs BNE).

**Q2: Why do episodes only have 2 timesteps?**  
The environment is single-decision (t=0) + terminal (t=1); multi-round BNE happens inside the decision step.

**Q3: What early-stopping logic is active?**  
BNE convergence checks inside the controller, plus the optional single-metric stopper in `train.py` (patience on `reward_mean` when enabled).

## Contact

For questions, technical support, or collaboration inquiries:

- Issues: [GitHub Issues](https://github.com/tmlr-group/ECON/issues)
