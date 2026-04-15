# src/train.py
# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from utils.logging import get_logger
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from utils.early_stopping import EarlyStopping, EarlyStoppingConfig
from components.episode_buffer import EpisodeBatch

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")

# ------------------------
# Config helpers
# ------------------------
def _expand_env_vars(obj):

    if isinstance(obj, str):
        s = os.path.expandvars(obj).strip()
        if _ENV_PATTERN.fullmatch(s):
            return ""
        return s
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    return obj

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(x) for x in d]
    else:
        return d

def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML and expand ${ENV_VARS} safely."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    expanded = _expand_env_vars(raw)


    if isinstance(expanded, dict):
        root_key = expanded.get("together_api_key") or ""
        llm_key = expanded.get("llm", {}).get("together_api_key") if isinstance(expanded.get("llm"), dict) else ""
        effective = (root_key or llm_key or "").strip()
        if "llm" not in expanded:
            expanded["llm"] = {}
        expanded["together_api_key"] = effective
        expanded["llm"]["together_api_key"] = effective

    return _dict_to_namespace(expanded)

def update_config_with_args(config: SimpleNamespace, args: Any) -> SimpleNamespace:
    if getattr(args, "executor_model", None) and hasattr(config, 'llm'):
        config.llm.executor_model = args.executor_model
        config.executor_model = args.executor_model
    if getattr(args, "coordinator_model", None) and hasattr(config, 'llm'):
        config.llm.coordinator_model = args.coordinator_model
        config.coordinator_model = args.coordinator_model
    if getattr(args, "n_agents", None):
        config.n_agents = args.n_agents
    if getattr(args, "experiment_name", None) and hasattr(config, 'logging'):
        config.logging.experiment_name = args.experiment_name
    if getattr(args, "log_dir", None) and hasattr(config, 'logging'):
        config.logging.log_path = args.log_dir
    if getattr(args, "api_key", None):
        config.together_api_key = args.api_key
        if hasattr(config, 'llm'):
            config.llm.together_api_key = args.api_key
    if getattr(args, "seed", None) and hasattr(config, 'system'):
        config.system.seed = args.seed
    if getattr(args, "env", None):
        config.env = args.env

    if not hasattr(config, 'wandb'):
        config.wandb = SimpleNamespace()
    config.wandb.use_wandb = bool(getattr(args, "use_wandb", False))
    if getattr(args, "wandb_project", None):
        config.wandb.project = args.wandb_project
    if getattr(args, "wandb_entity", None):
        config.wandb.entity = args.wandb_entity
    if getattr(args, "wandb_tags", None):
        config.wandb.tags = args.wandb_tags.split(',')

    return config

# ------------------------
# Setup
# ------------------------
def setup_experiment(config: SimpleNamespace):
    logger = get_logger()
    logger.info("Setting up experiment environment...")

    # seeds
    seed = config.system.seed if hasattr(config, 'system') and hasattr(config.system, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # device
    use_cuda = hasattr(config, 'system') and hasattr(config.system, 'use_cuda') and config.system.use_cuda and torch.cuda.is_available()
    device_num = config.system.device_num if hasattr(config, 'system') and hasattr(config.system, 'device_num') else 0
    device = torch.device(f"cuda:{device_num}" if use_cuda else "cpu")
    config.device = device
    if use_cuda:
        torch.cuda.set_device(device_num)


    runner = r_REGISTRY[config.runner](config, logger)


    env_info = runner.get_env_info()
    if not hasattr(config, 'n_actions'):
        setattr(config, 'n_actions', int(env_info.get("n_actions", 2)))

    #  Scheme & groups
    commitment_dim = int(getattr(config, 'commitment_embedding_dim', 768))
    belief_dim = int(getattr(config, 'belief_dim', 128))
    memory_dim = int(getattr(config, 'memory_dim', belief_dim + 9))
    setattr(config, "memory_dim", memory_dim)
    n_actions = int(getattr(config, 'n_actions', 2))
    n_agents = int(getattr(config, 'n_agents', 3))

    scheme = {
        "state": {"vshape": tuple(env_info.get("state_shape", (1,)))},
        "obs": {"vshape": tuple(env_info.get("obs_shape", (1024,))), "group": "agents", "dtype": torch.long},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": torch.int64},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.long},
        "is_correct": {"vshape": (1,), "dtype": torch.float32},

      
        "q_values": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "prompt_embeddings": {"vshape": (2,), "group": "agents", "dtype": torch.float32},
        "belief_states": {"vshape": (belief_dim,), "group": "agents", "dtype": torch.float32},
        "group_representation": {"vshape": (belief_dim,), "dtype": torch.float32},
        "commitment_embedding": {"vshape": (commitment_dim,), "dtype": torch.float32},
        "agent_memory": {"vshape": (memory_dim,), "group": "agents", "dtype": torch.float32},

       
        "bne_e_init": {"vshape": (2,), "group": "agents", "dtype": torch.float32},
        "bne_e_refined": {"vshape": (2,), "group": "agents", "dtype": torch.float32},
        "bne_commitment_emb_0": {"vshape": (commitment_dim,), "dtype": torch.float32},
        "bne_output_emb_0": {"vshape": (commitment_dim,), "group": "agents", "dtype": torch.float32},
        "bne_output_emb_1": {"vshape": (commitment_dim,), "group": "agents", "dtype": torch.float32},
        "bne_group_repr": {"vshape": (belief_dim,), "dtype": torch.float32},

       
        "reward_al": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_ts": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_cc": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
    }
    groups = {"agents": n_agents}


    mac = mac_REGISTRY[config.mac](scheme, groups, config)


    runner.setup(scheme, groups, None, mac)

    learner = le_REGISTRY[config.learner](mac=mac, scheme=scheme, logger=logger, args=config)
    if hasattr(runner, "set_alpha_provider"):
        runner.set_alpha_provider(learner)

    return runner, mac, learner, logger, device

# ------------------------
# run_training
# ------------------------
def run_training(config: SimpleNamespace, runner, learner, logger, device):
    # torch.autograd.set_detect_anomaly(True)

    logger.info("Starting training...")

    begin_time = time.time()
    log_dir = Path(getattr(getattr(config, 'logging', SimpleNamespace()), 'log_path', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV 
    csv_path = log_dir / 'training_metrics.csv'
    if not csv_path.exists():
        with open(csv_path, 'w') as f:
            f.write('episode,t_env,'
                    'loss_total,loss_belief,loss_encoder,loss_mixer,'
                    'mixer_L_TD_tot,mixer_L_cons,mixer_L_align,'
                    'q_local_mean,reward_mean,status\n')

    trace_train_path = log_dir / 'llm_traces_train.json'
    all_traces_train: List[Dict[str, Any]] = []

    t_max = int(getattr(config, 't_max', 2000000))
    log_interval = getattr(getattr(config, 'logging', SimpleNamespace()), 'log_interval', 2000)
    save_model_interval = getattr(getattr(config, 'logging', SimpleNamespace()), 'save_model_interval', 10000)

    # Early stopping (optional)
    es_cfg_ns = getattr(config, "early_stopping", None)
    stopper = None
    es_metric_name = "reward_mean"
    use_theoretical_es = False
    if es_cfg_ns and getattr(es_cfg_ns, "enabled", False):
        # If thresholds provided, use theoretical triple-guard; otherwise fallback to metric-based stopper
        has_thresholds = all([
            hasattr(es_cfg_ns, "commitment_threshold"),
            hasattr(es_cfg_ns, "reward_threshold"),
            hasattr(es_cfg_ns, "loss_threshold"),
        ])
        use_theoretical_es = has_thresholds
        if not use_theoretical_es:
            es_metric_name = getattr(es_cfg_ns, "metric", "reward_mean")
            es_mode = getattr(es_cfg_ns, "mode", "max")
            es_patience = int(getattr(es_cfg_ns, "patience", 10))
            es_min_delta = float(getattr(es_cfg_ns, "min_delta", 0.0))
            es_warmup = int(getattr(es_cfg_ns, "warmup", 0))
            stopper = EarlyStopping(EarlyStoppingConfig(
                patience=es_patience,
                mode=es_mode,
                min_delta=es_min_delta,
                warmup=es_warmup,
            ))
            logger.info(f"Early stopping enabled on metric='{es_metric_name}' mode={es_mode}, patience={es_patience}, warmup={es_warmup}")

    episode = 0
    t_env = 0

    prev_loss = None
    prev_commitment = None
    es_ok_count = 0
    es_warmup_steps = int(getattr(es_cfg_ns, "warmup", 0)) if es_cfg_ns else 0
    es_patience = int(getattr(es_cfg_ns, "patience", 10)) if es_cfg_ns else 10
    commitment_threshold = float(getattr(es_cfg_ns, "commitment_threshold", 0.0)) if es_cfg_ns else 0.0
    reward_threshold = float(getattr(es_cfg_ns, "reward_threshold", 0.0)) if es_cfg_ns else 0.0
    loss_threshold = float(getattr(es_cfg_ns, "loss_threshold", 0.0)) if es_cfg_ns else 0.0

    def _norm_commit(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    while t_env < t_max:
     
        ep_batch: Optional[EpisodeBatch] = runner.run(test_mode=False)

      
        trace = getattr(runner, "get_last_trace", lambda: None)()
        if trace is not None:
            all_traces_train.append(trace)
         
            try:
                with open(trace_train_path, "w", encoding="utf-8") as jf:
                    json.dump(all_traces_train, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write train traces JSON: {e}")

       
        train_stats = learner.train(ep_batch, t_env, episode)

       
        if isinstance(train_stats, dict) and train_stats.get("status") == "nan":
            logger.warning("NaN in loss; skip step")
        else:
            
            if isinstance(train_stats, dict):
                try:
                    row = [
                        str(episode), str(t_env),
                        str(train_stats.get('loss_total', 'N/A')),
                        str(train_stats.get('loss_belief', 'N/A')),
                        str(train_stats.get('loss_encoder', 'N/A')),
                        str(train_stats.get('loss_mixer', 'N/A')),
                        str(train_stats.get('mixer_L_TD_tot', 'N/A')),
                        str(train_stats.get('mixer_L_cons', 'N/A')),
                        str(train_stats.get('mixer_L_align', 'N/A')),
                        str(train_stats.get('q_local_mean', 'N/A')),
                        str(train_stats.get('reward_mean', 'N/A')),
                        str(train_stats.get('status', 'N/A')),
                    ]
                    with open(csv_path, 'a') as f:
                        f.write(','.join(row) + '\n')
                except Exception as e:
                    logger.warning(f"Failed to write CSV row: {e}")


        if episode % max(1, int(log_interval)) == 0:

            if trace:
                pred = trace.get("pred_answer", "None")
                gt = trace.get("ground_truth", "None")
                succ = trace.get("is_correct", 0.0)
                rew = trace.get("reward", 0.0)


                strategy_preview = trace.get("strategy", "")[:50] + "..." if len(trace.get("strategy", "")) > 50 else trace.get("strategy", "")
                exec_count = len(trace.get("executor_outputs", []))
                commitment_preview = trace.get("final_commitment", "")[:50] + "..." if len(trace.get("final_commitment", "")) > 50 else trace.get("final_commitment", "")

                logger.info(f"Episode {episode}, t_env: {t_env}")
                logger.info(f"  ├─ Reward: {rew:.3f} | Success: {float(succ):.3f}")
                logger.info(f"  ├─ Pred: {pred} | GT: {gt}")
                logger.info(f"  ├─ Strategy: {strategy_preview}")
                logger.info(f"  ├─ Executors: {exec_count} agents")
                logger.info(f"  └─ Commitment: {commitment_preview}")

            if hasattr(config, 'wandb') and getattr(config.wandb, 'use_wandb', False) and WANDB_AVAILABLE and wandb.run is not None:
                try:
                    safe_stats = {f"train/{k}": v for k, v in (train_stats or {}).items() if isinstance(v, (int, float))}

                    if trace:
                        safe_stats["train/episode_reward"] = float(trace.get("reward", 0.0))
                        safe_stats["train/is_correct"] = float(trace.get("is_correct", 0.0))
                    wandb.log(safe_stats, step=episode)
                except Exception:
                    pass

        if use_theoretical_es and es_cfg_ns:
            if episode >= es_warmup_steps and isinstance(train_stats, dict) and trace:
                curr_loss = train_stats.get("loss_total", None)
                curr_commit = _norm_commit(trace.get("final_commitment", ""))
                curr_reward = float(trace.get("reward", 0.0))

                commit_match = bool(prev_commitment and curr_commit == prev_commitment)
                loss_delta = abs(curr_loss - prev_loss) if (curr_loss is not None and prev_loss is not None) else float("inf")

                # Default: text equality drives ΔC check; threshold kept for backward compatibility on loss
                cond_commit = commit_match
                cond_reward = curr_reward >= reward_threshold
                cond_loss = loss_delta <= loss_threshold

                if cond_commit and cond_reward and cond_loss:
                    es_ok_count += 1
                else:
                    es_ok_count = 0

                if es_ok_count >= es_patience:
                    logger.info(f"[EarlyStopping] Theoretical guard triggered at episode {episode} "
                                f"(ΔC<={commitment_threshold}, reward>={reward_threshold}, ΔL<={loss_threshold})")
                    break

                prev_loss = curr_loss if curr_loss is not None else prev_loss
                prev_commitment = curr_commit if curr_commit else prev_commitment

        elif stopper and isinstance(train_stats, dict):
            metric_val = train_stats.get(es_metric_name, None)
            if metric_val is None and trace:
                metric_val = trace.get("reward", None)
            if metric_val is not None:
                should_stop, is_best = stopper.step(float(metric_val))
                if is_best:
                    logger.info(f"[EarlyStopping] New best {es_metric_name}: {metric_val:.4f}")
                if should_stop:
                    logger.info(f"[EarlyStopping] Stop triggered at episode {episode} (metric={metric_val:.4f})")
                    break

        if episode > 0 and (episode % int(save_model_interval) == 0):
            save_path = Path(getattr(getattr(config, 'logging', SimpleNamespace()), 'checkpoint_path', './models')) / f"episode_{episode}"
            save_path.mkdir(parents=True, exist_ok=True)
            if hasattr(learner, "save_models"):
                learner.save_models(str(save_path))
                logger.info(f"Model saved at episode {episode}")
            else:
                logger.warning("Learner has no save_models(); skip saving.")


        steps_in_episode = int(trace.get("rounds", 1)) if trace else 1
        episode += 1
        t_env += max(1, steps_in_episode)


        max_episodes = getattr(config, 'max_episodes', None)
        if isinstance(max_episodes, int) and episode >= max_episodes:
            logger.info(f"Reached max_episodes={max_episodes}. Saving and exiting.")
            save_path = Path(getattr(getattr(config, 'logging', SimpleNamespace()), 'checkpoint_path', './models')) / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            if hasattr(learner, "save_models"):
                learner.save_models(str(save_path))
            break
    


  
    save_path = Path(getattr(getattr(config, 'logging', SimpleNamespace()), 'checkpoint_path', './models')) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(learner, "save_models"):
        learner.save_models(str(save_path))
    logger.info("Training completed. Final model saved.")

    total_time = time.time() - begin_time
    logger.info(f"Total training time: {total_time:.2f} seconds")


def run_test(config: SimpleNamespace, runner, learner, logger, device):
    logger.info("Starting test...")

   
    log_dir = Path(getattr(getattr(config, 'logging', SimpleNamespace()), 'log_path', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_test_path = log_dir / 'llm_traces_test.json'
    all_traces_test: List[Dict[str, Any]] = []

    nepisode = int(getattr(config, 'test_nepisode', 100))
    n_succ = 0
    rew_sum = 0.0

    for ep in range(nepisode):
        ep_batch = runner.run(test_mode=True)

       
        trace = getattr(runner, "get_last_trace", lambda: None)()
        if trace is not None:
            all_traces_test.append(trace)
            if bool(trace.get("is_correct", False)):
                n_succ += 1
            rew_sum += float(trace.get("reward", 0.0))

            try:
                with open(trace_test_path, "w", encoding="utf-8") as jf:
                    json.dump(all_traces_test, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write test traces JSON: {e}")

    acc = n_succ / max(1, nepisode)
    rew_avg = rew_sum / max(1, nepisode)
    logger.info(f"[TEST] episodes={nepisode} | acc={acc:.3f} | reward_avg={rew_avg:.3f}")
    logger.info("Test finished.")

# ------------------------
# wandb
# ------------------------
def setup_wandb(config: SimpleNamespace, logger):
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, skipping wandb initialization")
        return False
    if hasattr(config, 'wandb') and config.wandb.use_wandb:
        logger.info("Initializing wandb...")
        wandb.init(
            project=getattr(config.wandb, 'project', 'ECON-Framework'),
            entity=getattr(config.wandb, 'entity', None),
            tags=getattr(config.wandb, 'tags', None),
            config=dict(config.__dict__) if hasattr(config, '__dict__') else None,
            name=getattr(getattr(config, 'logging', SimpleNamespace()), 'experiment_name', 'econ_experiment')
        )
        logger.info("wandb initialized successfully")
        return True
    return False
