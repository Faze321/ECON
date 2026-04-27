
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))




def run_training(config_file, episodes, log_dir, model_dir, logger):

    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    t_max_from_yaml = config.get('t_max', 'unknown')
    logger.info(f"\n{'='*60}")
    if episodes is not None:
        logger.info(f"start: {episodes} episodes")
    logger.info(f"{'='*60}\n")
    if episodes is not None:
        config['t_max'] = episodes

    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['checkpoint_path'] = model_dir
    config['logging']['log_path'] = log_dir

    temp_config = config_file.replace('.yaml', '_temp.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    try:
  
        from train import load_config, setup_experiment, run_training as train_run, setup_wandb
        
        
    
        cfg = load_config(temp_config)
        
       

        runner, mac, learner, logger, device = setup_experiment(cfg, logger)
        
      
        setup_wandb(cfg, logger)
        
    
        train_run(cfg, runner, learner, logger, device)
        
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)


    trace_path = os.path.join(log_dir, 'llm_traces_train.json')
    if os.path.exists(trace_path):
        with open(trace_path, 'r', encoding='utf-8') as f:
            traces = json.load(f)
        trace_records = [
            t for t in traces
            if not (isinstance(t, dict) and t.get('record_type') == 'train_summary')
        ]
        correct = sum(1 for t in trace_records if t.get('is_correct'))
        total = len(trace_records)
        acc = correct / total * 100 if total else 0.0
        token_usage_summary = {
            'agents': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'requests': 0,
            },
            'coordinator': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'requests': 0,
            },
            'total': {
                'requests': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            },
        }
        for trace in trace_records:
            token_usage = trace.get('token_usage', {}) if isinstance(trace, dict) else {}
            for role in ('agents', 'coordinator'):
                role_usage = token_usage.get(role, {})
                token_usage_summary[role]['prompt_tokens'] += int(role_usage.get('prompt_tokens', 0))
                token_usage_summary[role]['completion_tokens'] += int(role_usage.get('completion_tokens', 0))
                token_usage_summary[role]['total_tokens'] += int(role_usage.get('total_tokens', 0))
                token_usage_summary[role]['requests'] += int(role_usage.get('requests', 0))
        token_usage_summary['total']['requests'] = (
            token_usage_summary['agents']['requests'] + token_usage_summary['coordinator']['requests']
        )
        token_usage_summary['total']['prompt_tokens'] = (
            token_usage_summary['agents']['prompt_tokens'] + token_usage_summary['coordinator']['prompt_tokens']
        )
        token_usage_summary['total']['completion_tokens'] = (
            token_usage_summary['agents']['completion_tokens'] + token_usage_summary['coordinator']['completion_tokens']
        )
        token_usage_summary['total']['total_tokens'] = (
            token_usage_summary['agents']['total_tokens'] + token_usage_summary['coordinator']['total_tokens']
        )
        trace_records.append({
            'record_type': 'train_summary',
            'correct': correct,
            'total': total,
            'accuracy': f"{acc:.1f}%",
            'token_usage': token_usage_summary,
        })
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(trace_records, f, indent=2, ensure_ascii=False)
        logger.info(f"\nTrain accuracy: {correct}/{total} = {acc:.1f}%")
        logger.info(f"Train agent tokens: {token_usage_summary['agents']['total_tokens']}")
        logger.info(f"Train total LLM tokens: {token_usage_summary['total']['total_tokens']}")

def run_testing(config_file, episodes, bne_rounds, log_dir, model_dir, tag, logger):
 
    bne_mode = "mute" if bne_rounds == 0 else f"{bne_rounds}轮"
    logger.info(f"\n{'='*60}")
    logger.info(f"test: {episodes} episodes (BNE {bne_mode})")
    logger.info(f"{'='*60}\n")

    import yaml
    import os
    import re
    from types import SimpleNamespace

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables (fix for API key issue)
    def expand_env_vars(obj):
        if isinstance(obj, str):
            s = os.path.expandvars(obj).strip()
            # If still looks like ${VAR}, return empty
            if re.fullmatch(r"\$\{[^}]+\}", s):
                return ""
            return s
        if isinstance(obj, list):
            return [expand_env_vars(x) for x in obj]
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        return obj

    config = expand_env_vars(config)

    test_env_overrides = config.get('env_args_test')
    if isinstance(test_env_overrides, dict):
        base_env_args = config.get('env_args') or {}
        merged_env_args = dict(base_env_args)
        merged_env_args.update(test_env_overrides)
        config['env_args'] = merged_env_args

    config['test_nepisode'] = episodes
    config['bne']['enabled'] = (bne_rounds > 0)
    config['bne']['refine_at_infer'] = (bne_rounds > 0)
    config['bne']['max_iterations_infer'] = bne_rounds

    def dict_to_sns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        return d

    args_obj = dict_to_sns(config)

    import torch
    from controllers.basic_mac import BasicMAC
    from runners.episode_runner import EpisodeRunner
    from learners.q_learner import ECONLearner

    runner = EpisodeRunner(args_obj, logger)
    env_info = runner.get_env_info()


    n_agents = int(getattr(args_obj, 'n_agents', 3))
    memory_dim = int(getattr(args_obj, 'memory_dim', getattr(args_obj, 'belief_dim', 128) + 9))
    setattr(args_obj, "memory_dim", memory_dim)
    scheme = {
        "state": {"vshape": tuple(env_info.get("state_shape", (1,)))},
        "obs": {"vshape": tuple(env_info.get("obs_shape", (1024,))), "group": "agents", "dtype": torch.long},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info.get("n_actions", 2),), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "belief_states": {"vshape": (args_obj.belief_dim,), "group": "agents", "dtype": torch.float32},
        "prompt_embeddings": {"vshape": (2,), "group": "agents", "dtype": torch.float32},
        "group_repr": {"vshape": (args_obj.belief_dim,), "dtype": torch.float32},
        "commitment_embedding": {"vshape": (getattr(args_obj, 'commitment_embedding_dim', 1024),), "dtype": torch.float32},
        "reward_al": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_ts": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_cc": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "agent_memory": {"vshape": (memory_dim,), "group": "agents", "dtype": torch.float32},
    }
    groups = {"agents": n_agents}

    mac = BasicMAC(scheme, groups, args_obj, logger)
    runner.setup(scheme, groups, None, mac)
    learner = ECONLearner(mac, scheme, logger, args_obj)
    if hasattr(runner, "set_alpha_provider"):
        runner.set_alpha_provider(learner)


    if os.path.exists(model_dir):
        try:
            learner.load_models(model_dir)
            logger.info(f"✓ model loaded: {model_dir}")
        except:
            logger.warning(f"⚠ random")

 
    traces = []
    correct = 0

    for ep_idx in range(episodes):
        batch = runner.run(test_mode=True)
        trace = runner.get_last_trace()
        if trace:
            traces.append(trace)
            if trace.get('is_correct'):
                correct += 1

            if (ep_idx + 1) % 5 == 0:
                acc = correct / (ep_idx + 1) * 100
                mark = "✓" if trace.get('is_correct') else "✗"
                print(f"Episode {ep_idx+1}/{episodes}: {mark} accuracy={acc:.1f}%")

    total = len(traces)
    acc = correct / total * 100 if total else 0.0
    token_usage_summary = {
        'agents': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'requests': 0,
        },
        'coordinator': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'requests': 0,
        },
        'total': {
            'requests': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        },
    }
    for trace in traces:
        token_usage = trace.get('token_usage', {}) if isinstance(trace, dict) else {}
        for role in ('agents', 'coordinator'):
            role_usage = token_usage.get(role, {})
            token_usage_summary[role]['prompt_tokens'] += int(role_usage.get('prompt_tokens', 0))
            token_usage_summary[role]['completion_tokens'] += int(role_usage.get('completion_tokens', 0))
            token_usage_summary[role]['total_tokens'] += int(role_usage.get('total_tokens', 0))
            token_usage_summary[role]['requests'] += int(role_usage.get('requests', 0))
    token_usage_summary['total']['requests'] = (
        token_usage_summary['agents']['requests'] + token_usage_summary['coordinator']['requests']
    )
    token_usage_summary['total']['prompt_tokens'] = (
        token_usage_summary['agents']['prompt_tokens'] + token_usage_summary['coordinator']['prompt_tokens']
    )
    token_usage_summary['total']['completion_tokens'] = (
        token_usage_summary['agents']['completion_tokens'] + token_usage_summary['coordinator']['completion_tokens']
    )
    token_usage_summary['total']['total_tokens'] = (
        token_usage_summary['agents']['total_tokens'] + token_usage_summary['coordinator']['total_tokens']
    )
    traces.append({
        'record_type': 'test_summary',
        'correct': correct,
        'total': total,
        'accuracy': f"{acc:.1f}%",
        'token_usage': token_usage_summary,
    })

    trace_path = os.path.join(log_dir, f'llm_traces_test_{tag}.json')
    with open(trace_path, 'w', encoding='utf-8') as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    logger.info(f"\n ({tag}): {correct}/{total} = {acc:.1f}%")
    logger.info(f"Test agent tokens ({tag}): {token_usage_summary['agents']['total_tokens']}")
    logger.info(f"Test total LLM tokens ({tag}): {token_usage_summary['total']['total_tokens']}")

    if bne_rounds > 0 and traces:
        has_meta = sum(1 for t in traces if 'commitment_metadata' in t)
        if has_meta > 0:
            from collections import Counter
            methods = [t['commitment_metadata']['parse_method'] for t in traces if 'commitment_metadata' in t]
            method_dist = Counter(methods)
            json_rate = method_dist.get('json', 0) / has_meta * 100
            logger.info(f"{json_rate:.1f}% {dict(method_dist)}")

    return acc

def main():
    parser = argparse.ArgumentParser(description='ECON')
    parser.add_argument('--train-eps', type=int, default=100, )
                  
    parser.add_argument('--test-eps', type=int, default=30, help='test episode')
    parser.add_argument('--config', default=os.path.join(_PROJECT_ROOT, 'scripts', 'config_p0.yaml'),
                      )
    parser.add_argument('--log-dir', default='logs_p0_test')
    parser.add_argument('--model-dir', default='models_p0_test')
    args = parser.parse_args()


    if not os.path.isabs(args.config) and not os.path.exists(args.config):
        candidate = os.path.join(_PROJECT_ROOT, args.config)
        if os.path.exists(candidate):
            args.config = candidate

    Path(args.log_dir).mkdir(exist_ok=True)
    Path(args.model_dir).mkdir(exist_ok=True)

    from utils.logging import get_logger
    logger = get_logger(args.log_dir)

    run_training(args.config, args.train_eps, args.log_dir, args.model_dir, logger)

    print("="*60 + "\n")

    # Run testing with BNE
    run_testing(args.config, args.test_eps, 3, args.log_dir, args.model_dir, "bne_3rounds", logger)


if __name__ == '__main__':
    main()
