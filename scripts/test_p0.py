#!/usr/bin/env python3
"""
"""
import os, sys, json
sys.path.insert(0, "src")


import yaml

# Model directory to load (change this to your trained model path)
MODEL_DIR = "./models_gsm8k/final"  # Using existing trained model


def test(bne_rounds, tag):
    print(f"\n{'='*60}")
    print(f"{2 if bne_rounds==0 else 3}: {'Baseline' if bne_rounds==0 else f'P0 BNE {bne_rounds}round'} (10 episodes)")
    print("="*60 + "\n")

    import yaml, torch, os, re
    from types import SimpleNamespace
    from controllers.basic_mac import BasicMAC
    from runners.episode_runner import EpisodeRunner
    from utils.logging import get_logger
    from learners.q_learner import ECONLearner

    with open("scripts/config_p0.yaml") as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, str):
            s = os.path.expandvars(obj).strip()
            if re.fullmatch(r"\$\{[^}]+\}", s):
                return ""
            return s
        if isinstance(obj, list):
            return [expand_env_vars(x) for x in obj]
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        return obj

    config = expand_env_vars(config)

    config['test_nepisode'] = 10
    config['bne']['enabled'] = (bne_rounds > 0)
    config['bne']['refine_at_infer'] = (bne_rounds > 0)
    config['bne']['max_iterations_infer'] = bne_rounds

    def d2s(d):
        return SimpleNamespace(**{k: d2s(v) if isinstance(v, dict) else v for k, v in d.items()})

    args = d2s(config)
    logger = get_logger()
    runner = EpisodeRunner(args, logger)
    env_info = runner.get_env_info()

    n_agents = int(getattr(args, 'n_agents', 3))
    memory_dim = int(getattr(args, 'memory_dim', getattr(args, 'belief_dim', 128) + 9))
    setattr(args, "memory_dim", memory_dim)

    scheme = {
        "state": {"vshape": tuple(env_info.get("state_shape", (1,)))},
        "obs": {"vshape": tuple(env_info.get("obs_shape", (1024,))), "group": "agents", "dtype": torch.long},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info.get("n_actions", 2),), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "belief_states": {"vshape": (args.belief_dim,), "group": "agents", "dtype": torch.float32},
        "prompt_embeddings": {"vshape": (2,), "group": "agents", "dtype": torch.float32},
        "group_repr": {"vshape": (args.belief_dim,), "dtype": torch.float32},
        "commitment_embedding": {"vshape": (getattr(args, 'commitment_embedding_dim', 1024),), "dtype": torch.float32},
        "reward_al": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_ts": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward_cc": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "agent_memory": {"vshape": (memory_dim,), "group": "agents", "dtype": torch.float32},
    }
    groups = {"agents": n_agents}

    mac = BasicMAC(scheme, groups, args)
    runner.setup(scheme, groups, None, mac)
    learner = ECONLearner(mac, scheme, logger, args)
    if hasattr(runner, "set_alpha_provider"):
        runner.set_alpha_provider(learner)

    if os.path.exists(MODEL_DIR):
        try:
            learner.load_models(MODEL_DIR)
            print(f"✓ model loaded from: {MODEL_DIR}\n")
        except Exception as e:
            print(f"⚠ failed to load model: {e}\n")
    else:
        print(f"⚠ model directory not found: {MODEL_DIR}\n")

    traces, correct = [], 0
    for ep in range(10):
        runner.run(test_mode=True)
        t = runner.get_last_trace()
        if t:
            traces.append(t)
            if t.get('is_correct'): correct += 1
        if (ep+1) % 5 == 0:
            print(f"Episode {ep+1}/10: {'✓' if t.get('is_correct') else '✗'} 准确率={correct/(ep+1)*100:.1f}%")

    with open(f"logs_p0_test_{tag}.json", "w") as f:
        json.dump(traces, f)

    acc = correct/len(traces)*100 if traces else 0
    print(f"\n{tag}result: {correct}/{len(traces)} = {acc:.1f}%")


    if bne_rounds > 0:
        from collections import Counter
        meta = [t['commitment_metadata'] for t in traces if 'commitment_metadata' in t]
        if meta:
            methods = Counter([m['parse_method'] for m in meta])
            print(f"P0 Metadata: {dict(methods)}, JSON={methods.get('json',0)/len(meta)*100:.0f}%")

    return acc


acc_baseline = test(0, "baseline")
acc_p0 = test(3, "p0")
