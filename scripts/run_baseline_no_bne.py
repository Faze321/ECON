# scripts/run_baseline_no_bne.py
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))


def _write_temp_config(config_path: str, log_dir: str, episodes: int) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["test_nepisode"] = episodes
    config.setdefault("logging", {})
    config["logging"]["log_path"] = log_dir

    test_env_overrides = config.get("env_args_test")
    if isinstance(test_env_overrides, dict):
        base_env_args = config.get("env_args") or {}
        merged_env_args = dict(base_env_args)
        merged_env_args.update(test_env_overrides)
        config["env_args"] = merged_env_args

    temp_path = config_path.replace(".yaml", "_temp.yaml")
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return temp_path


def _empty_token_usage():
    return {
        "agents": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        },
        "coordinator": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        },
        "total": {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _without_reward_fields(record):
    if not isinstance(record, dict):
        return record

    reward_keys = {"reward", "reward_al", "reward_ts", "reward_cc", "r_al", "r_ts", "r_cc"}
    cleaned = {key: value for key, value in record.items() if key not in reward_keys}

    for trajectory_key in ("trajectory", "trajectories"):
        trajectories = cleaned.get(trajectory_key)
        if isinstance(trajectories, list):
            cleaned[trajectory_key] = [_without_reward_fields(item) for item in trajectories]

    return cleaned


def _summarize_traces(traces, record_type: str):
    records = [
        t for t in traces
        if not (isinstance(t, dict) and str(t.get("record_type", "")).endswith("_summary"))
    ]
    correct = sum(1 for t in records if t.get("is_correct"))
    total = len(records)
    accuracy = correct / total * 100 if total else 0.0
    token_usage = _empty_token_usage()

    for trace in records:
        usage = trace.get("token_usage", {}) if isinstance(trace, dict) else {}
        for role in ("agents", "coordinator"):
            role_usage = usage.get(role, {})
            token_usage[role]["prompt_tokens"] += int(role_usage.get("prompt_tokens", 0))
            token_usage[role]["completion_tokens"] += int(role_usage.get("completion_tokens", 0))
            token_usage[role]["total_tokens"] += int(role_usage.get("total_tokens", 0))
            token_usage[role]["requests"] += int(role_usage.get("requests", 0))

    token_usage["total"]["requests"] = token_usage["agents"]["requests"] + token_usage["coordinator"]["requests"]
    token_usage["total"]["prompt_tokens"] = token_usage["agents"]["prompt_tokens"] + token_usage["coordinator"]["prompt_tokens"]
    token_usage["total"]["completion_tokens"] = (
        token_usage["agents"]["completion_tokens"] + token_usage["coordinator"]["completion_tokens"]
    )
    token_usage["total"]["total_tokens"] = token_usage["agents"]["total_tokens"] + token_usage["coordinator"]["total_tokens"]

    records.append({
        "record_type": record_type,
        "correct": correct,
        "total": total,
        "accuracy": f"{accuracy:.1f}%",
        "token_usage": token_usage,
    })
    return records


def run_baseline(config_path: str, episodes: int, log_dir: str):
    from train import load_config, setup_experiment
    from utils.logging import get_logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_dir, experiment_name="baseline_no_bne", use_tensorboard=False)

    temp_config = _write_temp_config(config_path, log_dir, episodes)
    try:
        config = load_config(temp_config)
        runner, mac, learner, logger, device = setup_experiment(config, logger)

        traces = []
        for episode_idx in range(episodes):
            runner.run(test_mode=True)
            trace = runner.get_last_trace()
            if trace:
                traces.append(_without_reward_fields(trace))
            if (episode_idx + 1) % 5 == 0:
                correct = sum(1 for t in traces if t.get("is_correct"))
                logger.info(f"Episode {episode_idx + 1}/{episodes}: accuracy={correct / max(1, len(traces)) * 100:.1f}%")

        traces = _summarize_traces(traces, "baseline_summary")
        trace_path = os.path.join(log_dir, "llm_traces_baseline_no_bne.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2, ensure_ascii=False)

        summary = traces[-1]
        logger.info(f"Baseline accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']}")
        logger.info(f"Baseline agent tokens: {summary['token_usage']['agents']['total_tokens']}")
        logger.info(f"Baseline total LLM tokens: {summary['token_usage']['total']['total_tokens']}")
        logger.info(f"Trace saved to: {trace_path}")
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)


def main():
    parser = argparse.ArgumentParser(description="Run no-BNE 1-coordinator + 3-agent baseline.")
    parser.add_argument("--config", default=os.path.join(_SCRIPT_DIR, "baseline_no_bne.yaml"))
    parser.add_argument("--test-eps", type=int, default=30, help="test episode")
    parser.add_argument("--log-dir", default="logs_baseline_gsm8k")
    args = parser.parse_args()

    run_baseline(args.config, args.test_eps, args.log_dir)


if __name__ == "__main__":
    main()
