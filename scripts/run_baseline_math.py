import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from run_baseline_no_bne import main as base_main  # noqa: E402


def main():
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", os.path.join(SCRIPT_DIR, "baseline_math_no_bne.yaml")])
    if "--log-dir" not in sys.argv:
        sys.argv.extend(["--log-dir", os.path.join(PROJECT_ROOT, "logs_baseline_math")])
    base_main()


if __name__ == "__main__":
    main()

