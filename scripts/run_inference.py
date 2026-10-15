"""Write the shared predictions file (parquet) used by both pairs.

Usage: python scripts/run_inference.py [--config path/to/config.yaml]
"""

import argparse

from common import CONFIG_PATH, load_config
from common.inference import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH, help="YAML config (default: common/config.yaml)")
    args = parser.parse_args()
    run_inference(load_config(args.config))
