# adapted from: https://github.com/facebookresearch/SlowFast

"""Argument parser functions."""

import argparse
import sys
import os
import time
from config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="sgshin-OracleAD"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Overwrite cfg.DATA.N_VAR and related variables according to dataset
    valid_datasets = {
        "SMD": 38,
        "MSL": 55,
        "PSM": 25,
        "SWaT": 51,
        "WADI": 123,
    }

    sliding_windows = {
    "PSM": 5,
    "SMD": 11,
    "SWaT": 447,
    "WADI": 100,  #wadi, msl은 논문에 없음
    "MSL": 100,
    }

    if cfg.DATA.NAME == 'SWaT':
        cfg.DATA.N_VAR = valid_datasets["SWaT"]
        cfg.TEST.SLIDING_WINDOW = sliding_windows["SWaT"]
    elif cfg.DATA.NAME == 'WADI':
        cfg.DATA.N_VAR = valid_datasets["WADI"]
        cfg.SCORER.TYPE = "cos"
        cfg.TEST.SLIDING_WINDOW = sliding_windows["WADI"]
    elif "SMD" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["SMD"]
        cfg.TEST.SLIDING_WINDOW = sliding_windows["SMD"]
    elif "MSL" in cfg.DATA.NAME:
        cfg.DATA.N_VAR = valid_datasets["MSL"]
        cfg.TEST.SLIDING_WINDOW = sliding_windows["MSL"]
    else:
        cfg.DATA.N_VAR = valid_datasets[cfg.DATA.NAME]
        cfg.TEST.SLIDING_WINDOW = sliding_windows.get(cfg.DATA.NAME, 100)
    
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cfg.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.DATA.NAME, date)
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, cfg.DATA.NAME, date)

    return cfg
