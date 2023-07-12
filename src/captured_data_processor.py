import yaml
import json
import time
import logging
import joblib
import argparse
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from data_loader import captured_data_loader


def load_and_process(prob_config: ProblemConfig):

    logging.info("Load captured data")

    captured_x = captured_data_loader(prob_config)

    print(captured_x.info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    load_and_process(prob_config)
