# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from lamcts import find_checkpoint

MINIMIZING_FUNCS = {"ackley100", "levy100", "rastrigin100"}


def plot(func, algs, tb_rlts, tb_title, cb_rslts, cb_title):
    minimize = func in MINIMIZING_FUNCS
    title = f"{func} ({'minimize' if minimize else 'maximize'})"
    fig, (tb_ax, cb_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
    fig.suptitle(title)

    y_pos = np.arange(len(algs))
    mean = np.array([tb_rlts[alg][0] for alg in algs])
    best = mean.min() if minimize else mean.max()
    error = np.array([tb_rlts[alg][1] for alg in algs])
    hbars = tb_ax.barh(y_pos[mean != best], mean[mean != best], xerr=error[mean != best], align='center')
    bhbars = tb_ax.barh(y_pos[mean == best], mean[mean == best], xerr=error[mean == best], align='center',
                        color='#01ff07')
    tb_ax.set_yticks(y_pos)
    tb_ax.set_yticklabels(algs)
    tb_ax.invert_yaxis()  # labels read top-to-bottom
    tb_ax.set_xlabel('Value')
    tb_ax.set_title(tb_title)
    tb_ax.bar_label(hbars, fmt='%.2f')
    tb_ax.bar_label(bhbars, fmt='%.2f')

    mean = np.array([cb_rslts[alg][0] for alg in algs])
    best = mean.min() if minimize else mean.max()
    error = np.array([cb_rslts[alg][1] for alg in algs])
    hbars = cb_ax.barh(y_pos[mean != best], mean[mean != best], xerr=error[mean != best], align='center')
    bhbars = cb_ax.barh(y_pos[mean == best], mean[mean == best], xerr=error[mean == best], align='center',
                        color='#01ff07')
    cb_ax.set_yticks(y_pos)
    cb_ax.set_yticklabels(algs)
    # cb_ax.invert_yaxis()  # labels read top-to-bottom
    cb_ax.set_xlabel('Value')
    cb_ax.set_title(cb_title)
    cb_ax.bar_label(hbars, fmt='%.2f')
    cb_ax.bar_label(bhbars, fmt='%.2f')

    # plt.subplots_adjust(left=0.16)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline comparison')
    parser.add_argument('--func', '-f', help='function to optimize', default="ackley100")

    args = parser.parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(f"{file_dir}/output/21-12-12/{args.func}_*.pkl")
    if len(files) == 0:
        print(f"No results found for {args.func}")
        exit(1)

    algs = (
    "random", "mcts-random", "cmaes", "mcts-cmaes", "nevergrad", "mcts-nevergrad", "bo", "mcts-bo", "mcts-turbo")

    min_tc = float('inf')
    all_stats = []
    tb_rslts = dict()
    tb_budget = 0
    cb_rslts = dict()
    cb_budget = 0
    for file in files:
        basename = os.path.basename(file)
        with open(file, "rb") as f:
            bm_args, stats = pickle.load(f)
        if bm_args.opt == 'mcts':
            alg = f"{bm_args.opt}-{bm_args.sampler}"
        else:
            alg = f"{bm_args.opt}"
        cps = []
        rslts = None
        if not math.isinf(bm_args.call_budget):
            rslts = cb_rslts
            cb_budget = int(bm_args.call_budget)
            for stat in stats:
                cps.append(find_checkpoint(stat.call_history, call_mark=bm_args.call_budget))
        else:
            rslts = tb_rslts
            tb_budget = int(bm_args.time_budget)
            for stat in stats:
                cps.append(find_checkpoint(stat.call_history, time_mark=bm_args.time_budget))
        if alg not in rslts:
            rslts[alg] = dict()
        cps.sort(key=lambda x: x.fx, reverse=False if args.func in MINIMIZING_FUNCS else True)
        fxs = np.array([cp.fx for cp in cps[:10]])
        rslts[alg] = (fxs.mean(), fxs.std())

    plot(args.func, algs, tb_rslts, f"{tb_budget}s Time Budget", cb_rslts, f"{cb_budget} Call Budget")
