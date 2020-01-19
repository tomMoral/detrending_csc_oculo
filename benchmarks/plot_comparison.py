import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Plot CSC detrending performances from simulated oculo "
        "signals")
    parser.add_argument('--file', type=str, default=None,
                        help="# of run to perform")
    args = parser.parse_args()

    file_name = args.file
    if args.file is None:
        file_pattern = os.path.join(OUTPUT_DIR, "results*.pkl")
        file_list = glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)
    for col_name in df.columns:
        if 'corr' in col_name:
            df[col_name] = [c[0][0] for c in df[col_name]]

    width = .2
    props_full = dict(boxes="C0", whiskers="Black",
                      medians="DarkBlue", caps="k")
    props_init = dict(boxes="C1", whiskers="Black",
                      medians="DarkBlue", caps="k")
    props_no = dict(boxes="C2", whiskers="Black",
                    medians="DarkBlue", caps="k")
    for lmbd in df.nyst_reg.unique():
        fig_corr, ax_corr = plt.subplots(num=f"Corr lambda={lmbd}")
        fig_r2, ax_r2 = plt.subplots(num=f"R2 lambda={lmbd}")
        df_l1 = df[df.nyst_reg == lmbd]
        xticks = []
        for i, reg in enumerate(df_l1.trend_reg.unique()):
            pos = 3 * width * i
            patches = []
            df_l1tv = df_l1[df_l1.trend_reg == reg]
            if reg < 1:
                xticks += [(pos, reg)]
                df_l1tv.corr_full.plot(
                    kind='box', ax=ax_corr, positions=[pos - width / 2],
                    color=props_full, patch_artist=True, showfliers=False,
                    label='CSC with joint detrending')
                df_l1tv.corr_init.plot(
                    kind='box', ax=ax_corr, positions=[pos + width / 2],
                    color=props_init, patch_artist=True, showfliers=False,
                    label='detrending then CSC')
                df_l1tv.r2_full.plot(
                    kind='box', ax=ax_r2, positions=[pos - width / 2],
                    patch_artist=True, color=props_full, showfliers=False)
                df_l1tv.r2_init.plot(
                    kind='box', ax=ax_r2, positions=[pos + width / 2],
                    patch_artist=True, color=props_init, showfliers=False)
            else:
                xticks += [(pos, f"No reg")]
                df_l1tv.corr_full.plot(
                    kind='box', ax=ax_corr, positions=[pos],
                    color=props_no, patch_artist=True, showfliers=False,
                    label='CSC with no detrending')
                df_l1tv.r2_full.plot(
                    kind='box', ax=ax_r2, positions=[pos],
                    patch_artist=True, color=props_no, showfliers=False)
        xticks = np.array(xticks).T
        for ax in [ax_corr, ax_r2]:
            ax.set_xticks(xticks[0].astype(float))
            ax.set_xticklabels(xticks[1])
        ax_r2.set_yscale('log')

    plt.show()
