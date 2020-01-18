import os
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
    props = dict(boxes="cornflowerblue", whiskers="Black",
                 medians="DarkBlue", caps="k")
    for lmbd in df.nyst_reg.unique():
        fig, ax = plt.subplots(num=str(lmbd))
        df_l1 = df[df.nyst_reg == lmbd]
        for i, reg in enumerate(df_l1.trend_reg.unique()):
            df_l1tv = df_l1[df_l1.trend_reg == reg]
            df_l1tv.corr_full.plot(
                kind='box', ax=ax, positions=[(3*width)*i - width / 2])
            df_l1tv.corr_init.plot(
                kind='box', ax=ax, positions=[(3*width)*i + width / 2],
                color=props)

    plt.figure()
    df.plot(kind='box', y=['r2_full', 'r2_init'],
            patch_artist=True, color=props, showfliers=False)

    plt.show()
