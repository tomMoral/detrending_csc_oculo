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

    props = dict(boxes="cornflowerblue", whiskers="Black",
                 medians="DarkBlue", caps="k")

    plt.figure()
    df.plot(kind='box', y=['corr_full', 'corr_no', 'corr_init'],
            patch_artist=True, color=props, showfliers=False)

    plt.figure()
    df.plot(kind='box', y=['r2_full', 'r2_no', 'r2_init'],
            patch_artist=True, color=props, showfliers=False)

    plt.show()
