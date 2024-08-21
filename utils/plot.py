import os

import glob
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_psnrs(batch: int, lr: float, epochs: int, plot_dir: str, metrics: pd.DataFrame) -> None:
    """ Plot the losses and the psnr
        Input:
            - batch
            - lr
            - epochs
            aqaaa
            - plot_dir 
            - metrics
    """
    plt.figure(1)
    plt.plot(metrics["mse_train"].values, label = "Train loss")
    plt.plot(metrics["mse_val"].values, label = "Valid loss")
    plt.title(
        f"Batch Size: {batch} | Learning Rate: {lr} | Epochs: {epochs} ")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "fig_losses.png"))

    plt.figure(2)
    plt.plot(metrics["psnr_train"].values, label = "Train PSNR")
    plt.plot(metrics["psnr_val"].values, label = "Valid PSNR")
    plt.title(
        f"Batch Size: {batch} | Learning Rate: {lr} | Epochs: {epochs} ")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "psnr.png"))
    plt.close("all")

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotter for csv graphs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_file', type=str, help='Path to the folder where the CSV file is located')
    parser.add_argument('-b', '--batch', type=int, default=2, help='Batch size of Reference') # used only for legend
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of  the training epoch ') # used only for legend
    parser.add_argument('--lr', type=float, default=10e-5, help='Learning rate of Reference') # used only for legend
 
    args = parser.parse_args()

    csv_file_paths = glob.glob(os.path.normpath(args.csv_file) + "*/**/log.csv")
    for csv_file_path in csv_file_paths:
        print("Processing: " + csv_file_path)
        out_dir = os.path.dirname(csv_file_path)

        data = pandas.read_csv(csv_file_path)

        plot_psnrs(args.batch, args.lr, args.epoch, out_dir, metrics=data)

        data = None
