import os

import glob
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

from pytz import NonExistentTimeError

def getInfoPath(datapath: str, infoFileName: str):
    dataFileName = os.path.basename(datapath)
    return datapath.replace(dataFileName, infoFileName)

def getIndexNameFromInfoData(info: pd.DataFrame):
    return "ok"

def getIndexNameFromPath(path: str, root: str):
    baseName = os.path.basename(path)
    return path.replace(baseName, '').replace(root, '')


def loadAllData(rootFolder, dataFileName, infoFileName, filter=None):
    datas = {}
    infos = {}

    csv_file_paths = glob.glob(os.path.join(rootFolder, "*/**/{}".format(dataFileName)), recursive=True)
    for csv_file_path in tqdm(csv_file_paths):
        csv_info__path = getInfoPath(csv_file_path, infoFileName)

        data_to_append = pandas.read_csv(csv_file_path)
        info_to_append = pandas.read_csv(csv_info__path)

        index = getIndexNameFromPath(csv_file_path, rootFolder)

        datas[index] = data_to_append
        infos[index] = info_to_append

        data_to_append = None
        info_to_append = None

    return datas, infos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the graphs all together',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help='Path to the roo folder where the CSVs are located')
    parser.add_argument('-dn', '--dataName', type=str, default='log.csv', help='The name filter used to select the csv containg the data') # used only for legend
    parser.add_argument('-in', '--infoName', type=str, default='params.csv', help='The name filter used to select the csv containg the train settings') # used only for legend
    parser.add_argument('-f', '--filter', type=str, default=None, help='String to use to ignore some paths in the folder') # used only for legend
    args = parser.parse_args()

    ROOT_FOLDER = args.folder
    DATA_FILE_NAME = args.dataName
    INFO_FILE_NAME = args.infoName
    PATH_FILTER = args.filter


    datas, infos = loadAllData(ROOT_FOLDER, DATA_FILE_NAME, INFO_FILE_NAME, PATH_FILTER)

    f, axs = plt.subplots(2, 2, figsize=(12, 12))
    for key, data in tqdm(datas.items()):
        sns.lineplot(data=data, x="epoch", y="mse_train", ax=axs[0][0], label=key)
        sns.lineplot(data=data, x="epoch", y="psnr_train", ax=axs[0][1])
        sns.lineplot(data=data, x="epoch", y="mse_train", ax=axs[1][0])
        sns.lineplot(data=data, x="epoch", y="psnr_val", ax=axs[1][1])

    scale = 0.8
    for row in range(2):
        for col in range(2):
            box = axs[row][col].get_position()
            x = box.x0 + ((box.width * (1 - scale)))
            y = box.y0 - ((box.height * (1 - scale)))
            axs[row][col].set_position([x, y, box.width * scale, box.height * scale])

    axs[0][0].legend(loc='center', bbox_to_anchor=(-0.10, 0.10), shadow=False, ncol=1, )
    f.savefig('plot.png', dpi=600)

    print("Done!")

    