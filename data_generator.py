import os
import numpy as np
from torch.utils.data import DataLoader
import load_data as ld

def generate_data():
    root = './VOCdevkit/VOC2007'
    tempfilename_list = os.listdir(root + '/ImageSets/Main')
    for i, name in enumerate(tempfilename_list):
        tempfilename_list[i] = name[:name.find('_')]
    classList = np.array([name for name in tempfilename_list if 'tx' not in name])
    classList = np.append(['background'], np.unique(classList))

    train_data = ld.VOCSegDataset(True, [334,496], ld.img_transform)
    test_data = ld.VOCSegDataset(False, [334,496], ld.img_transform)
    train_d = DataLoader(train_data, 64, shuffle=True, num_workers=4)
    val_d = DataLoader(test_data, 128, shuffle=True, num_workers=4)
    return(train_d, val_d, train_data, test_data, classList)