import os
from PIL import Image
import numpy as np
import torchvision.transforms as tfs
from torch.utils.data import Dataset
import torch

def read_list(train=True):
    root = './VOCdevkit/VOC2007'
    if train:
        txt_name = root + '/ImageSets/Segmentation/train.txt'
        with open(txt_name, 'r') as reader:
            image_name_list = reader.read().split()
        train_data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in image_name_list]
        label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in image_name_list]
        return train_data, label
    else:
        txt_name = root + '/ImageSets/Segmentation/val.txt'
        with open(txt_name, 'r') as reader:
            image_name_list = reader.read().split()
        test_data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in image_name_list]
        label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in image_name_list]
        return test_data, label

def image2label(im, cm2lbl):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')

def img_rndCrop(data, label, height, width):
    data, rectangle = tfs.RandomCrop((height, width))(data)
    label = tfs.FixedCrop(*rectangle)(label)
    return data, label

def img_transform(data, label, cropsize, cm2lbl):
    data, label = rand_Crop(data, label, cropsize)
    tf_pipeline = tfs.Compose([tfs.ToTensor(),
                          tfs.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    data = tf_pipeline(data)
    label = image2label(label, cm2lbl)
    label = torch.from_numpy(label)
    return data, label

def rand_Crop(img, label, crop_size):
    # crop
    start_x = np.random.randint(low=0,high=(img.size[0]-crop_size[0]+1))
    end_x = start_x + crop_size[0]
    start_y = np.random.randint(low=0,high=(img.size[1]-crop_size[1]+1))
    end_y = start_y + crop_size[1]
    crop = (start_x,start_y,end_x,end_y) # y1,x1,y2,x2
    img = img.crop(crop)
    label = label.crop(crop)
    return(img, label)

class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, trainflg, crop_size, transforms):

        colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                    [0, 192, 0], [128, 192, 0], [0, 64, 128]]
        self.cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            self.cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_list(train=trainflg)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[0] >= self.crop_size[0] and
                                        Image.open(im).size[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size, self.cm2lbl)
        return img, label

    def __len__(self):
        return len(self.data_list)
