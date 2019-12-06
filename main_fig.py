import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import data_generator as dgtr
import PIL.Image as Image
import numpy as np
import fcn_model_32s as fcn_32
import fcn_model_16s as fcn_16
import fcn_model as  fcn_8

# load the data
train_d, val_d, train_data, test_data, classList = dgtr.generate_data()
im, target = train_data[0]
# define the general prediction function
def predict(net, im, label):
    """
    :param net: The fcnn structure object
    :param im: the test image
    :param label: the target
    :return: the prediction and the colormap
    """
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    cm = np.array(colormap).astype('uint8')
    im = Variable(im.unsqueeze(0))
    out = net(im)
    pred = out.max(1)[1].squeeze().data.numpy()
    pred = cm[pred]
    return pred, cm[label.numpy()]


def set_invisible(figs):
    for i in range(0,4):
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)

# load three structures
net_32s = fcn_32.fcn_32()
net_16s = fcn_16.fcn_16()
net_8s = fcn_8.fcn()

# load the pre-trained parameters
net_32s.load_state_dict(torch.load('./model_params/fcn-32s.pkl',
                                   map_location=lambda storage, loc: storage))
net_16s.load_state_dict(torch.load('./model_params/fcn-16s.pkl',
                                   map_location=lambda storage, loc: storage))
net_8s.load_state_dict(torch.load('./model_params/fcn-8s.pkl',
                                  map_location=lambda storage, loc: storage))
net_32s.eval()
net_16s.eval()
net_8s.eval()

pred_32, label_32 = predict(net_32s, im, target)
pred_16, label_16 = predict(net_16s, im, target)
pred_8, label_8 = predict(net_8s, im, target)

### Make the figures
plt_1, figs = plt.subplots(1, 4, figsize=(12, 10))

figs[0].imshow(pred_32)
figs[1].imshow(pred_16)
figs[2].imshow(pred_8)
figs[3].imshow(label_16)
set_invisible(figs)
figs[0].set_title('FCN-32s',fontsize=20)
figs[1].set_title('FCN-16s',fontsize=20)
figs[2].set_title('FCN-8s',fontsize=20)
figs[3].set_title('Ground Truth',fontsize=20)
plt_1.show()
plt_1.savefig('./Figs/resultsfigs.png')
