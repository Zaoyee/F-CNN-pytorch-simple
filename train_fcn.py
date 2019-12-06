import torch.nn as nn
import fcn_model as md
import data_generator as dgtr
import datetime
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import acc_computer as acpt


def train(net):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

    train_d, val_d,\
    train_data, test_data,\
    classList = dgtr.generate_data()

    num_classes = len(classList)
    for e in range(1000):
        # if e > 0 and e % 50 == 0:
        #     optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0

        prev_time = datetime.datetime.now()
        for data in train_d:

            with torch.no_grad():
                im = data[0].cuda()
                label = data[1].cuda()
            # forward
            out = net(im)
            out = F.log_softmax(out, dim=1)  # (b, n, h, w)
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = acpt.label_accuracy(lbt, lbp, num_classes)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc

        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0
        for data in val_d:

            with torch.no_grad():
                im = data[0].cuda()
                label = data[1].cuda()
            # forward
            out = net(im)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, label)
            eval_loss += loss.data

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = acpt.label_accuracy(lbt, lbp, num_classes)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
    Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            e+1, train_loss / len(train_d), train_acc / len(train_data), train_mean_iu / len(train_data),
               eval_loss / len(val_d), eval_acc / len(test_data), eval_mean_iu / len(test_data)))


        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str )
    return(net)

