from __future__ import print_function
import argparse
import random
import gc
import os
import warnings
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import scipy.io as scio
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from load_pd_images import pd_data
from region_model import CNN_V1

parser = argparse.ArgumentParser(description='pd_detection')

parser.add_argument('--aug', type=int, default=1,
                    help='data augmentation strategy')
parser.add_argument('--datapath', type=str, default='/home/choumy/PD_fa_0607/',
                    help='root folder for data.It contains two \
                    sub-directories train and val')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--model', type=int, default=0,
                    help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limlsit')
parser.add_argument('--data_format', type=int, default=2,
                    help='upper epoch limlsit')
parser.add_argument('--device', type=str, default='0',
                    help='choose gpu')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
data_format = args.data_format

def file_name(root_path):  

    for root, dirs, files in os.walk(root_path): 
        NL_FA = []
        PD_FA = []
        
        for i in range(len(files)):

            if files[i][0] == 'N' and files[i][2] == 'F' and files[i][4] != 'a':
                NL_FA.append(files[i])
#            elif files[i][0] == 'P' and files[i][2] == 'F':
                PD_FA.append('PDFA'+files[i].split('A')[1])
        # print(NL_FA)
        # print(len(NL_FA))
        return NL_FA, PD_FA


def load_data(PD_file, NL_file, PD_data, NL_data):
    # load data
    if data_format == 2:
        data = scio.loadmat(PD_file)
        x_data_pd = data[PD_data]
        data = scio.loadmat(NL_file)
        x_data_nl = data[NL_data]
        x_data_pd = x_data_pd.transpose(3, 0, 1, 2)
        x_data_nl = x_data_nl.transpose(3, 0, 1, 2)
        x_data_pd = shuffle(x_data_pd, random_state = 5)
        x_data_nl = shuffle(x_data_nl, random_state = 5)
        x_data_pd = shuffle(x_data_pd, random_state = 10)
        x_data_nl = shuffle(x_data_nl, random_state = 10)

        x_data = np.concatenate((x_data_nl, x_data_pd), axis=0)
        data_var = x_data.var()
        data_mean = x_data.mean()
        x_data_nl = (x_data_nl - data_mean) / np.sqrt(data_var)
        x_data_pd = (x_data_pd - data_mean) / np.sqrt(data_var)
        del x_data
        gc.collect()

        x_cv_pd = x_data_pd
        x_cv_nl = x_data_nl
        del x_data_pd, x_data_nl
        gc.collect()
#             x_cv_pd = x_data_pd[0:337, :, :, :]
#             x_cv_nl = x_data_nl[0:220, :, :, :]
#             x_test_pd = x_data_pd[277:337, :, :, :]
#             x_test_nl = x_data_nl[180:220, :, :, :]
    else:
        with h5py.File(PD_file, 'r') as file:
            x_data_pd = np.array(file[PD_data])
        with h5py.File(NL_file, 'r') as file:
            x_data_nl = np.array(file[NL_data])   
        x_data_pd = x_data_pd.transpose(0, 3, 2, 1)
        x_data_nl = x_data_nl.transpose(0, 3, 2, 1)
        # x_data_pd = shuffle(x_data_pd, random_state = 5)
        # x_data_nl = shuffle(x_data_nl, random_state = 5)
        x_data_pd = shuffle(x_data_pd, random_state = 10)
        x_data_nl = shuffle(x_data_nl, random_state = 10)

        x_data = np.concatenate((x_data_nl, x_data_pd), axis=0)
        data_var = x_data.var()
        data_mean = x_data.mean()
        x_data_nl = (x_data_nl - data_mean) / np.sqrt(data_var)
        x_data_pd = (x_data_pd - data_mean) / np.sqrt(data_var)
        del x_data
        gc.collect()
#             x_cv_pd = x_data_pd[0:304, :, :, :]
#             x_cv_nl = x_data_nl[0:204, :, :, :]
#             x_test_pd = x_data_pd[304:364, :, :, :]
        x_cv_pd = x_data_pd
        x_cv_nl = x_data_nl
        del x_data_pd, x_data_nl
        gc.collect()
    print(x_cv_pd.shape)
    print(x_cv_nl.shape)

    return  x_cv_pd, x_cv_nl


def pd_image_load(x_cv_pd, x_cv_nl, is_train=0, fold=0):
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_cv_pd)
    train_pd_index = []  
    val_pd_index = []
    for train_index, val_index in kf.split(x_cv_pd):
        train_pd_index.append(train_index)
        val_pd_index.append(val_index)
    kf.get_n_splits(x_cv_nl)
    train_nl_index = []
    val_nl_index = []
    for train_index, val_index in kf.split(x_cv_nl):
        train_nl_index.append(train_index)
        val_nl_index.append(val_index)

    if is_train == 0:
        #---------combine train set---------   
        x_train_nl = x_cv_nl[train_nl_index[fold], :, :, :]
        x_train_pd = x_cv_pd[train_pd_index[fold], :, :, :]
        # x_train_nl = x_cv_nl
        # x_train_pd = x_cv_pd
        x_train_nl_copy = x_train_nl[random.sample(range(
            x_train_nl.shape[0]), x_train_pd.shape[0] - x_train_nl.shape[0]), :, :, :]
        #____balanced train set____
        x_train = np.concatenate((x_train_nl, x_train_nl_copy, x_train_pd), axis=0)
        y_train = np.zeros(x_train.shape[0], int)
        y_train[x_train_pd.shape[0]:] = 1
        #____unbalanced train set____
        # x_train = np.concatenate((x_train_nl, x_train_pd), axis=0)
        # y_train = np.zeros(x_train.shape[0], int)
        # y_train[x_train_nl.shape[0]:] = 1

        #---------combine val set---------
        x_val_nl = x_cv_nl[val_nl_index[fold], :, :, :]
        x_val_pd = x_cv_pd[val_pd_index[fold], :, :, :]
        x_val = np.concatenate((x_val_nl, x_val_pd), axis=0)
        y_val = np.zeros(x_val.shape[0], int)
        y_val[x_val_nl.shape[0]:] = 1
        return x_train.astype(np.float32), x_val.astype(np.float32), y_train, y_val

    elif is_train == 1:
        #---------combine val set---------
        x_val_nl = x_cv_nl[val_nl_index[fold], :, :, :]
        x_val_pd = x_cv_pd[val_pd_index[fold], :, :, :]
        x_val = np.concatenate((x_val_nl, x_val_pd), axis=0)
        y_val = np.zeros(x_val.shape[0], int)
        y_val[x_val_nl.shape[0]:] = 1
        return x_val.astype(np.float32), y_val
    # elif is_train == 2:
    #     #---------combine test set---------
    #     x_test = np.concatenate((x_test_nl, x_test_pd), axis=0)
    #     y_test = np.zeros(x_test.shape[0], int)
    #     y_test[x_test_nl.shape[0]:] = 1
    #     return x_test.astype(np.float32), y_test


def generate_net_parameter(x_train):
    temp1 = [2, 2, 2]
    temp2 = [2, 2, 2]
    temp3 = [2, 2, 2]
    if x_train.shape[1] < 8:
        temp2[0] = 1
        temp2[1] = 1
        temp2[2] = 1
    elif x_train.shape[1] < 16:
        temp2[1] = 1
        temp2[2] = 1
    elif x_train.shape[1] < 32:
        temp2[2] = 1

    if x_train.shape[2] < 8:
        temp3[0] = 1
        temp3[1] = 1
        temp3[2] = 1
    elif x_train.shape[2] < 16:
        temp3[1] = 1
        temp3[2] = 1
    elif x_train.shape[2] < 32:
        temp3[2] = 1

    if x_train.shape[3] < 8:
        temp1[0] = 1
        temp1[1] = 1
        temp1[2] = 1
    elif x_train.shape[3] < 16:
        temp1[1] = 1
        temp1[2] = 1
    elif x_train.shape[3] < 32:
        temp1[2] = 1

    return temp1, temp2, temp3

def main(epochs = 70, num_classes = 2, save = True,
          is_singleData = False, single_region_str = '44'):
    # choose training model: train single mdoel or multiple models
    NL_FA, PD_FA  = file_name(args.datapath)
    if is_singleData:
        times = 1
    else:
        times = len(NL_FA)

    for i in range(times):
        arrays = {}
        print('Training model:' + str(i))
        # choose data
        if is_singleData:  
            if data_format == 0:
                name_file_NLFA = 'NLFAmedia' + single_region_str + '.mat'
                name_file_PDFA = 'PDFAmedia' + single_region_str + '.mat'
            elif (data_format == 1) | (data_format == 2):
                name_file_NLFA = 'NLFA' + single_region_str + '.mat'
                name_file_PDFA = 'PDFA' + single_region_str + '.mat'

            PD_FA[i] = name_file_PDFA
            NL_FA[i] = name_file_NLFA
            print('Data name:' + PD_FA[i])
        else:
            print('Data name:' + PD_FA[i])

        # create folders to save trained model and running details
        path = args.datapath+'2018_1_17/'+'pytorch_aug/'+PD_FA[i]
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        # load data address
        PD_file = args.datapath + PD_FA[i]
        NL_file = args.datapath + NL_FA[i]
        PD_data = PD_FA[i].split('.')[0]
        NL_data = NL_FA[i].split('.')[0]
        x_cv_pd, x_cv_nl = load_data(PD_file, NL_file, PD_data, NL_data)

        for fold in range(10):
            foldpath = path + '/' + 'fold' + str(fold)
            isExists=os.path.exists(foldpath)
            if not isExists:
                os.makedirs(foldpath)
            train_samples, val_samples, train_targets, val_targets = pd_image_load(
                x_cv_pd, x_cv_nl, fold=fold)
            gc.collect()
            print('------------------------------------------')
            print('fold:', fold)
            print('------------------------------------------')
            trainset = pd_data(samples=train_samples, targets=train_targets,
                               is_train=0,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # normalize,
                               ]))
            valset = pd_data(samples=val_samples, targets=val_targets,
                             is_train=1,
                             transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # normalize,
                             ]))

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

            valloader = torch.utils.data.DataLoader(valset, batch_size=20,
                                                    shuffle=False, num_workers=4)

            temp1, temp2, temp3 = generate_net_parameter(train_samples)

            for times in range(10):
                print('------------------------------------------')
                print('times:', times)
                print('------------------------------------------')
                subpath = foldpath + str(times) + '/'
                isExists=os.path.exists(subpath)
                if not isExists:
                    os.makedirs(subpath)
                net = CNN_V1(temp1, temp2, temp3)
                net = net.cuda()
                # print(net)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)                    
                best_result = 0.0
                for epoch in range(args.epochs):
                    net.train()
                    # adjust_learning_rate(optimizer, epoch)
                    scheduler.step()
                    for i, data in enumerate(trainloader, 0):
                        # get the inputs
                        inputs, labels = data
                        inputs = inputs.unsqueeze(1)
                        inputs = inputs.float().cuda()
                        labels = labels.cuda()
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        print('[%d, %5d] loss: %.4f' %
                            (epoch + 1, i + 1, loss.item()))
                        del inputs, labels, outputs, loss
                    if epoch % 1 == 0:
                        with torch.no_grad(): 
                            correct = 0.0
                            total = 0.0
                            net.eval()
                            for data in valloader:
                                inputs, labels = data
                                inputs = inputs.unsqueeze(1)
                                outputs = net(inputs.float().cuda())

                                _, predicted = torch.max(outputs.data, 1)
                                predicted = predicted.cpu()
                                total += labels.size(0)
                                correct += (predicted == labels).sum()
                            print('Accuracy of the network on the val set: %.2f %%' %
                                (100. * correct.float() / total))
                            if best_result < correct:
                                torch.save(net.state_dict(),
                                        subpath + 'pd_classifier.pth')
                                best_result = correct
                                print('This is the best model')
                            del inputs, labels, outputs, correct, predicted
                print('Finished Training')

                print('The best accuracy of the network on the val sets: %.2f %%' %
                    (100. * best_result.float() / total))
                del best_result, total
                del optimizer, net, criterion
                torch.cuda.empty_cache()
                gc.collect()
            del trainset, trainloader, valset, valloader
            torch.cuda.empty_cache()
            gc.collect()

main()
