from __future__ import print_function
import argparse
import random
import gc
import os
import warnings
import h5py
import xlwt

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
parser.add_argument('--datapath', type=str, default='/data/zhou/PD_NL0429/',
                    help='root folder for data.It contains two \
                    sub-directories train and val')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--model', type=int, default=0,
                    help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=250,
                    help='upper epoch limlsit')
parser.add_argument('--data_format', type=int, default=2,
                    help='upper epoch limlsit')
parser.add_argument('--device', type=str, default='0',
                    help='choose gpu')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
data_format = args.data_format

wb = xlwt.Workbook(encoding = 'ascii')
ws_0 = wb.add_sheet('accuracy')
ws_1 = wb.add_sheet('validation') 
ws_2 = wb.add_sheet('test')
ws_0.write(0,0,'Region')
ws_1.write(0,0,'Region')
ws_2.write(0,0,'Region')

def file_name(root_path):  

    for root, dirs, files in os.walk(root_path): 
        NL_FA = []
        PD_FA = []
        PD_FA = ['PDFA58.mat','PDFA46.mat','PDFA74.mat','PDFA8.mat','PDFA65.mat','PDFA57.mat','PDFA2.mat','PDFA64.mat','PDFA44.mat','PDFA16.mat','PDFA10.mat','PDFA13.mat','PDFA69.mat','PDFA71.mat','PDFA72.mat','PDFA52.mat','PDFA3.mat','PDFA36.mat','PDFA33.mat','PDFA83.mat','PDFA25.mat','PDFA87.mat']
        for i in range(len(PD_FA)):
#            elif files[i][0] == 'P' and files[i][2] == 'F':
             NL_FA.append('NLFA'+PD_FA[i].split('A')[1])
        # print(NL_FA)
        # print(len(NL_FA))
        return NL_FA, PD_FA


def load_data(PD_file, NL_file, PD_data, NL_data):
    # load data
    if data_format == 2:
        with h5py.File(PD_file, 'r') as file:
            x_data_pd = np.array(file[PD_data])
        with h5py.File(NL_file, 'r') as file:
            x_data_nl = np.array(file[NL_data])
        x_data_pd = x_data_pd.transpose(0, 3, 2, 1)
        x_data_nl = x_data_nl.transpose(0, 3, 2, 1)
        x_data_pd = shuffle(x_data_pd, random_state = 5)
        x_data_nl = shuffle(x_data_nl, random_state = 5)
        x_data_pd = shuffle(x_data_pd, random_state = 10)
        x_data_nl = shuffle(x_data_nl, random_state = 10)

        print(x_data_nl.shape)
        print(x_data_pd.shape)
        x_data = np.concatenate((x_data_nl, x_data_pd), axis=0)
        data_var = x_data.var()
        data_mean = x_data.mean()
        x_data_nl = (x_data_nl - data_mean) / np.sqrt(data_var)
        x_data_pd = (x_data_pd - data_mean) / np.sqrt(data_var)
        del x_data
        gc.collect()

        x_cv_pd = x_data_pd[0:248, :, :, :]
        x_cv_nl = x_data_nl[0:184, :, :, :]
        x_test_pd = x_data_pd[248:305, :, :, :]
        x_test_nl = x_data_nl[184:227, :, :, :]

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
        x_cv_pd = x_data_pd[0:248, :, :, :]
        x_cv_nl = x_data_nl[0:184, :, :, :]
        del x_data_pd, x_data_nl
        gc.collect()
    print(x_cv_pd.shape)
    print(x_cv_nl.shape)

    return  x_cv_pd, x_cv_nl, x_test_pd, x_test_nl


def pd_image_load(x_cv_pd, x_cv_nl, x_test_pd, x_test_nl, is_train=0, fold=0):
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

        x_test = np.concatenate((x_test_nl, x_test_pd), axis=0)
        y_test = np.zeros(x_test.shape[0], int)
        y_test[x_test_nl.shape[0]:] = 1
        return x_val.astype(np.float32), x_test.astype(np.float32), \
               y_val, y_test
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

    for i in range(90):
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
            ws_type = PD_FA[i]
            print('Data name:' + PD_FA[i])
        else:
            ws_type = PD_FA[i]
            print('Data name:' + PD_FA[i])

        # create folders to save trained model and running details
        path = args.datapath+'2019_6_13/'+'pytorch_aug/'+PD_FA[i]
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        # load data address
        PD_file = args.datapath + PD_FA[i]
        NL_file = args.datapath + NL_FA[i]
        PD_data = PD_FA[i].split('.')[0]
        NL_data = NL_FA[i].split('.')[0]
        x_cv_pd, x_cv_nl, x_test_pd, x_test_nl = load_data(PD_file, NL_file, PD_data, NL_data)
        
        ws_0.write(1+i, 0, ws_type)
        ws_1.write(0, 1+i, ws_type)  
        ws_2.write(0, 1+i, ws_type) 
        sum_sign_val = 0
        sum_sign_test = 0
        for fold in range(10):
            foldpath = path + '/' + 'fold' + str(fold)
            isExists=os.path.exists(foldpath)
            if not isExists:
                os.makedirs(foldpath)
            val_samples, test_samples, val_targets, test_targets = pd_image_load(
                x_cv_pd, x_cv_nl, x_test_pd, x_test_nl, is_train=1, fold=fold)
            gc.collect()
            print('------------------------------------------')
            print('fold:', fold)
            print('------------------------------------------')
            # trainset = pd_data(samples=train_samples, targets=train_targets,
            #                    is_train=0,
            #                    transform=transforms.Compose([
            #                         transforms.ToTensor(),
            #                         # normalize,
            #                    ]))
            valset = pd_data(samples=val_samples, targets=val_targets,
                             is_train=1,
                             transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # normalize,
                             ]))
            testset = pd_data(samples=test_samples, targets=test_targets,
                             is_train=1,
                             transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # normalize,
                             ]))

            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            #                                           shuffle=True, num_workers=2)

            valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                    shuffle=False, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                    shuffle=False, num_workers=2)

            temp1, temp2, temp3 = generate_net_parameter(val_samples)
            
            result_list = []
            for times in range(3):
                print('------------------------------------------')
                print('times:', times)
                print('------------------------------------------')
                subpath = foldpath + str(times) + '/'
                isExists=os.path.exists(subpath)
                if not isExists:
                    os.makedirs(subpath)
                net = CNN_V1(temp1, temp2, temp3)
                state_dict = torch.load(
                    subpath + 'pd_classifier.pth')
                net.load_state_dict(state_dict)                
                net = net.cuda()                 

                with torch.no_grad(): 
                    correct = 0.0
                    total = 0.0
                    net.eval()
                    for data in valloader:
                        inputs, labels = data
                        inputs = inputs.unsqueeze(1)
                        outputs, _ = net(inputs.float().cuda())

                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.cpu()
                        total += labels.size(0)
                        # print('labels:', labels)
                        # print('predicted:', predicted)
                        correct += (predicted == labels).sum()
                    result_list.append((100. * correct.float().item() / total))
                    # print('Accuracy of the network on the val set: %.2f %%' %
                    #     (100. * correct.float().item() / total))
                del inputs, labels, outputs, correct, predicted
                del total
                torch.cuda.empty_cache()
                gc.collect()

            max_result = max(result_list)
            max_num = result_list.index(max_result)
            subpath = foldpath + str(max_num) + '/'
            state_dict = torch.load(
                subpath + 'pd_classifier.pth')
            net.load_state_dict(state_dict) 
            with torch.no_grad(): 
                correct = 0.0
                total = 0.0
                net.eval()
                sign_feature = 0
                for data in valloader:
                    inputs, labels = data
                    inputs = inputs.unsqueeze(1)
                    outputs, features  = net(inputs.float().cuda())
                    if sign_feature == 0:
                        FT = features.detach().cpu().numpy()
                    if sign_feature == 1:
                        FT = np.concatenate((FT, features.detach().cpu().numpy()), axis=0)

                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu()
                    total += labels.size(0)
                    # print('labels:', labels)
                    # print('predicted:', predicted)
                    correct += (predicted == labels).sum()
                    sign_feature = 1

                scio.savemat(args.datapath+'2019_6_13/' + 'val_' + str(fold) + '_'  + PD_FA[i], {'data':FT})
                result_list.append((100. * correct.float().item() / total))
                # print('Accuracy of the network on the val set: %.2f %%' %
                #     (100. * correct.float().item() / total))
            ws_0.write(i+1, 1+fold, (100. * correct.float().item() / total))
            print('Accuracy of the network on the val set: %.2f %%' %
                (100. * correct.float().item() / total))
            del inputs, labels, outputs, correct, predicted
            del total
            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad(): 
                correct = 0.0
                total = 0.0
                net.eval()
                for data in valloader:
                    sum_sign_val += 1
                    inputs, labels = data
                    inputs = inputs.unsqueeze(1)
                    outputs, _ = net(inputs.float().cuda())

                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu().numpy().item()
                    ws_1.write(1+sum_sign_val, i+1, predicted)
                    if i == 0:
                        ws_1.write(1+sum_sign_val, 91, labels.numpy().item())
                    # total += labels.size(0)
                    # if predicted == labels.item():
                    #     correct += 1
                # result_list.append((100. * correct.float() / total))
            del inputs, labels, outputs, correct, predicted
            del total
            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad(): 
                correct = 0.0
                total = 0.0
                net.eval()
                sign_feature = 0
                for data in testloader:
                    sum_sign_test += 1
                    inputs, labels = data
                    inputs = inputs.unsqueeze(1)
                    outputs, features = net(inputs.float().cuda())
                    if sign_feature == 0:
                        FT = features.detach().cpu().numpy()
                    if sign_feature == 1:
                        FT = np.concatenate((FT, features.detach().cpu().numpy()), axis=0)

                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu().numpy().item()
                    ws_2.write(1+sum_sign_test, i+1, predicted)
                    # total += labels.size(0)
                    # if predicted == labels.item():
                    #     correct += 1.0
                # result_list.append((100. * correct.float() / total))
                    sign_feature = 1
                scio.savemat(args.datapath+'2019_6_13/' + 'test_' + str(fold)  + '_'  + PD_FA[i], {'data':FT})
            del inputs, labels, outputs, correct, predicted
            del net, total
            torch.cuda.empty_cache()
            gc.collect()

            del testset, testloader, valset, valloader
            torch.cuda.empty_cache()
            gc.collect()

    wb.save(args.datapath+'2019_6_13/'+'FA_test.xlsx')

main()

