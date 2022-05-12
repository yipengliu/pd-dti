import numpy as np
# import torch
import xlrd
from xlwt import *
import argparse
import copy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PD_detection')
parser.add_argument('--fold', type=int, default=1,
                    help='choose the fold 1-5')
args = parser.parse_args()

fold = str(args.fold)
print('The current fold is: ', fold)
results_path = '/home/hengling/my/DTI/CGU/5-fold test/FA/'
data = xlrd.open_workbook(results_path+fold+'/FA_test.xlsx')
max_results = data.sheets()[0]  
results = data.sheets()[1]
pre_results = data.sheets()[2]
# weights = max_results.col_values(11)[0:90]
results_names = results.row_values(0)
pre_names = pre_results.row_values(0) 

#--------sort the region by AUC----------
# y_data = results.col_values(91)[1:458]
fpr_array = []
tpr_array = []
auc_array = []
name_list = []
# for region_num in range(90):
#     # print('the number of rigions is: ', region_num)
#     # region_num = j
#     region_name = max_results.col_values(0)[0:90]
#     region = region_name[region_num]
#     x_data = results.col_values(results_names.index(region))[1:] 

#     fpr,tpr,threshold = roc_curve(y_data, x_data)
#     roc_auc = auc(fpr,tpr)
#     name_list.append(region)
#     auc_array.append(roc_auc)
    
# region_sorted = []
# auc_sorted = []
# for i in range(len(name_list)):
#     max_value = max(auc_array)
#     index_max = auc_array.index(max_value)
#     max_region = name_list[index_max]
#     region_sorted.append(max_region)
#     auc_sorted.append(max_value)
#     auc_array.pop(index_max)
#     name_list.pop(index_max)

region_sorted = max_results.col_values(0)[0:90]
auc_sorted = max_results.col_values(11)[0:90]
region_reverse = region_sorted[::-1]
#--------print region's name and its weights---------
for i in range(len(region_sorted)):
    print(region_sorted[i], ': ', auc_sorted[i])

#--------calculate the performance of combined model with fixed number of regions---------
# regions: the set including region's name
# mode = 1: test mode; mode = 0: val mode. 
# save = 1: save the roc fig.
# specific = print the specific results (accuracy, precision, recall)
def calculate_combine_results(regions, mode=0, save=0, specific=0):
    # results_names = results.row_values(0)
    # rank_names = max_results.col_values(0)
    rank_names = region_sorted
    weights = auc_sorted
    if mode == 1:
        y_data = np.zeros(100, int)
        y_data[43:] = 1
        x_data = np.zeros((1000, len(regions)))
        test_results = np.zeros((100, len(regions)))
        region_name = regions
        temp_weights = 0
        for i in range(len(regions)):
            # region_name = max_results.col_values(0)
            x_data[:, i] = pre_results.col_values(
                pre_names.index(region_name[i]))[2:]
            for m in range(100):
                temp = 0
                for n in range(10):
                    temp += x_data[m + n * 100, i]
                if len(regions) != 1:
                    if temp/10.0 >= 0.5:
                        test_results[m, i] = 1 * weights[
                            rank_names.index(regions[i])]
                    else:
                        test_results[m, i] = 0
                else:
                    test_results[m, i] = temp/10 * weights[
                        rank_names.index(regions[i])]
            temp_weights = weights[rank_names.index(
                regions[i])] + temp_weights
        x_data = test_results
        x_data = x_data / (temp_weights / len(regions))
    else:
        y_data = results.col_values(91)[2:434]
        x_data = np.zeros((432, len(regions)))
        temp_weights = 0
        for i in range(len(regions)):
            temp_data = np.array(results.col_values(
                results_names.index(regions[i]))[2:])
            x_data[:, i] = temp_data
            temp_weights = weights[rank_names.index(
                regions[i])] + temp_weights
        x = np.zeros_like(x_data)
        y = np.ones_like(x_data)

        #-------if the number of region is 1, don't binarize the results------
        if len(regions) != 1:
            x_data = np.where(x_data>=0.5, y, x)

    if len(regions) != 1:
        y_pre = np.squeeze(np.sum(x_data, axis=1) / len(regions))
        # print(y_pre.shape)
    else:
        y_pre = np.squeeze(x_data)

    if specific == 1:
        x = np.zeros_like(y_pre)
        y = np.ones_like(y_pre)
    # for i in range(80, 90):
        y_temp = np.where(y_pre>=0.585, y, x)
        print('accuracy:', accuracy_score(y_data, y_temp))
        target_names = ['NL', 'PD']
        print(classification_report(y_data, y_temp, target_names=target_names))
    fpr,tpr,threshold = roc_curve(y_data, y_pre) 

    #--------print the specific informations among different thresholds ---------
    # if save == 1:
    #     print('threshold', threshold)
    #     print('False Positive Rat', fpr)
    #     print('True Positive Rate', tpr)

    roc_auc = auc(fpr,tpr)
    if save == 1:
        plt.plot(fpr, tpr, lw=1.5, label=' (AUC = %0.3f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if mode==0:
            savefig(results_path+fold+'/ROC_single/roc_val_' + str(len(regions)) + '.jpg')
        else:
            savefig(results_path+fold+'/ROC_single/roc_test_' + str(len(regions)) + '.jpg')
    plt.close()

    return roc_auc


if __name__ == "__main__":
    error_list = []

    file = Workbook(encoding='utf-8')
    table = file.add_sheet('data')
    table.write(0, 0, 'region_num')
    table.write(0, 1, 'val_auc')
    table.write(0, 2, 'test_auc')
    # table.write(0, 3, 'specificity')
    for region_num in range(1, 91):
        print('the number of rigions is: ', region_num)
        initial_regions = region_sorted[:region_num]
        # initial_regions = region_reverse[:region_num]
        sign = 1
        initial_region_num = []

        choosed_set = initial_regions
        unchoosed_set = copy.deepcopy(region_sorted)
        for chooset_lenth in range(len(choosed_set)):
            unchoosed_set.remove(choosed_set[chooset_lenth])
        
        while sign >= 1:
            if len(choosed_set) == 1:
                roc_auc = calculate_combine_results(choosed_set)
                roc_auc_pre = calculate_combine_results(choosed_set, mode=1, save=0, specific=0)
                # print('auc:', roc_auc)
                # print('auc_pre:', roc_auc_pre)
                break

            for i in range(len(choosed_set)):
                temp_choosed_set = copy.deepcopy(choosed_set)
                temp_choosed_set.pop(i)
                # error, _, _ = calculate_combine_results(temp_choosed_set)
                roc_auc = calculate_combine_results(temp_choosed_set)
                if i > 0:
                    if highest_roc_auc < roc_auc:
                        lowest_error = copy.deepcopy(roc_auc)
                        remove_name = choosed_set[i]
                        choosed_temp_set = copy.deepcopy(temp_choosed_set)
                elif i == 0:
                    highest_roc_auc = copy.deepcopy(roc_auc)
                    remove_name = choosed_set[i]
                    choosed_temp_set = copy.deepcopy(temp_choosed_set)
            choosed_set = choosed_temp_set
            # error, _, _ = calculate_combine_results(choosed_set)
            # print('after_rem_error', error)
            unchoosed_set = copy.deepcopy(region_sorted)
            for chooset_lenth in range(len(choosed_set)):
                unchoosed_set.remove(choosed_set[chooset_lenth])
            for i in range(len(unchoosed_set)):
                temp_choosed_set = copy.deepcopy(choosed_set)
                temp_choosed_set.append(unchoosed_set[i])
                # error, _, _ = calculate_combine_results(temp_choosed_set)
                roc_auc = calculate_combine_results(temp_choosed_set)
                if i > 0:
                    if highest_roc_auc < roc_auc:
                        lowest_error = copy.deepcopy(roc_auc)
                        add_name = unchoosed_set[i]
                        choosed_temp_set = copy.deepcopy(temp_choosed_set)
                elif i == 0:
                    highest_roc_auc = copy.deepcopy(roc_auc)
                    add_name = unchoosed_set[i]
                    choosed_temp_set = copy.deepcopy(temp_choosed_set)
            choosed_set = choosed_temp_set
            if remove_name == add_name:
                # print('-----------')
                # print('validation')
                # print('-----------')
                roc_auc = calculate_combine_results(choosed_set, save=0, specific=1)
                # print('-----------')
                # print('test')
                # print('-----------')
                roc_auc_pre = calculate_combine_results(choosed_set, mode=1, save=0, specific=1)

                print('auc:', roc_auc)
                print('auc_pre:', roc_auc_pre)
                #--------print choosed region name---------
                # for i in range(len(choosed_set)):
                #     print(choosed_set[i])
                break
            else:
                sign += 1
        table.write(region_num, 0, region_num)
        table.write(region_num, 1, roc_auc)
        table.write(region_num, 2, roc_auc_pre)
        # table.write(region_num-1, 0, region_num)
        # for i in range(len(choosed_set)):
        #     table.write(region_num, i+1, choosed_set[i])

        error_list.append(roc_auc)
            
    # file.save(results_path+fold+'/name.xlsx')
    file.save(results_path + fold + '/data.xlsx')
    max_num = max(error_list)
    max_region_num = error_list.index(max_num)
    print('The best result is:', max_num)
    print('The best region number is:', max_region_num + 1)
