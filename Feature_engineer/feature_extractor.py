#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Created by Yulun @ Aug 4, 2016
import pandas as pd
import time
import math
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import Binarizer

from operator import itemgetter
import pickle
import sys
import os

"""
Some global parameters to be tuned here.
"""
date_range = ("2016-01-01T00:00:00", "2016-03-31T23:59:59")

time_start = int(time.mktime(time.strptime(date_range[0], '%Y-%m-%dT%H:%M:%S')))
time_end = int(time.mktime(time.strptime(date_range[1], '%Y-%m-%dT%H:%M:%S')))

# TODO: def foo()
# 7 days as a period
period = 604800.0
n_period = int(math.ceil((time_end - time_start)/(period)))



def save_obj(obj, name):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def time_str2int(in_time):
    return int(time.mktime(time.strptime(in_time, '%Y-%m-%d %H:%M:%S'))) 

def string2list(in_str):
    ret = in_str.split('],[')
    temp = []
    for each in ret:
        temp.append(each.strip('[').strip(']').split(','))
    return temp
    #return pd.DataFrame(temp)

def get_uuids(filename):
    # Get logs within the time range
    t = pd.read_csv(filename, sep='\t', header = None)
    t1 = t[t[2] >= time_start]
    data_all = t1[t1[2] <= time_end]

    grouped_by_uuid = data_all.groupby([0])
    uuids = grouped_by_uuid.groups.keys()

    return uuids

# def get_uids(userbase_dataframe, uuid_list):
#     # TODO
#     u_by_uuid = userbase_dataframe[userbase_dataframe[1].isin(set(uuid_list))]
#     u_by_uuid = u_by_uuid[u_by_uuid[1]!= 'cfcd208495d565ef66e7dff9f98764da']
#     return u_by_uuid[0].unique()

def get_eventids(filename):
    # Get logs within the time range
    t = pd.read_csv(filename, sep='\t', header = None)
    t1 = t[t[2] >= time_start]
    data_all = t1[t1[2] <= time_end]

    grouped_by_eventid = data_all.groupby([1])
    eventids = grouped_by_eventid.groups.keys()
    return eventids

def get_uuid2uid(userbase_dataframe, uuid_list):
    # TODO
    u_by_uuid = userbase_dataframe[userbase_dataframe[1].isin(set(uuid_list))]
    u_by_uuid = u_by_uuid[u_by_uuid[1]!= 'cfcd208495d565ef66e7dff9f98764da']
    uuid2uid = {}
    for each in u_by_uuid.iterrows():
        if each[1][1] not in uuid2uid:
            uuid2uid[each[1][1]] = []
        uuid2uid[each[1][1]].append(each[1][0])
    return uuid2uid

def get_uid2uuid(userbase_dataframe, uuid_list):
    u_by_uuid = userbase_dataframe[userbase_dataframe[1].isin(set(uuid_list))]
    u_by_uuid = u_by_uuid[u_by_uuid[1]!= 'cfcd208495d565ef66e7dff9f98764da']
    uid2uuid = {}
    for each in u_by_uuid.iterrows():
        if each[1][0] not in uid2uuid:
            uid2uuid[each[1][0]] = []
        uid2uuid[each[1][0]].append(each[1][1])
    return uid2uuid

def feature_expenditure(filename, userbase_dataframe, uuid_list):

    uid2uuid = get_uid2uuid(userbase_dataframe, uuid_list)
    for each in uid2uuid:
        if len(uid2uuid[each]) > 1:
            print "Found one-to-many uid to uuid relationship, but I ignored it for simplicity."

    e = pd.read_csv(filename, sep='\t', header = None)
    exp = e[e[0].isin(set(uid2uuid.keys()))]
    exp[-1] = exp[0].map(lambda x: uid2uuid[x][0])
    exp_by_uuid = exp.groupby([-1])

    expenditure_dict = {}
    for each in exp_by_uuid.groups:
        index_list = exp_by_uuid.groups[each]
        data_frame = exp.loc[index_list]

        reg = []
        rec = []
        pay = []
        for row in data_frame.iterrows():
            reg += string2list(row[1][1])
            rec += string2list(row[1][2])
            pay += string2list(row[1][3])
        reg = pd.DataFrame(reg)
        rec = pd.DataFrame(rec)
        pay = pd.DataFrame(pay)


        """reg"""
        reg = reg[reg[3]!='0']
        try: # no reg means no rec or pay
            #[0] #uid
            temp = [len(index_list)]
            #[1] #games
            temp.append(len(reg[0]))
            #[2] # unique games
            temp.append(len(reg[1].unique()))
            #[3] # unique cid
            temp.append(len(reg[2].unique()))
            #[4] # unique game_uid
            temp.append(len(reg[4].unique()))
            #[5] # unique ucid
            temp.append(len(reg[5].unique()))

            reg_time = reg[3].map(time_str2int)
            #[6] max reg time
            temp.append(max(reg_time))
            #[7] min reg time
            temp.append(min(reg_time))
            #[8] reg time span
            temp.append(max(reg_time) - min(reg_time))
            #[9] reg time mean
            temp.append(reg_time.mean())
            #[10] reg time mid
            temp.append(reg_time.quantile(0.5))
            #[11] reg time std
            temp.append(np.std(reg_time))
        except:
            continue

        """rec"""
        rec = rec[rec[2]!='0']
        if len(rec) == 0:
            temp += [np.nan] * 12
        else:
            #[12] # rec
            temp.append(len(rec[0]))
            #[13] # unique games
            temp.append(len(rec[0].unique()))
            #[14] # unique types
            temp.append(len(rec[1].unique()))
            #[15] # unique cid
            temp.append(len(rec[3].unique()))
            #[16] # unique item
            temp.append(len(rec[4].unique()))    
            #[17] # unique appid
            temp.append(len(rec[5].unique()))  
            #[18] # unique pay_from
            temp.append(len(rec[6].unique()))  

            rec_time = rec[2].map(time_str2int)
            #[19] max reg time
            temp.append(max(rec_time))
            #[20] min reg time
            temp.append(min(rec_time))
            #[21] reg time span
            temp.append(max(rec_time) - min(rec_time))
            #[22] reg time mean
            temp.append(rec_time.mean())
            #[23] reg time mid
            temp.append(rec_time.quantile(0.5))
            #[24] reg time std
            temp.append(np.std(rec_time))

        pay = pay[pay[2]!='0']
        if len(pay) == 0:
            temp += [np.nan] * 11
        else:
            #[25] # payments
            temp.append(len(pay[0]))
            #[26] # unique games
            temp.append(len(pay[0].unique()))
            #[27] # unique cid
            temp.append(len(pay[1].unique()))
            #[28] # unique appid
            temp.append(len(pay[3].unique()))
            #[29] # unique item
            temp.append(len(pay[4].unique()))    
            #[30] # unique porder
            temp.append(len(pay[5].unique()))  


            pay_time = pay[2].map(time_str2int)
            #[31] max reg time
            temp.append(max(pay_time))
            #[32] min reg time
            temp.append(min(pay_time))
            #[33] reg time span
            temp.append(max(pay_time) - min(pay_time))
            #[34] reg time mean
            temp.append(pay_time.mean())
            #[35] reg time mid
            temp.append(pay_time.quantile(0.5))
            #[36] reg time std
            temp.append(np.std(pay_time))

        expenditure_dict[each] = temp

    return expenditure_dict  


def cleaning_filter(input_element, filter_list, replacement):
    if input_element not in filter_list:
        return replacement
    return input_element

def ucid_cleaner(input_element):
    try:
        ret = eval(input_element)
    except:
        ret = []
    if type(ret) is list:
        return ret
    elif type(ret) is dict:
        return map(lambda x: int(x), ret.values())
    elif type(ret) is int:
        return [ret]
    else:
        print "Error in ucid_cleaner"

def get_userbase(dir_name):
    file_count = sum([file.startswith("user_base_") for file in os.listdir(dir_name)])
    if file_count < 1:
        print "Error: Cannot find user_base_i.txt file in the directory. Please check your directory name."
        return None

    u = pd.read_csv(dir_name + 'user_base_0.txt', sep='\t', header = None)
    for i in range(1, file_count):
        u = u.append(pd.read_csv(dir_name + 'user_base_' + str(i) + '.txt', sep='\t', header = None))

    return u

def feature_userbase(userbase_dataframe, uuid_list):
    """
    Input: userbase_dataframe-- stacked userbases by pandas.read_csv()
           uuid_list-- -- uuids to be extracted from userbase

    Output: a dictionary of (uuid, feature) pairs
    """

    u_by_uuid = userbase_dataframe[userbase_dataframe[1].isin(set(uuid_list))]
    # remove empty uuids: 'cfcd208495d565ef66e7dff9f98764da' 
    u_by_uuid = u_by_uuid[u_by_uuid[1] != 'cfcd208495d565ef66e7dff9f98764da']
    # remove NaNs
    u_by_uuid.drop(u_by_uuid[u_by_uuid[9].isnull()].index, inplace=True)

    """Some data cleaning"""
    # clean sex
    value_counts = u_by_uuid[5].value_counts()
    u_by_uuid[5] = u_by_uuid[5].map(lambda x: cleaning_filter(x, value_counts.index[:4], value_counts.index[0]))

    # clean platform
    u_by_uuid[6] = u_by_uuid[6].map(lambda x: str(x))
    value_counts = u_by_uuid[6].value_counts()
    u_by_uuid[6] = u_by_uuid[6].map(lambda x: cleaning_filter(x, value_counts.index[:3], '0'))

    # clean status
    u_by_uuid[7] = u_by_uuid[7].map(lambda x: str(x))
    u_by_uuid[7] = u_by_uuid[7].map(lambda x: cleaning_filter(x, ['0'], '1'))

    # clean ucid
    u_by_uuid[8] = u_by_uuid[8].map(lambda x: str(x))

    """some processing about ucid"""
    # clean ucid
    u_by_uuid[10] = u_by_uuid[10].map(ucid_cleaner)
    ucids = []
    for each in u_by_uuid[10]:
        ucids += each
    dictinct_ucids = list(set(ucids))

    # Begin feature engineering
    grouped_by_u = u_by_uuid.groupby([1])
    
    userbase_dict = {}
    for each in grouped_by_u.groups:
        
        index_list = grouped_by_u.groups[each]
        data_frame = u_by_uuid.loc[index_list]
        
        try:
            #[0] uid
            temp = [len(index_list)]

            #[1] reg_ip
            temp.append(data_frame[2].nunique())

            #[2] has signature or not
            temp.append(int(sum(data_frame[3].isnull()) > 0))


            #[3] has nickname or not
            #nn = data_frame['4'].map(lambda x: x == 'None')
            temp.append(int(sum(data_frame[4].map(lambda x: x == 'None')) > 0))

            #[4] sex majority -- One-Hot?
            temp.append(str(data_frame[5].value_counts().index[0]))
            #[5] sex unique count
            temp.append(data_frame[5].nunique())

            #[6] platform majority -- One-Hot ?
            temp.append(str(data_frame[6].value_counts().index[0]))
            #[7] platform unique count
            temp.append(data_frame[6].nunique())

            #[8] ucid majority -- One-hot ?
            temp.append(str(data_frame[8].value_counts().index[0]))
            #[9] ucid unique count
            temp.append(data_frame[8].nunique())

            #reg time stuff
            reg_time = data_frame[9].map(time_str2int)
            #[10] reg time max
            temp.append(max(reg_time))
            #[11] reg time min
            temp.append(min(reg_time))
            #[12] reg time span
            temp.append(max(reg_time) - min(reg_time))
            #[13] reg time mean
            temp.append(reg_time.mean())
            #[14] reg time mid
            temp.append(reg_time.quantile(0.5))
            #[15] reg time std
            temp.append(np.std(reg_time))


            #[16] group: number of groups
            temp.append(sum(data_frame[10].map(lambda x: len(x))) )
            #[17] group: dummy
            temp.append(int(temp[len(temp)-1] > 0))

            """TODO: a huge feature line of ucid down in the bottom"""

            #[18] name
            temp.append(int(sum(data_frame[11].map(lambda x: x != 'None')) > 0))

            #[19-25] dummy
            for i in range(12, 19):
                temp.append(int(sum(data_frame[i]) > 0))

            """the huge line promised above"""
            df_ucids = []
            for ucid_list in data_frame[10]:
                df_ucids += ucid_list
            ucid_feature = [0] * len(dictinct_ucids)
            for i, e in enumerate(dictinct_ucids):
                if e in df_ucids:
                    ucid_feature[i] += 1

            temp += ucid_feature

            userbase_dict[each] = temp

        except:
            e = sys.exc_info()[0]
            print data_frame
            print e
            break

    return userbase_dict


def feature_event(filename):

    # Get logs within the time range
    t = pd.read_csv(filename, sep='\t', header = None)
    t1 = t[t[2] >= time_start]
    data_all = t1[t1[2] <= time_end]

    # binning whole logs in periods as a big dict()
    whole_dict = {i : {} for i in range(n_period)}
    # for each log
    for log in data_all.iterrows():
        # find its binned period
        idx = int(math.floor((log[1][2] - time_start)/period))

        if log[1][0] not in whole_dict[idx]:
            whole_dict[idx][log[1][0]] = []

        # append uuid, event id, and timestamp to the list
        whole_dict[idx][log[1][0]].append((log[1][1], log[1][2]))

    return whole_dict

def get_labels(whole_dict):

    # creating labels for churn = 1, stay = 0
    labels = {i : {} for i in range(n_period)}
    #for each period
    for i in range(n_period - 1):
        # for each uuid
        for uuid in whole_dict[i]:
            # check if the uuid appears in the next period, if yes-> stay; no-> churn
            if uuid in whole_dict[i + 1]:
                labels[i][uuid] = 0
            else:
                labels[i][uuid] = 1

    return labels

def feature_final(whole_dict, user_dict, exp_dict, labels, eventids):
    """
    This function takes feature dicts of event, userbase, expenditure, blends them, 
        and returns the total feature as a pandas dataframe
    """
    """
    labels = get_labels(whole_dict)
    eventids = get_eventids('timeline/timeline_event_gpapp.txt')
    """
    features = []
    num_handcraft_feature = 105
    for i in range(n_period - 2):
        for uuid in whole_dict[i]:
            e_len = len(eventids)

            row = np.zeros(e_len + num_handcraft_feature)

            # label
            row[0] = labels[i][uuid]

            times = []
            times_diff = []
            for each in whole_dict[i][uuid]:
                eid = each[0]
                ts = each[1]
                idx = eventids.index(eid)
                # [0] - [len(eventid)] event feature
                row[idx + 1] += 1

                times.append(ts)
                times_diff.append(ts - (time_start + period*(i)))

            # add handcraft features here
            """time features"""
            # time
            row[e_len + 1] = max(times)
            row[e_len + 2] = min(times)
            row[e_len + 3] = max(times) - min(times)
            row[e_len + 4] = np.mean(times)
            row[e_len + 5] = np.percentile(times, 50)
            row[e_len + 6] = np.std(times)
            # time_diff
            row[e_len + 7] = max(times_diff)
            row[e_len + 8] = min(times_diff)
            row[e_len + 9] = np.mean(times_diff)
            row[e_len + 10] = np.percentile(times_diff, 50)

            # userbase integrated time features
            try:
                row[e_len + 11] = max(times) - user_dict[uuid][10]
                row[e_len + 12] = max(times) - user_dict[uuid][11]
                row[e_len + 13] = max(times) - user_dict[uuid][13]
                row[e_len + 14] = max(times) - user_dict[uuid][14]
                row[e_len + 15] = min(times) - user_dict[uuid][10]
                row[e_len + 16] = min(times) - user_dict[uuid][11]
                row[e_len + 17] = min(times) - user_dict[uuid][13]
                row[e_len + 18] = min(times) - user_dict[uuid][14]
                row[e_len + 19] = np.percentile(times, 50) - user_dict[uuid][10]
                row[e_len + 20] = np.percentile(times, 50) - user_dict[uuid][11]
                row[e_len + 21] = np.percentile(times, 50) - user_dict[uuid][13]
                row[e_len + 22] = np.percentile(times, 50) - user_dict[uuid][14]
                row[e_len + 23] = np.mean(times) - user_dict[uuid][10]
                row[e_len + 24] = np.mean(times) - user_dict[uuid][11]
                row[e_len + 25] = np.mean(times) - user_dict[uuid][13]
                row[e_len + 26] = np.mean(times) - user_dict[uuid][14]
                row[e_len + 27] = (time_start + period*(i)) - user_dict[uuid][10]
                row[e_len + 28] = (time_start + period*(i)) - user_dict[uuid][11]
                row[e_len + 29] = (time_start + period*(i)) - user_dict[uuid][13]
                row[e_len + 30] = (time_start + period*(i)) - user_dict[uuid][14]
            except:
                row[e_len + 11 : e_len + 31] = np.nan

            # expenditure integrated time features
            try:
                row[e_len + 31] = max(times) - exp_dict[uuid][6]
                row[e_len + 32] = max(times) - exp_dict[uuid][7]
                row[e_len + 33] = max(times) - exp_dict[uuid][9]
                row[e_len + 34] = max(times) - exp_dict[uuid][10]
                row[e_len + 35] = min(times) - exp_dict[uuid][6]
                row[e_len + 36] = min(times) - exp_dict[uuid][7]
                row[e_len + 37] = min(times) - exp_dict[uuid][9]
                row[e_len + 38] = min(times) - exp_dict[uuid][10]
                row[e_len + 39] = np.percentile(times, 50) - exp_dict[uuid][6]
                row[e_len + 40] = np.percentile(times, 50) - exp_dict[uuid][7]
                row[e_len + 41] = np.percentile(times, 50) - exp_dict[uuid][9]
                row[e_len + 42] = np.percentile(times, 50) - exp_dict[uuid][10]
                row[e_len + 43] = np.mean(times) - exp_dict[uuid][6]
                row[e_len + 44] = np.mean(times) - exp_dict[uuid][7]
                row[e_len + 45] = np.mean(times) - exp_dict[uuid][9]
                row[e_len + 46] = np.mean(times) - exp_dict[uuid][10]
                row[e_len + 47] = (time_start + period*(i)) - exp_dict[uuid][6]
                row[e_len + 48] = (time_start + period*(i)) - exp_dict[uuid][7]
                row[e_len + 49] = (time_start + period*(i)) - exp_dict[uuid][9]
                row[e_len + 50] = (time_start + period*(i)) - exp_dict[uuid][10]
            except:
                row[e_len + 31 : e_len + 51] = np.nan

            try:
                row[e_len + 51] = max(times) - exp_dict[uuid][19]
                row[e_len + 52] = max(times) - exp_dict[uuid][20]
                row[e_len + 53] = max(times) - exp_dict[uuid][22]
                row[e_len + 54] = max(times) - exp_dict[uuid][23]
                row[e_len + 55] = min(times) - exp_dict[uuid][19]
                row[e_len + 56] = min(times) - exp_dict[uuid][20]
                row[e_len + 57] = min(times) - exp_dict[uuid][22]
                row[e_len + 58] = min(times) - exp_dict[uuid][23]
                row[e_len + 59] = np.percentile(times, 50) - exp_dict[uuid][19]
                row[e_len + 60] = np.percentile(times, 50) - exp_dict[uuid][20]
                row[e_len + 51] = np.percentile(times, 50) - exp_dict[uuid][22]
                row[e_len + 62] = np.percentile(times, 50) - exp_dict[uuid][23]
                row[e_len + 63] = np.mean(times) - exp_dict[uuid][19]
                row[e_len + 64] = np.mean(times) - exp_dict[uuid][20]
                row[e_len + 65] = np.mean(times) - exp_dict[uuid][22]
                row[e_len + 66] = np.mean(times) - exp_dict[uuid][23]
                row[e_len + 67] = (time_start + period*(i)) - exp_dict[uuid][19]
                row[e_len + 68] = (time_start + period*(i)) - exp_dict[uuid][20]
                row[e_len + 69] = (time_start + period*(i)) - exp_dict[uuid][22]
                row[e_len + 70] = (time_start + period*(i)) - exp_dict[uuid][23]
            except:
                row[e_len + 51 : e_len + 71] = np.nan

            try:
                row[e_len + 71] = max(times) - exp_dict[uuid][31]
                row[e_len + 72] = max(times) - exp_dict[uuid][32]
                row[e_len + 73] = max(times) - exp_dict[uuid][34]
                row[e_len + 74] = max(times) - exp_dict[uuid][35]
                row[e_len + 75] = min(times) - exp_dict[uuid][31]
                row[e_len + 76] = min(times) - exp_dict[uuid][32]
                row[e_len + 77] = min(times) - exp_dict[uuid][34]
                row[e_len + 78] = min(times) - exp_dict[uuid][35]
                row[e_len + 79] = np.percentile(times, 50) - exp_dict[uuid][31]
                row[e_len + 80] = np.percentile(times, 50) - exp_dict[uuid][32]
                row[e_len + 81] = np.percentile(times, 50) - exp_dict[uuid][34]
                row[e_len + 82] = np.percentile(times, 50) - exp_dict[uuid][35]
                row[e_len + 83] = np.mean(times) - exp_dict[uuid][31]
                row[e_len + 84] = np.mean(times) - exp_dict[uuid][32]
                row[e_len + 85] = np.mean(times) - exp_dict[uuid][34]
                row[e_len + 86] = np.mean(times) - exp_dict[uuid][35]
                row[e_len + 87] = (time_start + period*(i)) - exp_dict[uuid][31]
                row[e_len + 88] = (time_start + period*(i)) - exp_dict[uuid][32]
                row[e_len + 89] = (time_start + period*(i)) - exp_dict[uuid][34]
                row[e_len + 90] = (time_start + period*(i)) - exp_dict[uuid][35]
            except:
                row[e_len + 71 : e_len + 91] = np.nan

            """Event features"""
            # [57+] #events
            row[e_len + 91] = len(whole_dict[i][uuid])
            # break into 2 bins
            row[e_len + 92] = len(filter(lambda x: x[1] >= (time_start + period*(i)) and x[1] < (time_start + period*(i+0.5)), whole_dict[i][uuid]))
            row[e_len + 93] = len(filter(lambda x: x[1] >= (time_start + period*(i+0.5)) and x[1] < (time_start + period*(i+1)), whole_dict[i][uuid]))
            # break into 4 bins
            row[e_len + 94] = len(filter(lambda x: x[1] >= (time_start + period*(i)) and x[1] < (time_start + period*(i+0.25)), whole_dict[i][uuid]))
            row[e_len + 95] = len(filter(lambda x: x[1] >= (time_start + period*(i+0.25)) and x[1] < (time_start + period*(i+0.5)), whole_dict[i][uuid]))
            row[e_len + 96] = len(filter(lambda x: x[1] >= (time_start + period*(i+0.5)) and x[1] < (time_start + period*(i+0.75)), whole_dict[i][uuid]))
            row[e_len + 97] = len(filter(lambda x: x[1] >= (time_start + period*(i+0.75)) and x[1] < (time_start + period*(i+1)), whole_dict[i][uuid]))
            # break into 7 bins
            row[e_len + 98] = len(filter(lambda x: x[1] >= (time_start + period*(i)) and x[1] < (time_start + period*(i+1/7.0)), whole_dict[i][uuid]))
            row[e_len + 99] = len(filter(lambda x: x[1] >= (time_start + period*(i+1/7.0)) and x[1] < (time_start + period*(i+2/7.0)), whole_dict[i][uuid]))
            row[e_len + 100] = len(filter(lambda x: x[1] >= (time_start + period*(i+2/7.0)) and x[1] < (time_start + period*(i+3/7.0)), whole_dict[i][uuid]))
            row[e_len + 101] = len(filter(lambda x: x[1] >= (time_start + period*(i+3/7.0)) and x[1] < (time_start + period*(i+4/7.0)), whole_dict[i][uuid]))
            row[e_len + 102] = len(filter(lambda x: x[1] >= (time_start + period*(i+4/7.0)) and x[1] < (time_start + period*(i+5/7.0)), whole_dict[i][uuid]))
            row[e_len + 103] = len(filter(lambda x: x[1] >= (time_start + period*(i+5/7.0)) and x[1] < (time_start + period*(i+6/7.0)), whole_dict[i][uuid]))
            row[e_len + 104] = len(filter(lambda x: x[1] >= (time_start + period*(i+6/7.0)) and x[1] < (time_start + period*(i+1)), whole_dict[i][uuid]))

            """features.append(row)"""
            features.append((uuid, row))

    ret = []
    for each in features:
        temp = each[1].tolist()
        if each[0] in user_dict:
            temp += user_dict[each[0]]
        else:
            temp += ([np.nan]*145)
        if each[0] in exp_dict:
            temp += exp_dict[each[0]]
        else:
            temp += ([np.nan]*37)
        ret.append(temp)

    ret = pd.DataFrame(ret)

    # some preliminary processing
    ret[308] = ret[308].map(lambda x: ['None', '0', '1', '2'].index(x) if x in ['None', '0', '1', '2'] else 0)
    ret[310] = ret[310].map(lambda x: [ '0', '102', '101'].index(x) if x in [ '0', '102', '101'] else 1)

    ret[range(486,685)] = pd.DataFrame(Binarizer().fit_transform(ret[range(1,200)]))
    ret[range(685,699)] = pd.DataFrame(Binarizer().fit_transform(ret[range(290,304)]))
    ret[699] = pd.DataFrame(ret[448].notnull().map(lambda x: int(x)))
    ret[700] = pd.DataFrame(ret[479].notnull().map(lambda x: int(x)))
    ret[701] = pd.DataFrame(ret[202].map(lambda x: int(x <= 86400)))
    ret[702] = pd.DataFrame(ret[205].map(lambda x: int(x <= 86400)))
    ret[703] = ret[308].map(lambda x : int(x == 0))
    ret[704] = ret[308].map(lambda x : int(x == 1))
    ret[705] = ret[308].map(lambda x : int(x == 2))
    ret[706] = ret[308].map(lambda x : int(x == 3))
    ret[707] = ret[310].map(lambda x : int(x == 0))
    ret[708] = ret[310].map(lambda x : int(x == 1))
    ret[709] = ret[310].map(lambda x : int(x == 2))

    return ret

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print "Model with rank: {0}".format(i + 1)
        print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores))
        print "Parameters: {0}".format(score.parameters)


def eval_result():
    # Evaluation
    print "Below are the accuracy, AUC, F1 score, precision, and recall: "
    print clf.score(X_test, y_test)
    print roc_auc_score(y_test, y_pred)
    print f1_score(y_test, y_pred)
    print precision_score(y_test, y_pred)
    print recall_score(y_test, y_pred)

    print "Confusion Matrix: "
    cm = confusion_matrix(y_test, y_pred)
    print cm
    #false positive
    print 'false positive: ' + str(cm[0][1])
    #false negative
    print 'false negative: ' + str(cm[1][0])


if __name__ == '__main__':
    
    main()