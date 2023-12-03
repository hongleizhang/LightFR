# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as la
from collections import defaultdict
from math import log
import pandas as pd
import torch

from DataLoader import DataLoaderCenter
from Metrics import Metrics


class Base:

    def __init__(self):
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.u_i_r = defaultdict(dict)
        self.i_u_r = defaultdict(dict)
        self.minVal = 0.5
        self.maxVal = 4

        self.dataset_name = 'filmtrust'
        self.federated_train_data_path = 'data/' + self.dataset_name + '/' + self.dataset_name + '_train.csv'
        self.federated_valid_data_path = 'data/' + self.dataset_name + '/' + self.dataset_name + '_val.csv'
        self.federated_test_data_path = 'data/' + self.dataset_name + '/' + self.dataset_name + '_test.csv'
        pass


    def init_model(self):
        self.generate_vocabulary()
        self.rating_matrix, self.rating_matrix_bin, self.globalmean = self.get_rating_matrix()
        self.B = np.sign(np.array(np.random.randn(len(self.user), self.configs.code_len) / (self.configs.code_len ** 0.5)))
        self.D = np.sign(np.array(np.random.randn(len(self.item), self.configs.code_len) / (self.configs.code_len ** 0.5)))
        self.loss, self.last_delta_loss = 0.0, 0.0

    def trainSet(self):
        with open(self.federated_train_data_path, 'r') as f:
            for index, line in enumerate(f):
                if index != 0:  # 去除headers
                    u, i, r = line.strip('\r\n').split(',')
                    r = 2 * self.configs.code_len * (float(r)) - self.configs.code_len
                    yield (int(u), int(i), float(r))

    def containUser(self, user_id):
        if user_id in self.user:
            return True
        else:
            return False

    def containItem(self, item_id):
        if item_id in self.item:
            return True
        else:
            return False

    def valid_test_Set(self, path):
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                if index != 0:  # 去除headers
                    u, i, r = line.strip('\r\n').split(',')
                    # r = 2 * self.code_len * (float(int(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01) - self.code_len
                    yield (int(u), int(i), float(r))

    def read_federated_valid_dataset(self, path):
        data_val = pd.read_csv(path)
        return data_val

    def generate_vocabulary(self):
        for index, line in enumerate(self.trainSet()):
            user_id, item_id, rating = line
            self.u_i_r[user_id][item_id] = rating
            self.i_u_r[item_id][user_id] = rating
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id

        for index, line in enumerate(self.valid_test_Set(self.federated_valid_data_path)):
            user_id, item_id, rating = line
            self.u_i_r[user_id][item_id] = rating
            self.i_u_r[item_id][user_id] = rating
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id

        for index, line in enumerate(self.valid_test_Set(self.federated_test_data_path)):
            user_id, item_id, rating = line
            self.u_i_r[user_id][item_id] = rating
            self.i_u_r[item_id][user_id] = rating
            if user_id not in self.user:
                self.user[user_id] = len(self.user)
                self.id2user[self.user[user_id]] = user_id
            if item_id not in self.item:
                self.item[item_id] = len(self.item)
                self.id2item[self.item[item_id]] = item_id


    def get_rating_matrix(self):
        rating_matrix = np.zeros((len(self.user), len(self.item)))  # (943, 1596)
        globalmean = 0.0
        lens = 0
        for index, line in enumerate(self.trainSet()):
            lens += 1
            user_id, item_id, rating = line
            globalmean += rating
            rating_matrix[self.user[user_id]][self.item[item_id]] = int(rating)
        rating_matrix_bin = (rating_matrix > 0).astype('int')
        globalmean = globalmean / (lens)
        return rating_matrix, rating_matrix_bin, globalmean

    def K(self, x, y):
        return x if x != 0 else y

    def valid_test_model(self, path):
        pre_true_dict = defaultdict(list)
        for index, line in enumerate(self.valid_test_Set(path)):
            user_id, item_id, rating = line
            if (self.containUser(user_id) and self.containItem(item_id)):
                bu = self.B[self.user[user_id], :]
                di = self.D[self.item[item_id], :]
                pre = np.dot(bu, di)
            elif (self.containUser(user_id) and not self.containItem(item_id)):
                pre = sum(self.u_i_r[user_id].values()) / float(len(self.u_i_r[user_id]))
            elif (not self.containUser(user_id) and self.containItem(item_id)):
                pre = sum(self.i_u_r[item_id].values()) / float(len(self.i_u_r[item_id]))
            else:
                pre = self.globalmean
            pre_true_dict[user_id].append([pre, rating])
        metrics = Metrics()
        ndcg_10 = metrics.calDCG_k(pre_true_dict, 10)
        return ndcg_10