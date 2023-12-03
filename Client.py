
import numpy as np

from Metrics import Metrics


class Client:

    def __init__(self, configs):
        self.bu = None
        self.D = None
        self.data_u = None
        self.data_bin_u = None
        self.data_len_u = None
        self.configs = configs


    def client_update(self, client, master_flag):
        '''
        client process, could be implemented in parallel
        :param master_flag:
        :param bu:
        :param D:
        :param data_u:
        :param data_bin_u:
        :param l:
        :return:
        '''


        while True:
            flag = 0
            for k in range(self.configs.code_len):
                dk = client.D[:, k]
                buk_hat = np.sum(
                    ( client.data_u - np.dot(client.D, client.bu.T)) * dk * client.data_bin_u) + 2 * self.configs.lambdaa * client.data_len_u * client.bu[k]
                buk_new = np.sign(self.K(buk_hat, client.bu[k]))
                if (client.bu[k] != buk_new):
                    flag = 1
                    client.bu[k] = buk_new
            if (flag == 0):
                break
            master_flag = 1

        return client.bu, master_flag

    def get_inter_params(self, i, k):
        di = self.D[i, :]
        grads = (self.data_u[i] - np.dot(self.bu, di.T)) * self.bu[k] * self.data_bin_u[i]
        grads_len = self.data_bin_u[i]
        return grads, grads_len

    def K(self, x, y):
        return x if x != 0 else y

    def calculate_loss(self):
        local_loss = np.sum((self.data_u - np.dot(self.D, self.bu)) ** 2 * self.data_bin_u)
        return local_loss

    def evaluate_local(self, items, val_data):
        configs = {'top_k': 10, 'num_negative_test': 49, }
        metric = Metrics(configs)
        bus = self.bu
        dis = self.D[items]
        rating_pred = np.multiply(bus, dis)
        preds = np.sum(rating_pred, axis=1)
        val_data['pred'] = preds.tolist()

        hr = metric.get_hit_ratio(val_data)
        ndcg = metric.get_ndcg(val_data)
        return hr, ndcg