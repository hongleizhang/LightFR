import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as TDataLoader


class DataLoader():
    def __init__(self, configs, client_data):
        self.configs = configs
        self.train_data, self.val_data, self.test_data = client_data['train'], client_data['val'], client_data[
            'test']

    def get_train_dataloader(self):
        users, items, labels = torch.LongTensor(np.array(self.train_data['user_id'])), torch.LongTensor(
            np.array(self.train_data['item_id'])), torch.FloatTensor(np.array(self.train_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        return TDataLoader(dataset, batch_size=self.configs['local_batch_size'], shuffle=True)

    def get_val_dataloader(self):

        if self.val_data.empty:
            users, items, labels = torch.LongTensor(self.val_data['user_id']), torch.LongTensor(
                self.val_data['item_id']), torch.FloatTensor(self.val_data['ratings'])
        else:
            users, items, labels = torch.LongTensor(np.array(self.val_data['user_id'])), torch.LongTensor(
                np.array(self.val_data['item_id'])), torch.FloatTensor(np.array(self.val_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        client_data_len = len(items)  # 100 for implicit feedback, actual length for explicit feedback during validation in each local client

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)

    def get_test_dataloader(self):

        if self.test_data.empty:
            users, items, labels = torch.LongTensor(self.test_data['user_id']), torch.LongTensor(
                self.test_data['item_id']), torch.FloatTensor(self.test_data['ratings'])
        else:
            users, items, labels = torch.LongTensor(np.array(self.test_data['user_id'])), torch.LongTensor(
                np.array(self.test_data['item_id'])), torch.FloatTensor(np.array(self.test_data['ratings']))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        client_data_len = len(items)

        return TDataLoader(dataset, batch_size=client_data_len, shuffle=False)


class DataLoaderCenter():
    def __init__(self, configs, val_data):
        self.configs = configs
        self.val_data= val_data

    def get_train_dataloader(self):
        users, items, labels = torch.LongTensor(np.array(self.train_data['user_id'], dtype='int32')), torch.LongTensor(
            np.array(self.train_data['item_id'], dtype='int32')), torch.FloatTensor(
            np.array(self.train_data['ratings'], dtype='float32'))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        return TDataLoader(dataset, batch_size=self.configs['local_batch_size'], shuffle=True)

    def get_val_dataloader(self):

        if self.val_data.empty:
            users, items, labels = torch.LongTensor(self.val_data['user_id']), torch.LongTensor(
                self.val_data['item_id']), torch.FloatTensor(self.val_data['ratings'])
        else:
            users, items, labels = torch.LongTensor(np.array(self.val_data['user_id'], dtype='int32')), torch.LongTensor(
                np.array(self.val_data['item_id'], dtype='int32')), torch.FloatTensor(np.array(self.val_data['ratings'], dtype='float32'))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        data_len = self.configs['num_negative_test'] + 1

        return TDataLoader(dataset, batch_size=data_len, shuffle=False)

    def get_test_dataloader(self):

        if self.test_data.empty:
            users, items, labels = torch.LongTensor(self.test_data['user_id']), torch.LongTensor(
                self.test_data['item_id']), torch.FloatTensor(self.test_data['ratings'])
        else:
            users, items, labels = torch.LongTensor(np.array(self.test_data['user_id'], dtype='int32')), torch.LongTensor(
                np.array(self.test_data['item_id'], dtype='int32')), torch.FloatTensor(np.array(self.test_data['ratings'], dtype='float32'))

        dataset = UserItemRatingDataset(user_tensor=users, item_tensor=items, target_tensor=labels)

        data_len = self.configs['num_negative_test'] + 1

        return TDataLoader(dataset, batch_size=data_len, shuffle=False)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


if __name__ == '__main__':
    configs = {
        'dataset': 'ml-1m',
        'data_type': 'explicit',
        'num_negative_train': 4,
        'num_negative_test': 49,
        'local_batch_size': 100,
        'cold_nums': 10
    }
    dr = DataReader(configs)
    # client_data = dr.get_data_by_client(0)
    data = dr.get_train_val_test_data()
    dl_center = DataLoaderCenter(configs, data)
    td = dl_center.get_train_dataloader()
    vd = dl_center.get_val_dataloader()
    for index, data in enumerate(vd):
        if index == 0:
            print(data)
