import torch
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
from tqdm import tqdm
import time
import model_defination
import load_data
import configuration
import json
import os
import pandas as pd


class ModelTrain:
    def __init__(self, para_dict) -> None:
        self.save_location = para_dict['save_location']
        self.lr = para_dict['lr']
        self.num_epochs = para_dict['num_epochs']
        self.batch_size = para_dict['batch_size']
        self.drop_prob = para_dict['drop_prob']
        self.selected_remarks = para_dict['selected_remarks']
        self.net = model_defination.Net(drop_prob=self.drop_prob)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else'cpu')
        self.cmedata = load_data.CMEdata(
            self.save_location, self.selected_remarks)

    def __create_folder(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def __get_dataloader(self):
        self.cmedata.load_data_from_npz()
        train_dataset = self.cmedata.to_tensordataset(is_train=True)
        test_dataset = self.cmedata.to_tensordataset(is_train=False)
        train_iter = torch.utils.data.DataLoader(
            train_dataset, self.batch_size, shuffle=True)
        test_iter = torch.utils.data.DataLoader(
            test_dataset, self.batch_size, shuffle=True)
        return train_iter, test_iter

    def __save_epoch_info(self, train_info_path):
        df = pd.DataFrame(self.train_details_list)
        filename = os.path.join(train_info_path, 'epoch_info.xlsx')
        df.to_excel(filename)

    def __save_para_info(self, train_info_path):
        pos = self.cmedata.train_label.sum().item()
        neg = self.cmedata.size-pos
        para_dict['CME_count'] = pos
        para_dict['No_CME_count'] = neg
        para_dict['CME:NO CME'] = '{}:1'.format(pos/neg)
        filename = os.path.join(train_info_path, 'para.json')
        with open(filename, 'w') as f:
            json.dump(para_dict, f)

    def save_info(self):
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        train_info_path = os.path.join(os.getcwd(), 'train_info', current_time)
        self.__create_folder(train_info_path)
        self.__save_epoch_info(train_info_path)
        self.__save_para_info(train_info_path)
        torch.save(self.net.state_dict(), os.path.join(
            train_info_path, 'parameters.pkl'))
        print('Save training detail infomation to {}'.format(train_info_path))

    def evaluate_accuracy_on(self, X, y):
        num_accu, total = 0, 0
        X, y = X.to(self.device), y.to(self.device)
        num_accu += (torch.argmax(self.net(X), dim=1) == y).sum().item()
        total += X.shape[0]
        return num_accu/total

    def __evaluate_accuracy_on_testiter(self, test_iter):
        num_accu, total = 0, 0
        for X, y in test_iter:
            X, y = X.to(self.device), y.to(self.device)
            self.net.eval()  # 修改为评估模式
            num_accu += (torch.argmax(self.net(X), dim=1) == y).sum().item()
            self.net.train()
            total += X.shape[0]
        return num_accu/total

    def fit(self):
        train_iter, test_iter = self.__get_dataloader()
        self.net = self.net.to(self.device)
        self.net.train()
        print('training on', self.device)
        batch_count = 0
        # 计算总的iteration数量
        total_iterations = self.num_epochs * \
            (int(self.cmedata.train_size/self.batch_size)+1)
        print('begin training')
        print('{} epoches {} iterations each epoch'.format(
            self.num_epochs, int(self.cmedata.train_size/self.batch_size)))
        pbar = tqdm(total=total_iterations)
        train_start = time.time()
        # 记录每次epoch情况的列表
        self.train_details_list = []
        for epoch in range(self.num_epochs):
            # 分别表示每epoch训练总损失，每epoch训练正确数，每epoch样本总数
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            current_epoch_start = time.time()
            iteration_count = 0
            for X, y in train_iter:
                iteration_count += 1
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.net(X)
                l = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
                pbar.set_description(
                    'epoch {} iteration {}'.format(epoch+1, iteration_count))
                pbar.update(1)
            test_accu = self.__evaluate_accuracy_on_testiter(test_iter)
            epoch_detail_dict = {'epoch_num': epoch,
                                 'epoch_train_loss': train_l_sum/batch_count,
                                 'epoch_train_accu': train_acc_sum/n,
                                 'epoch_test_accu': test_accu,
                                 'epoch_time': time.time()-current_epoch_start,
                                 'total_time': time.time()-train_start}
            self.train_details_list.append(epoch_detail_dict)
            epoch_detail_dict.pop('epoch_num')
            pbar.set_postfix(epoch_detail_dict)
        pbar.close()


def save_para_info(para_dict, cmedata, path_to_folder):
    pos = cmedata.train_label.sum().item()
    neg = cmedata.size-pos
    para_dict['CME_count'] = pos
    para_dict['No_CME_count'] = neg
    para_dict['CME:NO CME'] = '{}:1'.format(pos/neg)
    filename = os.path.join(path_to_folder, 'para.json')
    with open(filename, 'w') as f:
        json.dump(para_dict, f)


def create_folder(path_to_folder):
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


if __name__ == '__main__':
    para_dict = configuration.para_dict
    modeltrain = ModelTrain(para_dict)
    modeltrain.fit()
    modeltrain.save_info()
