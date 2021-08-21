import torch
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
from tqdm import tqdm
import time
import model_defination
import load_data


class ModelTrain:
    def __init__(self, net, optimizer, num_epochs, loss,  batch_size, device) -> None:
        self.net = net
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate_accuracy_on(self, X, y):
        num_accu, total = 0, 0
        X, y = X.to(device), y.to(device)
        num_accu += (torch.argmax(self.net(X), dim=1) == y).sum().item()
        total += X.shape[0]
        return num_accu/total

    def __evaluate_accuracy_on_testiter(self, test_iter):
        num_accu, total = 0, 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            self.net.eval()  # 修改为评估模式
            num_accu += (torch.argmax(self.net(X), dim=1) == y).sum().item()
            self.net.train()
            total += X.shape[0]
        return num_accu/total

    def fit(self,  train_iter, test_iter, train_data_len):
        self.net = self.net.to(self.device)
        self.net.train()
        print('training on ', self.device)
        batch_count = 0
        total_iterations = self.num_epochs * \
            (int(train_data_len/self.batch_size)+1)
        print('begin training\n')
        print('{} epoches {} iterations each epoch\n'.format(
            self.num_epochs, int(train_data_len/self.batch_size)))
        pbar = tqdm(total=total_iterations)
        train_start = time.time()
        for epoch in range(self.num_epochs):
            # 分别表示每epoch训练总损失，每epoch训练正确数，每epoch样本总数
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            current_epoch_start = time.time()
            iteration_count = 0
            for X, y in train_iter:
                iteration_count += 1
                X = X.to(device)
                y = y.to(device)
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
            pbar.set_postfix(train_loss=train_l_sum/batch_count,
                             train_accu_count=train_acc_sum/n,
                             test_accu=test_accu,
                             epoch_time=time.time()-current_epoch_start,
                             total_time=time.time()-train_start)
        pbar.close()


print('start')
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
save_location = r'D:\Programming\CME_data'
selected_remarks = ['Halo', 'No Remark', 'Partial Halo']
train_percentage = 0.7
batch_size = 50
lr = 0.001
num_epochs = 5
net = model_defination.Net(2, 0.5)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
cmedata = load_data.CMEdata(
    save_location, selected_remarks, train_percentage)
cmedata.load_data_from_npz()
train_dataset = cmedata.to_tensordataset(is_train=True)
test_dataset = cmedata.to_tensordataset(is_train=False)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(
    test_dataset, batch_size, shuffle=True)
modeltrain = ModelTrain(net, optimizer, num_epochs,
                        loss, batch_size, device)
modeltrain.fit(train_iter, test_iter, cmedata.train_size)
