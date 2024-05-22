import torch
import torch.nn as nn
import torch.utils.data as tud
import math
from data import ThermalDataset, data_convertor_batch
class RegModel(nn.Module):
    def __init__(self):
        super(RegModel,self).__init__()
        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Linear(6, 16),
            nn.PReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
            nn.PReLU(),
        )
        pass


    def forward(self, x):
        return self.net(x)

class AttNet(nn.Module):
    def __init__(self, dim):
        super(AttNet, self).__init__()
        self.dim = dim
        self.build_network()

    def build_network(self):
        self.feat_init1 = nn.AdaptiveAvgPool1d(self.dim)
        self.feat_att1 = nn.Sequential(
            nn.Linear(self.dim,4),
            nn.PReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4,self.dim),
            nn.Sigmoid()
        )

        self.src_init1 = nn.AdaptiveAvgPool1d(1)
        self.src_att1 = nn.Sequential(
            nn.Linear(1, 2),
            nn.PReLU(),
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = x.unsqueeze(1)


        feat_init1 = self.feat_init1(x).squeeze(1)
        feat_att1 = self.feat_att1(feat_init1)


        #print(feat_att1.size(), feat_att2.size())

        src_init1 = self.src_init1(x).squeeze(1)
        src_att1 = self.src_att1(src_init1)


        #print(src_att1.size(), src_att2.size())
        ret_x = x*feat_att1*src_att1

        return ret_x

class ClsModel(nn.Module):
    def __init__(self):
        super(ClsModel,self).__init__()
        self.build_net()

    def build_net(self):
        self.b1 = nn.Sequential(
            nn.Linear(5, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.b2 = nn.Sequential(
            nn.Linear(5, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.b3 = nn.Sequential(
            nn.Linear(6, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.b4 = nn.Sequential(
            nn.Linear(5, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.b5 = nn.Sequential(
            nn.Linear(28, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.b6 = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.PReLU()
        )
        self.bd = {
            0:self.b1, 1:self.b2, 2:self.b3, 3:self.b4,4:self.b5, 5:self.b6
        }
        self.net1 = nn.Sequential(
            nn.Linear(48, 16),
            nn.PReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
            nn.PReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(48, 16),
            nn.PReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
            nn.PReLU(),
        )
        self.cls_att1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        pass


    def forward(self, x):
        x_list = []
        sz = x.size()[1]
        for idx in range(sz):

            ipt_ = data_convertor_batch(idx, x[:,idx])
            #print(idx, ipt_.size())
            x_list.append( self.bd[idx](ipt_))
            #print(idx, x_list[-1].shape)

        conx = torch.cat(x_list, dim=1)#x
        ret1, ret2= self.net1(conx), self.net2(conx)
        ret = self.net1(conx)#torch.cat([ret1, ret2], dim=1)

        #print(conx.size())
        #exit()
        return ret



def train():
    #model = RegModel()
    model = ClsModel()
    train_data = ThermalDataset('train.txt')
    test_data = ThermalDataset('test.txt')
    train_loader = tud.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = tud.DataLoader(dataset=test_data, batch_size=32)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(),  lr=1e-2)
    opt_sch = torch.optim.lr_scheduler.StepLR(opt, 80, 0.9)
    for epoch in range(5000):
        model.train()
        for idx, (input, output) in enumerate(train_loader):

            pred_ = model(input)
            loss_ = criterion(pred_, output)
            opt.zero_grad()
            loss_.backward()
            opt.step()


            #if idx % 20 ==0:
            #   print("epoch {} iter {} loss {}".format(epoch, idx, loss_.item()))
        opt_sch.step()
        model.eval()
        for idx, (input, output) in enumerate(test_loader):
            err = math.sqrt(nn.functional.mse_loss(model(input),output))
            #print(opt.state_dict()['param_groups'][0]['lr'])
            print('=>test: epoch {}, error {}'.format(epoch,err))


    pass

if __name__ == '__main__':
    train()


