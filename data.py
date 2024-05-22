import torch
import torch.utils.data as tud
import random
import numpy as np
random.seed(777)
col1 = [4., 0.,  -7., -12., -20.]
col2 = [0., 2., 5., 9., 12.]
col3 = [0.19, 0.212, 0.23, 0.246, 0.26, 0.32]
col4 = [1.4798, 1.5579, 1.6160, 1.6420, 1.6487]
col5 = [0.3826,0.3830,0.3831,0.3838,0.3883,0.3903,0.3930,0.3939,0.3967,0.4005,0.4072,
        0.4073,0.4074,0.4076,0.4077,0.4079,0.4081,0.4082,0.4084,0.4086,0.4095,0.4098,
        0.4114,0.4206,0.4215,0.4227,0.4244,0.4265 ]
col6_min, col6_max = 0.3167, 0.7859


def data_convertor(idx, input_data):
    idx_dict = {
        0: col1,  1: col2,  2: col3,   3: col4,   4: col5
    }
    input_data = float(input_data)
    #print(idx, input_data)
    if idx < 5:
        d_ = idx_dict[idx]
        ret = [0]*len(d_)
        ret[d_.index(input_data)]=1
        return ret
    elif idx == 5:
        return [(input_data-col6_min)/(col6_max-col6_min)]
    else:
        return [input_data]

def data_convertor_batch(idx, input_data):
    input_data = input_data.numpy()
    ret = [data_convertor(idx,d) for d in input_data]
    return torch.FloatTensor(ret)
class ThermalDataset(tud.Dataset):

    def __init__(self, filepath):
        self.load_data(filepath)

        pass

    def load_data(self, filepath):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as fr:
            #a = [ fr.readline() for _ in range(3)]
            for line in fr:
                segs = [float(s) for s in line.strip().split('\t')]
                #segs = [data_convertor(idx, d) for idx, d in enumerate(segs)]

                self.data.append(segs)
        self.data = np.array(self.data)#torch.FloatTensor(self.data)
        #print(self.data.size())
        pass

    def __getitem__(self, index):
        data_ = self.data[index]

        input = data_[:6]
        output = data_[-2:]
        return (input, torch.FloatTensor(output))

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    with open('data.txt', 'r', encoding='utf-8') as fr:
        lines = [line for line in fr]

    a = random.shuffle(lines)

    trainf =  open('train.txt','w+', encoding='utf-8')
    testf =  open('test.txt','w+', encoding='utf-8')

    for idx, line in enumerate(lines):
        if idx > 130:
            testf.write(line)
        else:
            trainf.write(line)

    trainf.close()
    testf.close()
    exit()

    td = ThermalDataset()
    dl = tud.DataLoader(dataset=td, batch_size=4,shuffle=True)

    for idx, (input, output) in enumerate(dl):
        print(idx, (input, output))
        exit()