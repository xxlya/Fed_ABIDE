'''
Interpreting fed and single model
refer feature_select.py for saliency features summary
'''
import random
import torch
from torch import nn
from torch.nn import functional as F
import deepdish as dd
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from networks import MLP
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #the gpu used to train models


class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        relu =  nn.ReLU()
        return relu(layer.grad).detach().cpu().numpy()


def main(args):
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    models = []
    if args.method == 'fed':
        for i in range (5):
            model = MLP(6105,16,2).to(device)
            model.load_state_dict(torch.load(os.path.join('./model/fed_overlap',str(i)+'.pth')))
            models.append(model)
    elif args.method == 'single':
        for i in range(5):
            model = MLP(6105,8,2).to(device)
            model.load_state_dict(torch.load(os.path.join('./model/single_overlap', args.site, str(i) + '.pth')))
            models.append(model)
    elif args.method == 'mix':
        for i in range (5):
            model = MLP(6105,16,2).to(device)
            model.load_state_dict(torch.load(os.path.join('./model/mix_overlap',str(i)+'.pth')))
            models.append(model)


    data1 = dd.io.load(os.path.join(args.vec_dir,'NYU_correlation_matrix.h5'))
    data2 = dd.io.load(os.path.join(args.vec_dir,'UM_correlation_matrix.h5'))
    data3 = dd.io.load(os.path.join(args.vec_dir,'USM_correlation_matrix.h5'))
    data4 = dd.io.load(os.path.join(args.vec_dir,'UCLA_correlation_matrix.h5'))

    x1 = torch.from_numpy(data1['data']).float()
    y1 = torch.from_numpy(data1['label']).long()
    x2 = torch.from_numpy(data2['data']).float()
    y2 = torch.from_numpy(data2['label']).long()
    x3 = torch.from_numpy(data3['data']).float()
    y3 = torch.from_numpy(data3['label']).long()
    x4 = torch.from_numpy(data4['data']).float()
    y4 = torch.from_numpy(data4['label']).long()

    idNYU = dd.io.load('./idx/NYU_sub_overlap.h5')
    idUM = dd.io.load('./idx/UM_sub_overlap.h5')
    idUSM = dd.io.load('./idx/USM_sub_overlap.h5')
    idUCLA = dd.io.load('./idx/UCLA_sub_overlap.h5')

    grad = dict()
    for i in range(5):
        if i == 0:
            tr1 = idNYU['1'] + idNYU['2'] + idNYU['3'] + idNYU['4']
            tr2 = idUM['1'] + idUM['2'] + idUM['3'] + idUM['4']
            tr3 = idUSM['1'] + idUSM['2'] + idUSM['3'] + idUSM['4']
            tr4 = idUCLA['1'] + idUCLA['2'] + idUCLA['3'] + idUCLA['4']
            te1 = idNYU['0']
            te2 = idUM['0']
            te3 = idUSM['0']
            te4 = idUCLA['0']
        elif i == 1:
            tr1 = idNYU['0'] + idNYU['2'] + idNYU['3'] + idNYU['4']
            tr2 = idUM['0'] + idUM['2'] + idUM['3'] + idUM['4']
            tr3 = idUSM['0'] + idUSM['2'] + idUSM['3'] + idUSM['4']
            tr4 = idUCLA['0'] + idUCLA['2'] + idUCLA['3'] + idUCLA['4']
            te1 = idNYU['1']
            te2 = idUM['1']
            te3 = idUSM['1']
            te4 = idUCLA['1']
        elif i == 2:
            tr1 = idNYU['0'] + idNYU['1'] + idNYU['3'] + idNYU['4']
            tr2 = idUM['0'] + idUM['1'] + idUM['3'] + idUM['4']
            tr3 = idUSM['0'] + idUSM['1'] + idUSM['3'] + idUSM['4']
            tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['3'] + idUCLA['4']
            te1 = idNYU['2']
            te2 = idUM['2']
            te3 = idUSM['2']
            te4 = idUCLA['2']
        elif i == 3:
            tr1 = idNYU['0'] + idNYU['1'] + idNYU['2'] + idNYU['4']
            tr2 = idUM['0'] + idUM['1'] + idUM['2'] + idUM['4']
            tr3 = idUSM['0'] + idUSM['1'] + idUSM['2'] + idUSM['4']
            tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['2'] + idUCLA['4']
            te1 = idNYU['3']
            te2 = idUM['3']
            te3 = idUSM['3']
            te4 = idUCLA['3']
        elif i == 4:
            tr1 = idNYU['0'] + idNYU['1'] + idNYU['2'] + idNYU['3']
            tr2 = idUM['0'] + idUM['1'] + idUM['2'] + idUM['3']
            tr3 = idUSM['0'] + idUSM['1'] + idUSM['2'] + idUSM['3']
            tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['2'] + idUCLA['3']
            te1 = idNYU['4']
            te2 = idUM['4']
            te3 = idUSM['4']
            te4 = idUCLA['4']

        x1_train = x1[tr1]
        x2_train = x2[tr2]
        x3_train = x3[tr3]
        x4_train = x4[tr4]

        x1_test = x1[te1]
        y1_test = y1[te1]
        x2_test = x2[te2]
        y2_test = y2[te2]
        x3_test = x3[te3]
        y3_test = y3[te3]
        x4_test = x4[te4]
        y4_test = y4[te4]

        mean = x1_train.mean(0, keepdim=True)
        dev = x1_train.std(0, keepdim=True)
        x1_test = (x1_test - mean) / dev


        mean = x2_train.mean(0, keepdim=True)
        dev = x2_train.std(0, keepdim=True)
        x2_test = (x2_test - mean) / dev


        mean = x3_train.mean(0, keepdim=True)
        dev = x3_train.std(0, keepdim=True)
        x3_test = (x3_test - mean) / dev


        mean = x4_train.mean(0, keepdim=True)
        dev = x4_train.std(0, keepdim=True)
        x4_test = (x4_test - mean) / dev


        if args.ASD:
            x1_test = x1_test[y1_test == 1]
            y1_test = y1_test[y1_test == 1]
            x2_test = x2_test[y2_test == 1]
            y2_test = y2_test[y2_test == 1]
            x3_test = x3_test[y3_test == 1]
            y3_test = y3_test[y3_test == 1]
            x4_test = x4_test[y4_test == 1]
            y4_test = y4_test[y4_test == 1]
        elif args.HC:
            x1_test = x1_test[y1_test == 0]
            y1_test = y1_test[y1_test == 0]
            x2_test = x2_test[y2_test == 0]
            y2_test = y2_test[y2_test == 0]
            x3_test = x3_test[y3_test == 0]
            y3_test = y3_test[y3_test == 0]
            x4_test = x4_test[y4_test == 0]
            y4_test = y4_test[y4_test == 0]
        else:
            x1_test = x1_test
            y1_test = y1_test
            x2_test = x2_test
            y2_test = y2_test
            x3_test = x3_test
            y3_test = y3_test
            x4_test = x4_test
            y4_test = y4_test




        test1 = TensorDataset(x1_test, y1_test)
        test2 = TensorDataset(x2_test, y2_test)
        test3 = TensorDataset(x3_test, y3_test)
        test4 = TensorDataset(x4_test, y4_test)
        test_loader1 = DataLoader(test1, batch_size=1, shuffle=False)
        test_loader2 = DataLoader(test2, batch_size=1, shuffle=False)
        test_loader3 = DataLoader(test3, batch_size=1, shuffle=False)
        test_loader4 = DataLoader(test4, batch_size=1, shuffle=False)

        if args.site == 'NYU':
            test_loader = test_loader1
        elif args.site == 'UM':
            test_loader = test_loader2
        elif args.site == 'USM':
            test_loader = test_loader3
        elif args.site == 'UCLA':
            test_loader = test_loader4


        grad[i] = list()
        gdbp = GuidedBackPropogation(models[i])
        models[i].eval()
        for data, target in test_loader:
            data = data.to(device)
            data = data.requires_grad_()
            out_b = gdbp(data)
            out_b[:, target.item()].backward()
            grad_b = gdbp.get(data)
            grad[i].append(grad_b)

    if args.ASD:
        dd.io.save(os.path.join('./interpretation',args.method+ '_' +args.site +'.h5'),{'grad':grad})
    elif args.HC:
        dd.io.save(os.path.join('./interpretation_hc', args.method + '_' + args.site + '.h5'), {'grad': grad})
    else:
        dd.io.save(os.path.join('./interpretation_2class', args.method + '_' + args.site + '.h5'), {'grad': grad})




#=======================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--method', type=str, default='mix', help='[single, fed, mix]')
    parser.add_argument('--site', type=str, default='UCLA', help='used for single model')
    parser.add_argument('--ASD', type=bool, default=False, help='test ASD only')
    parser.add_argument('--HC', type=bool, default=False, help='test HC only')
    parser.add_argument('--vec_dir', type=str, default='./data/HO_vector_overlap')


    args = parser.parse_args()
    main(args)