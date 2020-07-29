'''
Ensemble using averaging
'''
import random
import torch
import deepdish as dd
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from networks import MLP
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #the gpu used to train models

def main(args):
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    sites = ['NYU','UCLA','UM','USM']
    models_cross = []
    for file in sites:
        if file != args.site:
            model = MLP(6105,8,2).to(device)
            model.load_state_dict(torch.load(os.path.join('./model/cross_overlap',file+'.pth')))
            models_cross.append(model)

    model_single = MLP(6105,8,2).to(device)
    model_single.load_state_dict(torch.load(os.path.join('./model/single_overlap', args.site, str(args.split) + '.pth')))


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

    if args.split == 0:
        tr1 = idNYU['1'] + idNYU['2'] + idNYU['3'] + idNYU['4']
        tr2 = idUM['1'] + idUM['2'] + idUM['3'] + idUM['4']
        tr3 = idUSM['1'] + idUSM['2'] + idUSM['3'] + idUSM['4']
        tr4 = idUCLA['1'] + idUCLA['2'] + idUCLA['3'] + idUCLA['4']
        te1 = idNYU['0']
        te2 = idUM['0']
        te3 = idUSM['0']
        te4 = idUCLA['0']
    elif args.split == 1:
        tr1 = idNYU['0'] + idNYU['2'] + idNYU['3'] + idNYU['4']
        tr2 = idUM['0'] + idUM['2'] + idUM['3'] + idUM['4']
        tr3 = idUSM['0'] + idUSM['2'] + idUSM['3'] + idUSM['4']
        tr4 = idUCLA['0'] + idUCLA['2'] + idUCLA['3'] + idUCLA['4']
        te1 = idNYU['1']
        te2 = idUM['1']
        te3 = idUSM['1']
        te4 = idUCLA['1']
    elif args.split == 2:
        tr1 = idNYU['0'] + idNYU['1'] + idNYU['3'] + idNYU['4']
        tr2 = idUM['0'] + idUM['1'] + idUM['3'] + idUM['4']
        tr3 = idUSM['0'] + idUSM['1'] + idUSM['3'] + idUSM['4']
        tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['3'] + idUCLA['4']
        te1 = idNYU['2']
        te2 = idUM['2']
        te3 = idUSM['2']
        te4 = idUCLA['2']
    elif args.split == 3:
        tr1 = idNYU['0'] + idNYU['1'] + idNYU['2'] + idNYU['4']
        tr2 = idUM['0'] + idUM['1'] + idUM['2'] + idUM['4']
        tr3 = idUSM['0'] + idUSM['1'] + idUSM['2'] + idUSM['4']
        tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['2'] + idUCLA['4']
        te1 = idNYU['3']
        te2 = idUM['3']
        te3 = idUSM['3']
        te4 = idUCLA['3']
    elif args.split == 4:
        tr1 = idNYU['0'] + idNYU['1'] + idNYU['2'] + idNYU['3']
        tr2 = idUM['0'] + idUM['1'] + idUM['2'] + idUM['3']
        tr3 = idUSM['0'] + idUSM['1'] + idUSM['2'] + idUSM['3']
        tr4 = idUCLA['0'] + idUCLA['1'] + idUCLA['2'] + idUCLA['3']
        te1 = idNYU['4']
        te2 = idUM['4']
        te3 = idUSM['4']
        te4 = idUCLA['4']

    x1_train = x1[tr1]
    y1_train = y1[tr1]
    x2_train = x2[tr2]
    y2_train = y2[tr2]
    x3_train = x3[tr3]
    y3_train = y3[tr3]
    x4_train = x4[tr4]
    y4_train = y4[tr4]

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


    test1 = TensorDataset(x1_test, y1_test)
    test2 = TensorDataset(x2_test, y2_test)
    test3 = TensorDataset(x3_test, y3_test)
    test4 = TensorDataset(x4_test, y4_test)


    if args.site == 'NYU':
        test = test1
    elif args.site == 'UM':
        test = test2
    elif args.site == 'USM':
        test = test3
    elif args.site == 'UCLA':
        test = test4

    te_data = test.tensors[0].to(device)
    te_outputs = []
    targets = test.tensors[1].numpy()
    preds =[]
    #cross model
    for model in models_cross:
        model.eval()
        te_output = model(te_data)
        te_outputs.append(torch.exp(te_output))


    # single_model
    model_single.eval()
    te_output = model_single(te_data)
    te_outputs.append(torch.exp(te_output))
    outputtorch = torch.stack(te_outputs,dim=0)
    output_mean = torch.mean(outputtorch,dim=0)
    preds = output_mean.data.max(1)[1].detach().cpu().numpy()
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    dd.io.save(os.path.join(args.res_dir, args.site+ '_' + str(args.split) + '.h5'),
                {'preds': preds, 'targets': targets})




        #=======================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--split', type=int, default=0, help='select 0-4 fold')
    parser.add_argument('--method', type=str, default='fed', help='[single, fed]')
    parser.add_argument('--site', type=str, default='NYU', help='used for single model')
    parser.add_argument('--res_dir', type=str, default='./result/ensemble_overlap')


    args = parser.parse_args()
    main(args)