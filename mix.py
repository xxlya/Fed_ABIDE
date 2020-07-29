import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import deepdish as dd
from networks import MLP
import os
import argparse
import copy


EPS = 1e-15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    torch.manual_seed(args.seed)
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

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

    if args.overlap:
        idNYU = dd.io.load('./idx/NYU_sub_overlap.h5')
        idUM = dd.io.load('./idx/UM_sub_overlap.h5')
        idUSM = dd.io.load('./idx/USM_sub_overlap.h5')
        idUCLA = dd.io.load('./idx/UCLA_sub_overlap.h5')
    else:
        idNYU = dd.io.load('./idx/NYU_sub.h5')
        idUM = dd.io.load('./idx/UM_sub.h5')
        idUSM = dd.io.load('./idx/USM_sub.h5')
        idUCLA = dd.io.load('./idx/UCLA_sub.h5')

    if args.split==0:
        tr1 = idNYU['1']+idNYU['2']+idNYU['3']+idNYU['4']
        tr2 = idUM['1']+idUM['2']+idUM['3']+idUM['4']
        tr3 = idUSM['1']+idUSM['2']+idUSM['3']+idUSM['4']
        tr4 = idUCLA['1']+idUCLA['2']+idUCLA['3']+idUCLA['4']
        te1=  idNYU['0']
        te2 = idUM['0']
        te3=  idUSM['0']
        te4 = idUCLA['0']
    elif args.split==1:
        tr1 = idNYU['0']+idNYU['2']+idNYU['3']+idNYU['4']
        tr2 = idUM['0']+idUM['2']+idUM['3']+idUM['4']
        tr3 = idUSM['0']+idUSM['2']+idUSM['3']+idUSM['4']
        tr4 = idUCLA['0']+idUCLA['2']+idUCLA['3']+idUCLA['4']
        te1=  idNYU['1']
        te2 = idUM['1']
        te3=  idUSM['1']
        te4 = idUCLA['1']
    elif args.split==2:
        tr1 = idNYU['0']+idNYU['1']+idNYU['3']+idNYU['4']
        tr2 = idUM['0']+idUM['1']+idUM['3']+idUM['4']
        tr3 = idUSM['0']+idUSM['1']+idUSM['3']+idUSM['4']
        tr4 = idUCLA['0']+idUCLA['1']+idUCLA['3']+idUCLA['4']
        te1=  idNYU['2']
        te2 = idUM['2']
        te3=  idUSM['2']
        te4 = idUCLA['2']
    elif args.split==3:
        tr1 = idNYU['0']+idNYU['1']+idNYU['2']+idNYU['4']
        tr2 = idUM['0']+idUM['1']+idUM['2']+idUM['4']
        tr3 = idUSM['0']+idUSM['1']+idUSM['2']+idUSM['4']
        tr4 = idUCLA['0']+idUCLA['1']+idUCLA['2']+idUCLA['4']
        te1=  idNYU['3']
        te2 = idUM['3']
        te3=  idUSM['3']
        te4 = idUCLA['3']
    elif args.split==4:
        tr1 = idNYU['0']+idNYU['1']+idNYU['2']+idNYU['3']
        tr2 = idUM['0']+idUM['1']+idUM['2']+idUM['3']
        tr3 = idUSM['0']+idUSM['1']+idUSM['2']+idUSM['3']
        tr4 = idUCLA['0']+idUCLA['1']+idUCLA['2']+idUCLA['3']
        te1=  idNYU['4']
        te2 = idUM['4']
        te3=  idUSM['4']
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

    if args.sepnorm:
        mean = x1_train.mean(0, keepdim=True)
        dev = x1_train.std(0, keepdim=True)
        x1_train = (x1_train - mean) / dev
        x1_test = (x1_test - mean) / dev

        mean = x2_train.mean(0, keepdim=True)
        dev = x2_train.std(0, keepdim=True)
        x2_train = (x2_train - mean) / dev
        x2_test = (x2_test - mean) / dev

        mean = x3_train.mean(0, keepdim=True)
        dev = x3_train.std(0, keepdim=True)
        x3_train = (x3_train - mean) / dev
        x3_test = (x3_test - mean) / dev

        mean = x4_train.mean(0, keepdim=True)
        dev = x4_train.std(0, keepdim=True)
        x4_train = (x4_train - mean) / dev
        x4_test = (x4_test - mean) / dev
    else:
        mean = torch.cat((x1_train,x2_train,x3_train,x4_train),0).mean(0, keepdim=True)
        dev = torch.cat((x1_train,x2_train,x3_train,x4_train),0).std(0, keepdim=True)
        x1_train = (x1_train - mean) / dev
        x1_test = (x1_test - mean) / dev
        x2_train = (x2_train - mean) / dev
        x2_test = (x2_test - mean) / dev
        x3_train = (x3_train - mean) / dev
        x3_test = (x3_test - mean) / dev
        x4_train = (x4_train - mean) / dev
        x4_test = (x4_test - mean) / dev

    train = TensorDataset(torch.cat((x1_train,x2_train,x3_train,x4_train),0),  torch.cat((y1_train,y2_train,y3_train,y4_train),0))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    test1 = TensorDataset(x1_test, y1_test)
    test_loader1 = DataLoader(test1, batch_size=args.test_batch_size1, shuffle=False)
    test2 = TensorDataset(x2_test, y2_test)
    test_loader2 = DataLoader(test2, batch_size=args.test_batch_size2, shuffle=False)
    test3 = TensorDataset(x3_test, y3_test)
    test_loader3 = DataLoader(test3, batch_size=args.test_batch_size3, shuffle=False)
    test4 = TensorDataset(x4_test, y4_test)
    test_loader4 = DataLoader(test4, batch_size=args.test_batch_size4, shuffle=False)

    model = MLP(6105,args.dim,2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-2)
    print(model)
    nnloss = nn.NLLLoss()

    def train(data_loader,epoch):
        model.train()

        if epoch <= 50 and epoch % 20 == 0:
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = 0.5 * param_group1['lr']
        elif epoch > 50 and epoch % 20 == 0:
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = 0.5 * param_group1['lr']

        loss_all1 = 0

        for data, target in data_loader:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output1 = model(data)
            loss1 = nnloss(output1, target)
            loss1.backward()
            loss_all1 += loss1.item() * target.size(0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        return loss_all1 / (len(data_loader.dataset))


    def test(data_loader,train=False):
        model.eval()
        test_loss = 0
        correct = 0
        outputs = []
        preds = []
        targets = []
        for data, target in data_loader:
            data = data.to(device)
            targets.append(target[0].detach().numpy())
            target = target.to(device)
            output = model(data)
            outputs.append(output.detach().cpu().numpy())
            test_loss += nnloss(output, target).item() * target.size(0)
            pred = output.data.max(1)[1]
            preds.append(pred.detach().cpu().numpy())
            correct += pred.eq(target.view(-1)).sum().item()

        test_loss /= len(data_loader.dataset)
        correct /= len(data_loader.dataset)
        if train:
            print('Train set: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss,correct))
        else:
            print('Test set: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss,correct))
        return test_loss, correct, targets, outputs, preds

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"Epoch Number {epoch + 1}")
        l1= train(train_loader,epoch)
        test(train_loader,train=True)
        print(' L1 loss: {:.4f}'.format(l1))
        print('===NYU===')
        _, acc1,targets1, outputs1, preds1 = test(test_loader1,train=False)
        print('===UM===')
        _, acc2,targets2, outputs2, preds2 = test(test_loader2, train=False)
        print('===USM===')
        _, acc3,targets3, outputs3, preds3 = test(test_loader3, train=False)
        print('===UCLA===')
        _, acc4,targets4, outputs4, preds4 = test(test_loader4, train=False)
        if (acc1+acc2+acc3+acc4)/4 >best_acc:
            best_acc = (acc1+acc2+acc3+acc4)/4
            best_epoch = epoch
        total_time = time.time() - start_time
        print('Communication time over the network', round(total_time, 2), 's\n')
    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, os.path.join(args.model_dir, str(args.split) +'.pth'))
    dd.io.save(os.path.join(args.res_dir, 'NYU_' + str(args.split) + '.h5'),
                {'outputs': outputs1, 'preds': preds1, 'targets': targets1})
    dd.io.save(os.path.join(args.res_dir, 'UM_' + str(args.split) + '.h5'),
                {'outputs': outputs2, 'preds': preds2, 'targets': targets2})
    dd.io.save(os.path.join(args.res_dir, 'USM_' + str(args.split) + '.h5'),
                {'outputs': outputs3, 'preds': preds3, 'targets': targets3})
    dd.io.save(os.path.join(args.res_dir, 'UCLA_' + str(args.split) + '.h5'),
                {'outputs': outputs4, 'preds': preds4, 'targets': targets4})
    print('Best Acc:',best_acc, 'Best Epoch:', best_epoch)
    print('split:', args.split)

#===============================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--split', type=int, default=0, help='select 0-4 fold')
    # do not need to change
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clip')
    parser.add_argument('--dim', type=int, default=16,help='hidden dim of MLP')
    parser.add_argument('-bs', '--batch_size', type=int, default=250, help='training batch size')
    parser.add_argument('-tbs1', '--test_batch_size1', type=int, default=145, help='NYU test batch size')
    parser.add_argument('-tbs2', '--test_batch_size2', type=int, default=265, help='UM test batch size')
    parser.add_argument('-tbs3', '--test_batch_size3', type=int, default=205, help='USM test batch size')
    parser.add_argument('-tbs4', '--test_batch_size4', type=int, default=85, help='UCLA test batch size')
    parser.add_argument('--sepnorm', type=bool, default=True, help='normalization method')
    parser.add_argument('--id_dir', type=str, default='./idx')
    parser.add_argument('--res_dir', type=str, default='./result/mix_overlap')
    parser.add_argument('--vec_dir', type=str, default='./data/HO_vector_overlap')
    parser.add_argument('--model_dir', type=str, default='./model/mix_overlap')

    args = parser.parse_args()
    assert args.split in [0,1,2,3,4]
    main(args)