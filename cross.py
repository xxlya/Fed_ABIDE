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
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    if not os.path.exists(os.path.join(args.res_dir,args.trainsite)):
        os.mkdir(os.path.join(args.res_dir,args.trainsite))

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)


    torch.manual_seed(args.seed)

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

    if args.sepnorm:
        mean = x1.mean(0, keepdim=True)
        dev = x1.std(0, keepdim=True)
        x1 = (x1-mean)/dev
        mean = x2.mean(0, keepdim=True)
        dev = x2.std(0, keepdim=True)
        x2 = (x2 - mean) / dev
        mean = x3.mean(0, keepdim=True)
        dev = x3.std(0, keepdim=True)
        x3 = (x3 - mean) / dev
        mean = x4.mean(0, keepdim=True)
        dev = x4.std(0, keepdim=True)
        x4 = (x4 - mean) / dev

    else:
        if args.trainsite == 'NYU':
            mean = x1.mean(0, keepdim=True)
            dev = x1.std(0, keepdim=True)
        elif args.trainsite == 'UM':
            mean = x2.mean(0, keepdim=True)
            dev = x2.std(0, keepdim=True)
        elif args.trainsite == 'USM':
            mean = x3.mean(0, keepdim=True)
            dev = x3.std(0, keepdim=True)
        elif args.trainsite == 'UCLA':
            mean = x4.mean(0, keepdim=True)
            dev = x4.std(0, keepdim=True)
        x1 = (x1 - mean)/dev
        x2 = (x2 - mean) / dev
        x3 = (x3 - mean) / dev
        x4 = (x4 - mean) / dev


    datas = [TensorDataset(x1,y1),TensorDataset(x2,y2),TensorDataset(x3,y3),TensorDataset(x4,y4)]


    if args.trainsite == 'NYU':
        train_loader = DataLoader(datas[0], batch_size=args.batch_size, shuffle=True)
    elif args.trainsite == 'UM':
        train_loader = DataLoader(datas[1], batch_size=args.batch_size, shuffle=True)
    elif args.trainsite == 'USM':
        train_loader = DataLoader(datas[2], batch_size=args.batch_size, shuffle=True)
    elif args.trainsite == 'UCLA':
        train_loader = DataLoader(datas[3], batch_size=args.batch_size, shuffle=True)

    test_loader1 = DataLoader(datas[0], batch_size=args.test_batch_size1, shuffle=False)
    test_loader2 = DataLoader(datas[1], batch_size=args.test_batch_size2, shuffle=False)
    test_loader3 = DataLoader(datas[2], batch_size=args.test_batch_size3, shuffle=False)
    test_loader4 = DataLoader(datas[3], batch_size=args.test_batch_size4, shuffle=False)



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
            optimizer.step()

        return loss_all1 / (len(data_loader.dataset)), model


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
            output = federated_model(data)
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

    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"Epoch Number {epoch + 1}")
        l1,federated_model = train(train_loader,epoch)
        print(' L1 loss: {:.4f}'.format(l1))
        print('===NYU===')
        _, acc1, targets1, outputs1, preds1 = test(test_loader1, train=False)
        print('===UM===')
        _, acc2, targets2, outputs2, preds2 = test(test_loader2, train=False)
        print('===USM===')
        _, acc3, targets3, outputs3, preds3 = test(test_loader3, train=False)
        print('===UCLA===')
        _, acc4, targets4, outputs4, preds4 = test(test_loader4, train=False)
        total_time = time.time() - start_time
        print('Communication time over the network', round(total_time, 2), 's\n')

    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, os.path.join(args.model_dir, args.trainsite +'.pth'))
    dd.io.save(os.path.join(args.res_dir, args.trainsite, 'NYU.h5'),
                {'outputs': outputs1, 'preds': preds1, 'targets': targets1})
    dd.io.save(os.path.join(args.res_dir, args.trainsite, 'UM.h5'),
                {'outputs': outputs2, 'preds': preds2, 'targets': targets2})
    dd.io.save(os.path.join(args.res_dir, args.trainsite,'USM.h5'),
                {'outputs': outputs3, 'preds': preds3, 'targets': targets3})
    dd.io.save(os.path.join(args.res_dir, args.trainsite,'UCLA.h5'),
                {'outputs': outputs4, 'preds': preds4, 'targets': targets4})

#==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--trainsite', type=str, default='NYU', help='the site used for training')
    # do not need to change
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=2.0, help='gradient clip')
    parser.add_argument('--dim', type=int, default=8,help='hidden dim of MLP')
    parser.add_argument('-bs','--batch_size', type=int, default=250, help='training batch size')
    parser.add_argument('-tbs1', '--test_batch_size1', type=int, default=145, help='NYU test batch size')
    parser.add_argument('-tbs2', '--test_batch_size2', type=int, default=265, help='UM test batch size')
    parser.add_argument('-tbs3', '--test_batch_size3', type=int, default=205, help='USM test batch size')
    parser.add_argument('-tbs4', '--test_batch_size4', type=int, default=85, help='UCLA test batch size')
    parser.add_argument('--sepnorm', type=bool, default=False, help='normalization method')
    parser.add_argument('--overlap', type=bool, default=True, help='augmentation method')
    parser.add_argument('--res_dir', type=str, default='./result/cross_overlap')
    parser.add_argument('--model_dir', type=str, default='./model/cross_overlap')


    args = parser.parse_args()
    assert args.trainsite in ['NYU', 'UM', 'USM', 'UCLA']
    main(args)