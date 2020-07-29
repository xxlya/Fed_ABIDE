import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import deepdish as dd
from networks import MLP
import copy
import random
import os
import argparse
EPS = 1e-15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nnloss = nn.NLLLoss()


def main(args):
    torch.manual_seed(args.seed)
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(os.path.join(args.model_dir,args.site)):
        os.mkdir(os.path.join(args.model_dir,args.site))
    save_model_dir = os.path.join(args.model_dir,args.site)

    data = dd.io.load(os.path.join(args.vec_dir,args.site+'_correlation_matrix.h5'))
    x = torch.from_numpy(data['data']).float()
    y = torch.from_numpy(data['label']).long()

    if args.site == 'NYU':
        rep = 145 #7
    elif args.site == 'UM':
        rep = 265 #9
    elif args.site == 'USM':
        rep = 205 #8
    elif args.site == 'UCLA':
        rep = 85 #7

    split_dir = os.path.join(args.id_dir,args.site+'_sub_overlap.h5')

    if not os.path.exists(split_dir):  #save splitting
        n = len(y)//rep
        ll = list(range(n))
        random.seed(args.seed)
        random.shuffle(ll)
        list1 = dict()
        for i in range(5): # 5 splits
            list1[i] = list()
            if i!=4:
                temp = ll[i*n//5:(i+1)*n//5]
            else:
                temp = ll[4*n//5:]
            for t in temp:
                list1[i]+= [t*rep+j for j in range(rep)]

        dd.io.save(split_dir,{'0':list1[0],'1':list1[1],'2':list1[2],'3':list1[3],'4':list1[4]})
        print("data saved")

    id = dd.io.load(split_dir)

    if args.split==0:
        tr = id['1']+id['2']+id['3']+id['4']
        te = id['0']
    elif args.split==1:
        tr = id['0']+id['2']+id['3']+id['4']
        te = id['1']
    elif args.split==2:
        tr = id['0']+id['1']+id['3']+id['4']
        te = id['2']
    elif args.split==3:
        tr = id['0']+id['1']+id['2']+id['4']
        te = id['3']
    elif args.split==4:
        tr = id['0']+id['1']+id['2']+id['3']
        te = id['4']


    x_train = x[tr]
    y_train = y[tr]
    x_test = x[te]
    y_test = y[te]

    mean = x_train.mean(0, keepdim=True)
    dev = x_train.std(0, keepdim=True)
    x_train = (x_train - mean) / dev
    x_test = (x_test - mean) / dev


    train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test, batch_size=rep,shuffle= False)

    model = MLP(6105,args.dim,2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-2)
    print(model)

    def train(data_loader, optimizer, epoch):
        model.train()
        if epoch <= 50 and epoch % 20 == 0:
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = 0.5 * param_group1['lr']
        elif epoch > 50 and epoch % 20 == 0:
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = 0.5 * param_group1['lr']

        loss_all = 0

        for data, target in data_loader:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = nnloss(output, target)
            loss.backward()
            loss_all += loss.item() * target.size(0)
            optimizer.step()

        return loss_all / (len(data_loader.dataset))

    def test(data_loader, train):
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
            print('Train set: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss, correct))
        else:
            print('Test set: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss, correct))
            return targets, outputs, preds

    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"Epoch Number {epoch + 1}")
        l1 = train(train_loader,optimizer,epoch)
        print(' L1 loss: {:.4f}'.format(l1))
        test(train_loader, train=True)
        targets,outputs,preds = test(test_loader,train= False)
        total_time = time.time() - start_time
        print('Communication time over the network', round(total_time, 2), 's\n')
    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, os.path.join(save_model_dir, str(args.split) +'.pth'))
    dd.io.save(os.path.join(args.res_dir, args.site + '_' + str(args.split) + '.h5'),
                {'outputs': outputs, 'preds': preds, 'targets': targets})
    print('site:',args.site, '  split:', args.split)



#=============================================================================#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--site', type=str, default='UCLA')
    parser.add_argument('--split', type=int, default=0, help='select 0-4 fold')
    # do not need to change
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dim', type=int, default=8,help='hidden dim of MLP')
    parser.add_argument('-bs', '--batch_size', type=int, default=200, help='training batch size')
    parser.add_argument('--id_dir', type=str, default='./idx')
    parser.add_argument('--res_dir', type=str, default='./result/single_overlap')
    parser.add_argument('--vec_dir', type=str, default='./data/HO_vector_overlap')
    parser.add_argument('--model_dir', type=str, default='./model/single_overlap')



    args = parser.parse_args()

    assert args.site in ['NYU', 'UM', 'USM', 'UCLA']
    assert args.split in [0,1,2,3,4]
    main(args)