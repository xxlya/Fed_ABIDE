import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import time
import deepdish as dd
from networks import Classifier, Discriminator
import torch.distributions as tdist
import os
import argparse
import numpy as np
import copy
from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15
site = ['NYU','UM','USM','UCLA']
def main(args):
    torch.manual_seed(args.seed)
    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    log_dir = os.path.join('./log', 'Align_' + str(args.split))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)


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

    train1 = TensorDataset(x1_train, y1_train)
    train_loader1 = DataLoader(train1, batch_size=len(train1)//args.nsteps, shuffle=True)
    train2 = TensorDataset(x2_train, y2_train)
    train_loader2 = DataLoader(train2, batch_size=len(train2)//args.nsteps, shuffle=True)
    train3 = TensorDataset(x3_train, y3_train)
    train_loader3 = DataLoader(train3, batch_size=len(train3)//args.nsteps, shuffle=True)
    train4 = TensorDataset(x4_train, y4_train)
    train_loader4 = DataLoader(train4, batch_size=len(train4)//args.nsteps, shuffle=True)
    train_loaders = [train_loader1, train_loader2, train_loader3, train_loader4]

    test1 = TensorDataset(x1_test, y1_test)
    test2 = TensorDataset(x2_test, y2_test)
    test3 = TensorDataset(x3_test, y3_test)
    test4 = TensorDataset(x4_test, y4_test)
    test_loader1 = DataLoader(test1, batch_size=args.test_batch_size1, shuffle=False)
    test_loader2 = DataLoader(test2, batch_size=args.test_batch_size2, shuffle=False)
    test_loader3 = DataLoader(test3, batch_size=args.test_batch_size3, shuffle=False)
    test_loader4 = DataLoader(test4, batch_size=args.test_batch_size4, shuffle=False)
    tbs= [args.test_batch_size1, args.test_batch_size2, args.test_batch_size3, args.test_batch_size4]
    test_loaders = [test_loader1,test_loader2,test_loader3,test_loader4]


    # federated setup
    model1 = Classifier(6105,args.dim,2).to(device)
    model2 = Classifier(6105,args.dim,2).to(device)
    model3 = Classifier(6105,args.dim,2).to(device)
    model4 = Classifier(6105,args.dim,2).to(device)
    models = [model1, model2, model3, model4]
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr1, weight_decay=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr2, weight_decay=1e-3)
    optimizer3 = optim.Adam(model3.parameters(), lr=args.lr3, weight_decay=1e-3)
    optimizer4 = optim.Adam(model4.parameters(), lr=args.lr4, weight_decay=1e-3)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    optimizerG1 = optim.Adam(model1.encoder.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizerG2 = optim.Adam(model2.encoder.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizerG3 = optim.Adam(model3.encoder.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizerG4 = optim.Adam(model4.encoder.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizerGs = [optimizerG1, optimizerG2, optimizerG3, optimizerG4]



    discriminators = dict()
    optimizerDs = dict()
    for i in range(4):
        discriminators[i] = Discriminator(args.dim).to(device)
        optimizerDs[i] = optim.Adam(discriminators[i].parameters(), lr= args.lr, weight_decay=1e-3)


    #global model
    model = Classifier(6105,args.dim,2).to(device)
    print(model)

    # loss functions
    celoss = nn.CrossEntropyLoss()
    def advDloss(d1,d2):
        res = -torch.log(d1).mean()-torch.log(1-d2).mean()
        return res
    def advGloss(d1,d2):
        res = -torch.log(d1).mean()-torch.log(d2).mean()
        return res.mean()


    def train(epoch):
        pace = args.pace
        for i in range(4):
            models[i].train()
            if epoch <= 50 and epoch % 20 == 0:
                for param_group1 in optimizers[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']
            elif epoch > 50 and epoch % 20 == 0:
                for param_group1 in optimizers[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']
            if epoch <= 50 and epoch % 20 == 0:
                for param_group1 in optimizerGs[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']
            elif epoch > 50 and epoch % 20 == 0:
                for param_group1 in optimizerGs[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']

            discriminators[i].train()
            if epoch <= 50 and epoch % 20 == 0:
                for param_group1 in optimizerDs[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']
            elif epoch > 50 and epoch % 20 == 0:
                for param_group1 in optimizerDs[i].param_groups:
                    param_group1['lr'] = 0.5 * param_group1['lr']


        #define weights
        w = dict()
        denominator = np.sum(np.array(tbs))
        for i in range(4):
            w[i] = 0.25 #tbs[i]/denominator

        loss_all = dict()
        lossD_all = dict()
        lossG_all = dict()
        num_data = dict()
        num_dataG = dict()
        num_dataD = dict()
        for i in range(4):
            loss_all[i] = 0
            num_data[i] = EPS
            num_dataG[i] = EPS
            lossG_all[i] = 0
            lossD_all[i] = 0
            num_dataD[i] = EPS

        count = 0
        for t in range(args.nsteps):
            fs=[]

            # optimize classifier

            for i in range(4):
                optimizers[i].zero_grad()
                a, b= next(iter(train_loaders[i]))
                num_data[i] += b.size(0)
                a = a.to(device)
                b = b.to(device)
                output = models[i](a)
                loss = celoss(output,b)
                loss_all[i] += loss.item() * b.size(0)
                if epoch >=0:
                    loss.backward(retain_graph=True)
                    optimizers[i].step()

                fs.append(models[i].encoder(a))

            #optimize alignment

            nn = []
            noises = []
            for i in range(4):
                nn =tdist.Normal(torch.tensor([0.0]), 0.001 * torch.std(fs[i].detach().cpu()))
                noises.append(nn.sample(fs[i].size()).squeeze().to(device))


            for i in range(4):
                for j in range(4):
                    if i!=j:
                        optimizerDs[i].zero_grad()
                        optimizerGs[i].zero_grad()
                        optimizerGs[j].zero_grad()

                        d1 = discriminators[i](fs[i]+noises[i])
                        d2 = discriminators[i](fs[j]+noises[j])
                        num_dataG[i] += d1.size(0)
                        num_dataD[i] += d1.size(0)
                        lossD = advDloss(d1, d2)
                        lossG = advGloss(d1, d2)
                        lossD_all[i] += lossD.item() * d1.size(0)
                        lossG_all[i] += lossG.item() * d1.size(0)
                        lossG_all[j] += lossG.item() * d2.size(0)
                        lossD = 0.1*lossD
                        if epoch >= 5:
                            lossD.backward(retain_graph=True)
                            optimizerDs[i].step()
                            lossG.backward(retain_graph=True)
                            optimizerGs[i].step()
                            optimizerGs[j].step()
                        writer.add_histogram('Hist/hist_'+ site[i]+'2'+site[j]+'_source', d1, epoch*args.nsteps+t)
                        writer.add_histogram('Hist/hist_' + site[i]+'2'+site[j]+'_target', d2,
                                             epoch * args.nsteps + t)

            count += 1
            if count % pace == 0 or i == args.nsteps - 1:
                for key in model.state_dict().keys():
                    if models[0].state_dict()[key].dtype == torch.int64:
                        model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(model.state_dict()[key])
                        # add noise
                        for s in range(4):
                            if args.type == 'G':
                                nn = tdist.Normal(torch.tensor([0.0]),
                                                  args.noise * torch.std(models[s].state_dict()[key].detach().cpu()))
                            else:
                                nn = tdist.Laplace(torch.tensor([0.0]),
                                                   args.noise * torch.std(models[s].state_dict()[key].detach().cpu()))
                            noise = nn.sample(models[s].state_dict()[key].size()).squeeze()
                            noise = noise.to(device)
                            temp += w[s] * (models[s].state_dict()[key] + noise)
                        # update global model
                        model.state_dict()[key].data.copy_(temp)
                        # updata local model
                        for s in range(4):
                            models[s].state_dict()[key].data.copy_(model.state_dict()[key])

        return loss_all,lossG_all,lossD_all,num_data,num_dataG,num_dataD


    def test(federated_model,data_loader,train = False):
        federated_model= federated_model.to(device)
        federated_model.eval()
        test_loss = 0
        correct = 0
        outputs = []
        preds = []
        targets = []
        for data, target in data_loader:
            targets.append(target[0].detach().numpy())
            data = data.to(device)
            target = target.to(device)
            output = federated_model(data)
            outputs.append(output.detach().cpu().numpy())
            test_loss += celoss(output, target).item()*target.size(0)
            pred = output.data.max(1)[1]
            preds.append(pred.detach().cpu().numpy())
            correct += pred.eq(target.view(-1)).sum().item()

        test_loss /= len(data_loader.dataset)
        correct /= len(data_loader.dataset)
        if train:
            print('Train set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss, correct))
        else:
            print('Test set local: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss, correct))
        return test_loss, correct,  targets, outputs, preds


    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"Epoch Number {epoch + 1}")
        l,lG,lD,n,nG,nD = train(epoch)
        print("===========================")
        print("L1: {:.7f}, L2: {:.7f}, L3: {:.7f}, L4: {:.7f} ".format(l[0]/n[0],l[1]/n[1],l[2]/n[2],l[3]/n[3]))
        print("G1: {:.7f}, G2: {:.7f}, G3: {:.7f}, G4: {:.7f} ".format(lG[0] / nG[0], lG[1] / nG[1], lG[2] / nG[2],
                                                                       lG[3] / nG[3]))
        print("D1: {:.7f}, D2: {:.7f}, D3: {:.7f}, D4: {:.7f} ".format(lD[0] / nD[0], lD[1] / nD[1], lD[2] / nD[2],
                                                                       lD[3] / nD[3]))
        writer.add_scalars('CEloss', {'l1': l[0]/n[0],'l2': l[1]/n[1],'l3': l[2]/n[2],'l4': l[3]/n[3] }, epoch)
        writer.add_scalars('Gloss', {'gl1': lG[0] / nG[0], 'gl2': lG[1] / nG[1], 'gl3': lG[2] / nG[2], 'gl4': lG[3] / nG[3]},
                           epoch)
        writer.add_scalars('Dloss', {'dl1': lD[0]/ nD[0],
                                     'dl2': lD[1]/ nD[1],
                                     'dl3': lD[2]/ nD[2],
                                     'dl4': lD[3]/ nD[3]},
                           epoch)

        print('===NYU===')
        test(model, train_loader1, train=True)
        _, acc1,targets1, outputs1, preds1 = test(model, test_loader1, train=False)
        print('===UM===')
        test(model, train_loader2, train=True)
        _, acc2,targets2, outputs2, preds2 = test(model, test_loader2, train=False)
        print('===USM===')
        test(model, train_loader3, train=True)
        _, acc3,targets3, outputs3, preds3 = test(model, test_loader3, train=False)
        print('===UCLA===')
        test(model, train_loader4, train=True)
        _, acc4,targets4, outputs4, preds4 = test(model, test_loader4, train=False)
        if (acc1+acc2+acc3+acc4)/4 > best_acc:
            best_acc = (acc1+acc2+acc3+acc4)/4
            best_epoch = epoch
        total_time = time.time() - start_time
        print('Communication time over the network', round(total_time, 2), 's\n')
    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, os.path.join(args.model_dir, str(args.split) +'.pth'))
    print('Best Acc:',best_acc, 'Beat Epoch:', best_epoch)
    print('split:', args.split,'   noise:', args.noise, '   pace:', args.pace)
    dd.io.save(os.path.join(args.res_dir, 'NYU_' + str(args.split) + '.h5'),
                {'outputs': outputs1, 'preds': preds1, 'targets': targets1})
    dd.io.save(os.path.join(args.res_dir, 'UM_' + str(args.split) + '.h5'),
                {'outputs': outputs2, 'preds': preds2, 'targets': targets2})
    dd.io.save(os.path.join(args.res_dir, 'USM_' + str(args.split) + '.h5'),
                {'outputs': outputs3, 'preds': preds3, 'targets': targets3})
    dd.io.save(os.path.join(args.res_dir, 'UCLA_' + str(args.split) + '.h5'),
                {'outputs': outputs4, 'preds': preds4, 'targets': targets4})

#==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # specify for dataset site
    parser.add_argument('--split', type=int, default=0, help='select 0-4 fold')
    # do not need to change
    parser.add_argument('--pace', type=int, default=50, help='communication pace')
    parser.add_argument('--noise', type=float, default=0, help='noise level for gaussian or err level for Lap')
    parser.add_argument('--type', type=str, default='G', help='Gaussian or Lap')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--lr3', type=float, default=1e-4)
    parser.add_argument('--lr4', type=float, default=1e-4)
    parser.add_argument('--lamb', type=float, default=0.1, help = 'lossG weight')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clip')
    parser.add_argument('--dim', type=int, default=16,help='hidden dim of MLP')
    parser.add_argument('--nsteps', type=int, default=60, help='training steps/epoach')
    parser.add_argument('-tbs1', '--test_batch_size1', type=int, default=145, help='NYU test batch size')
    parser.add_argument('-tbs2', '--test_batch_size2', type=int, default=265, help='UM test batch size')
    parser.add_argument('-tbs3', '--test_batch_size3', type=int, default=205, help='USM test batch size')
    parser.add_argument('-tbs4', '--test_batch_size4', type=int, default=85, help='UCLA test batch size')
    parser.add_argument('--overlap', type=bool, default=True, help='augmentation method')
    parser.add_argument('--sepnorm', type=bool, default=False, help='normalization method')
    parser.add_argument('--id_dir', type=str, default='./idx')
    parser.add_argument('--res_dir', type=str, default='./result/align_overlap')
    parser.add_argument('--vec_dir', type=str, default='./data/HO_vector_overlap')
    parser.add_argument('--model_dir', type=str, default='./model/align_overlap')

    args = parser.parse_args()
    assert args.split in [0,1,2,3,4]
    main(args)