import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from easydict import EasyDict

import numpy as np
import matplotlib.pyplot as plt

from wideresnet import *
#from resnet import *
from losses import*
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser = argparse.ArgumentParser(description='Weighted Perturbation Budget')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')


# settings
args = parser.parse_args()
model_dir = args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../jtian-lab/data_attack/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='../jtian-lab/data_attack/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)
"""

trainset = torchvision.datasets.SVHN(root='../jtian-lab/data_attack/', split='train', download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.SVHN(root='../jtian-lab/data_attack/', split='test', download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)
"""

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    m_pert = []
    count = 0
    count2 = 0
    total_values = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        #loss = 0.
      
        if  epoch <=80:
            loss= at_loss_initial(model=model,
                      x_natural=data,
                      y=target,
                      optimizer=optimizer,
                      step_size=args.step_size,
                      epsilon=args.epsilon,
                      perturb_steps=args.num_steps,
                      beta=args.beta)
            
            
        
        else:
            loss =  at_adapt(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,  
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
                   
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.) 
    loss = torch.sum(loss)
    return loss

        
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003, loss_fn = "CE"):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            ot = model(X_pgd)
            if loss_fn == "CE":
                loss = nn.CrossEntropyLoss()(ot, y)
            elif loss_fn == "CW":
                loss = cwloss(ot, y)
            else:
                raise RuntimeError('Invalid loss function') 
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        ot = model(X_pgd)
    err_pgd = (ot.data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    
    
    robust_err_cw_total = 0
    natural_err_cw_total = 0


    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y,loss_fn = "CE")
        #err_nat_cw, err_robust_cw = _pgd_whitebox(model, X, y,loss_fn = "CW")
       
        robust_err_total += err_robust
        natural_err_total += err_natural
        
        #robust_err_cw_total += err_robust_cw
        #natural_err_cw_total += err_nat_cw
        
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    
    #print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    #print('robust_cw_acc: ', 1- robust_err_cw_total / len(test_loader.dataset))
   
    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total / len(test_loader.dataset)


def main():
 
    model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   
    #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    rob_acc = 0.2000

    nat_acc = 0.9200
        
    natural_acc = []
    robust_acc = []
    
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        start_time = time.time()

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)


        print('================================================================')

        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader)

             
        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)
        print('using time:', time.time()-start_time)
        print('================================================================')
        

        if  robust_err_total > rob_acc:
            rob_acc = robust_err_total
           #nat_acc = natural_err_total
           # torch.save(model.state_dict(), 'trades-vanilla-wide-5e')
           #torch.save(model.state_dict(), '../weighted/5trades-exp3_plus1.4_wideresnet_7e_divg-epoch.pt')

        

        file_name = os.path.join(log_dir, 'train_stats.npy')
               

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))

  
if __name__ == '__main__':
    main()
