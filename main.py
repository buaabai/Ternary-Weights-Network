import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='Ternary-Weights-Network Pytorch MNIST Example.')
    parser.add_argument('--batch-size',type=int,default=100,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=100,metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs',type=int,default=20,metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval',type=int,default=100,metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():
    args = ParseArgs()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    #momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True,**kwargs)

    model = 
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    #bin_op = util.Binop(model)

    best_acc = 0.0 
    for epoch_index in range(1,args.epochs+1):
        adjust_learning_rate(learning_rate,optimizer,epoch_index,args.lr_epochs)
        train(args,epoch_index,train_loader,model,optimizer,criterion,bin_op)
        acc = test(model,test_loader,bin_op,criterion)
        if acc > best_acc:
            best_acc = acc
            #bin_op.Binarization()
            save_model(model,best_acc)
            #bin_op.Restore()

def train(args,epoch_index,train_loader,model,optimizer,criterion,bin_op):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()

        bin_op.Binarization()

        output = model(data)
        loss = criterion(output,target)
        loss.backward()

        bin_op.Restore()
        bin_op.UpdateBinaryGradWeight()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model,test_loader,bin_op,criterion):
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.Binarization()
    for data,target in test_loader:
        data,target = data.cuda(),target.cuda()
        data,target = Variable(data,volatile=True),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).data[0]
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    bin_op.Restore()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return acc
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    main()