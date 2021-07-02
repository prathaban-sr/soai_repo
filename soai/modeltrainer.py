'''Train CIFAR10 with PyTorch.'''
import os
from typing import no_type_check

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from soai.models.resnet import *
from soai.utils import progress_bar

class ModelTrainer:
    def __init__(self):
        self.lr= 0.01
        self.criterion = nn.CrossEntropyLoss()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        self.trainloader, self.testloader = self.initializeData()
        self.net, self.optimizer, self.scheduler = self.setupModel(self.device, self.lr)

    def initializeData(self):
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        
        return trainloader, testloader

    def setupModel(self, device, lr):
        # Model
        print('==> Building model..')
        net = ResNet18()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        optimizer = optim.SGD(net.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                    
        return net, optimizer, scheduler
    
    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        print("Train Loss:{:.3f},Acc:{:.3f}({}/{})".format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        epoch_train_loss = train_loss/(batch_idx+1)
        epoch_train_acc = 100.*correct/total
        return epoch_train_loss, epoch_train_acc

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print("Test Loss:{:.3f},Acc:{:.3f}({}/{})".format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc

        epoch_test_loss = test_loss/(batch_idx+1)
        epoch_test_acc = 100.*correct/total
        return epoch_test_loss, epoch_test_acc

    def showErrors(self):
        correct = []
        wrong = []
        for i in range (20):
            images, labels = next (iter (self.testloader))

        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = self.net (images.cuda())

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp (logps)
        probabs = []
        for j, element in enumerate (ps):
            pred = torch.argmax(element)
            if pred == labels[j].item ():
                correct.append ([images[j], ps[j], pred, labels[j].item ()])
            else:
                wrong.append ([images[j], ps[j], pred, labels[j].item ()])

        len (correct), len (wrong)
        for i in range (10):
            self.displayErrors(wrong[i][0],wrong[i][1])

    def displayErrors(self,img,ps):
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        m = nn.Softmax()
        ps = m(ps)
        print(ps)
        ps = ps.cpu().data.numpy().squeeze()

        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img.permute(1,2,0)) #(height,width,dim)
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(classes)
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()            

'''
if __name__ == "__main__":
    trainer = ModelTrainer()
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+200):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.scheduler.step()
'''