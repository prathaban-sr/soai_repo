'''Train CIFAR10 with PyTorch.'''
import os
import cv2
from typing import no_type_check

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image 

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
        self.wrongimgs = []

    def initializeData(self):
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        #transform_test = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #])
        
        transform_test2 = transforms.Compose([
            transforms.ToTensor(),
        ])


        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test2)#chaned to test2
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

    def show_errors(self):
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

        for x in wrong[:10]:
            self.wrongimgs.append(x[0])
            
        len (correct), len (wrong)
        for i in range (10):
            self.displayErrors(wrong[i][0],wrong[i][1])

    def show_attention(self):
        if self.device == 'cuda':
            target_layer = self.net.module.layer4[-1]
        else:
            target_layer = self.net.layer4[-1]

        '''
        input_tensors = []
        for in_img in wrongimgs_arr:
          # Create an input tensor image for your model..
          # Note: input_tensor can be a batch tensor with several images!
          print("in_img.shape:{}".format(in_img.shape))
          dtype = in_img.dtype
          mean = [0.5,0.5,0.5]
          std = [0.5,0.5,0.5]
          mean = torch.as_tensor(mean, dtype=dtype, device=in_img.device)
          std = torch.as_tensor(std, dtype=dtype, device=in_img.device)
          if (std == 0).any():
              raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
          if mean.ndim == 1:
              mean = mean.view(-1, 1, 1)
          if std.ndim == 1:
              std = std.view(-1, 1, 1)
          out_img = in_img.sub_(mean).div_(std)
    
          #out_img = preprocess_image(in_img.numpy().astype(np.uint8))
          input_tensors.append(out_img)

        print("After preprocess")

        input_tensors_t = torch.stack(input_tensors)
        print("input_tensors_t.shape:{}".format(input_tensors_t.shape))
        '''
        input_tensors_t = torch.stack(self.wrongimgs)
        
        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=self.net, target_layer=target_layer, use_cuda=True)
        
        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        target_category = None #281
        cam.batch_size = 10

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensors_t, target_category=target_category)
        heatmap_arr = []
        use_rgb = False
        for mask in grayscale_cam:
          heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
          if use_rgb:
              heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
          heatmap = np.float32(heatmap) / 255
          heatmap = np.transpose(heatmap, (2, 1, 0))
          heatmap_arr.append(heatmap)

        #if np.max(input_tensors_t[0]) > 1:
        #    raise Exception("The input image should np.float32 in the range [0, 1]")
        visualization_arr = []

        for idx,input_tensor in enumerate(input_tensors_t):
          cam = heatmap_arr[idx] + input_tensor.numpy()
          cam = cam / np.max(cam)
          visualization_arr.append(torch.from_numpy(np.uint8(255 * cam)))
          
        visualization_t = torch.stack(visualization_arr[:10])
        #visualization = show_cam_on_image(wrongimgs_arr, grayscale_cam)

        #return visualization
        grid_in = torchvision.utils.make_grid(input_tensors_t[:10], nrow=1)
        plt.figure(figsize=(32,32))
        plt.imshow(np.transpose(grid_in, (2,1,0)))

        print("Heatmap") 
        grid = torchvision.utils.make_grid(visualization_t, nrow=1)
        plt.figure(figsize=(32,32))
        plt.imshow(np.transpose(grid, (2,1,0)))
        

    def displayErrors(self,img,ps):
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        m = nn.Softmax()
        ps = m(ps)
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
        
