"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import os
import sys
import shutil
import numpy as np

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from net import *
from numpy import linalg as LA
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from imageloader import TripletImageLoader
import scipy.misc

def TinyImageNetLoader(root, batch_size_train, batch_size_test):
    """
    Tiny ImageNet Loader.

    Args:
        train_root:
        test_root:
        batch_size_train:
        batch_size_test:

    Return:
        trainloader:
        testloader:
    """
    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Loading Tiny ImageNet dataset
    print("==> Preparing Tiny ImageNet dataset ...")

    trainset = TripletImageLoader(
        base_path=root, triplets_filename="../triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=32)

    testset = TripletImageLoader(
        base_path=root, triplets_filename="", transform=transform_test, train=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=32)

    return trainloader, testloader


def cal_loss(input, tar):
    loss_cat = nn.CrossEntropyLoss()
    output = loss_cat(input, tar)
    return output


def get_predict(prob):
    y = torch.argmax(prob, dim=1)
    return y


def train(net, criterion, optimizer, scheduler, trainloader,
          testloader, start_epoch, epochs, is_gpu, batch_size=16):
    """
    Training process.
    Args:
        net: Triplet Net
        criterion: TripletMarginLoss
        optimizer: SGD with momentum optimizer
        scheduler: scheduler
        trainloader: training set loader
        testloader: test set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU
    """
    print("==> Start training ...")
    alpha = 1
    net.train()
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        acc = 0
        for batch_idx, (data1, data2, data3, label_a, label_p, label_n) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n, cat_a, cat_p, cat_n = net(data1, data2, data3)
            loss = alpha * criterion(embedded_a, embedded_p, embedded_n)

            pre_a = get_predict(cat_a)
            pre_p = get_predict(cat_p)
            pre_n = get_predict(cat_n)

            label_a = label_a.cuda()
            label_p = label_p.cuda()
            label_n = label_n.cuda()

            # print(pre_a, label_a)
            # print((pre_a == label_a).float())
            # print((pre_a == label_a).float().sum())
            # print((pre_a == label_a).float().sum().data[0])

            acc += (pre_a == label_a).float().sum().data[0]
            acc += (pre_p == label_p).float().sum().data[0]
            acc += (pre_n == label_n).float().sum().data[0]

            loss += cal_loss(cat_a, label_a)
            loss += cal_loss(cat_p, label_p)
            loss += cal_loss(cat_n, label_n)

            # compute gradient and do optimizer step
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]


            if batch_idx % 30 == 0:
                print("mini Batch Loss: {}".format(loss.data[0]), "acc:", acc / (30 * batch_size))
                acc = 0

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        print("Training Epoch: {0} | Loss: {1}".format(epoch+1, running_loss))

        # remember best acc and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
        }, False)

    print('==> Finished Training ...')


def calculate_distance(i1, i2):
    """
    Calculate euclidean distance of the ranked results from the query image.

    Args:
        i1: query image
        i2: ranked result
    """
    return np.sum((i1 - i2) ** 2)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint."""
    directory = "../checkpoint"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


def test(net, is_gpu):
    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu


    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        print('use gpu')
        cudnn.benchmark = True

    """
    training_images = []
    for line in open("../triplets.txt"):
        line_array = line.split(",")
        if line_array[0] not in training_images:
            training_images.append(line_array[0])

    """
    train_path = os.path.join('../train_img')
    training_images = os.listdir(train_path)

    embedded_features_train = np.fromfile(
        "../embedded_features_train.txt", dtype=np.float32)

    neigh = KNeighborsClassifier(
        n_neighbors=1, weights='distance', algorithm='kd_tree', n_jobs=-1)

    reshape_feature = embedded_features_train.reshape(-1, 4096)
    print(reshape_feature.shape)
    neigh.fit(embedded_features_train.reshape(-1, 4096),
              np.array(training_images))

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    testset = TripletImageLoader(
        base_path='', triplets_filename="", transform=transform_test, train=False, path='../test_dir')
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, num_workers=32)
    embedded_features_t = []

    test_images = os.listdir('../test_dir')
    with torch.no_grad():
        for test_id, test_data in enumerate(testloader):

            if test_id % 5 == 0:
                print("Now processing {}th test image".format(test_id))

            if is_gpu:
                test_data = test_data.cuda()
            test_data = Variable(test_data)

            embedded_test, _, _ = net(test_data, test_data, test_data)
            embedded_test_numpy = embedded_test.data.cpu().numpy()

            embedded_features_t.append(embedded_test_numpy)

        embedded_features_test = np.concatenate(embedded_features_t, axis=0)
        pred_img = neigh.predict(embedded_features_test)
        print(pred_img)
        print(test_images)



def save_embedded_txt(is_gpu):

    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    dataroot = ""
    trainloader, testloader = TinyImageNetLoader(dataroot,
                                                 30,
                                                 30)
    print('==> Retrieve model parameters ...')
    checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    net.load_state_dict(checkpoint['state_dict'])

    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    embedded_list = []
    path_root = os.path.join("..", "train_img")

    # for direc in os.listdir(path_root):
    for _ in range(1):

        path = path_root

        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        trainset = TripletImageLoader(
            base_path='', triplets_filename="../triplets.txt", train=False, transform=transform_train, path=path)
        train_images = torch.utils.data.DataLoader(
            trainset, batch_size=1, num_workers=32)
        # train_images = os.listdir(path)
        cnt = 0

        for _, data1 in enumerate(train_images):
            if is_gpu:
                data1 = data1.cuda()

                # wrap in torch.autograd.Variable
            data1 = Variable(data1)
            embedded_train1, _, _ = net(data1, data1, data1)
            embedded_feature = embedded_train1.data.cpu().numpy()
            print(embedded_feature.shape)
            embedded_list.append(embedded_feature)
            cnt += 1
        embedded_list = np.array(embedded_list)
    print(embedded_list.shape)
    embedded_list.tofile(path_root + "/../embedded_features_train.txt")

if __name__ == '__main__':
    net = TripletNet(resnet101())
    # test_image = scipy.misc.imread('../test_image.jpg')
    test(net, False)

