"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""
import pickle
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
from sample_place import list_pictures
from sample_place import Sampler
import scipy.misc


def ImageLoader(batch_size, triplets_name='../triplets.txt', train_flag=True, img_list=None, base_dir=None):
    """
    Tiny ImageNet Loader.

    Args:
        root:
        batch_size:
        triplets_name:
        train_flag:

    Return:
        loader
    """
    if train_flag:
        # Normalize training set together with augmentation
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Loading Tiny ImageNet dataset
        print("==> Preparing Tiny ImageNet dataset ...")

        trainset = TripletImageLoader(
             triplets_filename=triplets_name, transform=transform_train)
        loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=32)

    else:
        # Normalize test set same as training set without augmentation
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
        testset = TripletImageLoader(
            triplets_filename="", transform=transform_test, train=False, image_list=img_list, base_dir=base_dir)
        loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, num_workers=32)

    return loader


def cal_loss(input, tar):
    loss_cat = nn.CrossEntropyLoss().cuda()
    output = loss_cat(input, tar)
    return output


def get_predict(prob):
    y = torch.argmax(prob, dim=1)
    return y


def train(net, criterion, optimizer, scheduler, trainloader,
          start_epoch, epochs, is_gpu, batch_size=16):
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
    alpha = 0.99
    net.train()
    show_freq = 30
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        acc = 0
        loss_sum = 0
        loss_sum_t = 0
        loss_sum_c = 0
        for batch_idx, (data1, data2, data3, label_a, label_p, label_n) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n, cat_a, cat_p, cat_n = net(data1, data2, data3)
            loss_t = criterion(embedded_a, embedded_p, embedded_n)

            pre_a = get_predict(cat_a)
            pre_p = get_predict(cat_p)
            pre_n = get_predict(cat_n)

            label_a = label_a.cuda()
            label_p = label_p.cuda()
            label_n = label_n.cuda()


            # print((pre_a == label_a).float())
            # print((pre_a == label_a).float().sum())
            # print((pre_a == label_a).float().sum().data[0])

            acc += (pre_a == label_a).float().sum().data[0]
            acc += (pre_p == label_p).float().sum().data[0]
            acc += (pre_n == label_n).float().sum().data[0]

            loss_c = cal_loss(cat_a, label_a)
            loss_c += cal_loss(cat_p, label_p)
            loss_c += cal_loss(cat_n, label_n)

            loss = loss_t * alpha + loss_c * (1 - alpha)

            # compute gradient and do optimizer step
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            loss_sum += loss
            loss_sum_t += loss_t
            loss_sum_c += loss_c

            if batch_idx % show_freq == 0:
                loss_sum /= show_freq
                loss_sum_t /= show_freq
                loss_sum_c /= show_freq
                print("mini Batch Loss: {0} loss _t:{1} loss_c:{2}".format
                      (loss_sum.data[0], loss_sum_t.data[0], loss_sum_c.data[0]),
                      "acc:", acc / (show_freq * batch_size * 3))
                acc = 0
                loss_sum = 0
                loss_sum_t = 0
                loss_sum_c = 0

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


def save_to_txt(img_name, predicts):
    txt_name = img_name.split('.')[0]
    txt_name = txt_name + '.txt'
    txt_path = os.path.join('../result', txt_name)
    with open(txt_path, "w") as text_file:
        for predict in predicts:
            for img in predict:
                text_file.write(img)
                text_file.write('\n')


def test(net, is_gpu):

    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    # net.load_state_dict(checkpoint['state_dict'])
    sampler = Sampler()

    test_dir = '../test_dir2'
    test_images = os.listdir(test_dir)
    testloader = ImageLoader(batch_size=1, train_flag=False, img_list=test_images, base_dir=test_dir)

    net.eval()
    with torch.no_grad():
        for test_id, test_data in enumerate(testloader):

            if test_id % 5 == 0:
                print("Now processing {}th test image".format(test_id))

            if is_gpu:
                test_data = test_data.cuda()
            test_data = Variable(test_data)

            embedded_test, _, _ , pred_cat, _, _= net(test_data, test_data, test_data)
            # embedded_test = net.module.embeddingnet(test_data)
            # print(embedded_test)
            # print(net.module.sm(net.module.fc2(embedded_test)))
            # predict_cat = get_predict(net.module.sm(net.module.fc2(embedded_test)))
            predict_cat = get_predict(pred_cat)
            top_5_pred = pred_cat.topk(k=5, sorted=False)
            embedded_test_numpy = embedded_test.data.cpu().numpy()

            recom_imgs = []

            for pred in top_5_pred[1][0]:
                print(pred)
                dir_cat = sampler.get_dir(pred.item())
                kd_path = os.path.join(dir_cat, 'kd_tree.b')
                f = open(kd_path, 'rb')
                neigh = pickle.load(f)
                pred_imgs_id = neigh.kneighbors(embedded_test_numpy, return_distance=False)

                pred_imgs = []
                for id in pred_imgs_id:
                    pred_imgs.append(neigh.classes_.take(id))
                pred_imgs = np.array(pred_imgs)
                print('pred imgs: ', pred_imgs.shape)
                recom_imgs.append(pred_imgs[0])

                # print('test image', test_images[test_id])

            save_to_txt(test_images[test_id], recom_imgs)


def save_kd_tree(embedded_features, output_path, img_list):

    neigh = KNeighborsClassifier(
        n_neighbors=100, weights='distance', algorithm='kd_tree', n_jobs=-1)
    reshape_feature = embedded_features.reshape(-1, 2048)
    # print(reshape_feature.shape)
    neigh.fit(reshape_feature,
              np.array(img_list))

    f = open(output_path, 'wb')
    pickle.dump(neigh, f)


def save_embedded_txt(net, is_gpu):

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True
    path_root = "/home/cc4192/data/data/vision/torralba/deeplearning/images256/"

    net.eval()
    for direc in os.listdir(path_root):
        dir_path1 = os.path.join(path_root, direc)
        if not os.path.isdir(dir_path1):
            continue
        for dir2 in os.listdir(dir_path1):
            dir_path2 = os.path.join(dir_path1, dir2)
            print('processing:', dir_path2)
            if not os.path.isdir(dir_path2):
                continue

            embedded_list = None

            cnt = 0
            img_list = list_pictures(dir_path2)
            testloader = ImageLoader(batch_size=32, train_flag=False, img_list=img_list)
            flag = True
            for _, data1 in enumerate(testloader):
                if is_gpu:
                    data1 = data1.cuda()

                    # wrap in torch.autograd.Variable
                data1 = Variable(data1)
                embedded_train = net.module.embeddingnet(data1)
                embedded_feature = embedded_train.data.cpu().numpy()
                # print(embedded_feature.shape)
                if flag:
                    flag = False
                    embedded_list = embedded_feature
                else:
                    embedded_list = np.concatenate((embedded_feature, embedded_list))
                cnt += 1

            embedded_list = np.array(embedded_list)
            print('list_shape:', embedded_list.shape)
            # embedded_list.tofile(txt_path)
            save_kd_tree(embedded_list, os.path.join(dir_path2, 'kd_tree.b'), img_list)


def load_model(resume):
    net = TripletNet(resnet50(True))
    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    print('==> Retrieve model parameters ...')
    if resume:
        model_file = "../checkpoint/checkpoint.pth.tar"
        # checkpoint = torch.load()
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        if 'embeddingnet.fc1.bias' in state_dict:
            del state_dict['embeddingnet.fc1.bias']
            del state_dict['embeddingnet.fc1.weight']
        net.load_state_dict(state_dict)

    return net


if __name__ == '__main__':
    import argparse
    # net = TripletNet(resnet101())
    # test_image = scipy.misc.imread('../test_image.jpg')
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--ckptroot', type=str,
                        default="../checkpoint", help='path to checkpoint')

    parser.add_argument('--function', type=str, default='test')
    args = parser.parse_args()

    net = load_model(True)
    if args.function == 'test':
        # test(net, True)
        # net = load_model(True)
        test(net, True)
    elif args.function == 'savekd':
        save_embedded_txt(net, True)
    else:
        print('function not exist')
