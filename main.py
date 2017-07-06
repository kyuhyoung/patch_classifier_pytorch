# reference : https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/Colorful/src/model/models_colorful.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.datasets.utils import check_integrity, download_url
from os.path import join, exists, abspath, basename
from os import makedirs, listdir, getcwd, chdir
from PIL import Image
from time import time
import sys, os, cv2

import torch.utils.data.dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from ellie import Ellie



class ImageNetCustomFile(torch.utils.data.Dataset):
    #def __init__(self, dataset_path, data_size, data_transform, li_label, ext_img):
    def __init__(self, dataset_path, size, ext_img):
        self.dataset_path = dataset_path
        self.size_img = (size, size)
        #self.num_samples = data_size
        #self.transform = data_transform
        self.li_fn_img = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            print('Reading image names under %s' % (dirpath))
            self.li_fn_img += \
                [join(dirpath, f) for f in filenames
                 if f.lower().endswith(ext_img.lower())]
            if self.__len__() > 100:
                break
        return

    def __getitem__(self, index):

        fn_img = self.li_fn_img[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        #im_rgb = Image.open(fn_img).convert('RGB')
        im_bgr = cv2.imread(fn_img)
        im_bgr = cv2.resize(im_bgr, self.size_img, interpolation=cv2.INTER_AREA)
        #if self.transform is not None:
            #img = self.transform(img)
        im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
        #t1, t2 = torch.from_numpy(im_lab[:, :, 0:1]), torch.from_numpy(im_lab[:, :, 1:])
        #return torch.from_numpy(im_lab[:, :, 0:1]), torch.from_numpy(im_lab[:, :, 1:])
        #return im_lab[:, :, 0:1].astype(np.float32), torch.from_numpy(im_lab[:, :, 1:].astype(np.float32))
        #t1, t2 = ToTensor2(im_lab[:, :, 0:1]), ToTensor2(im_lab[:, :, 1:])
        return ToTensor2(im_lab[:, :, 0:1]), ToTensor2(im_lab[:, :, 1:])

    def __len__(self):
        return len(self.li_fn_img)


def softmaxND(input, axis=1):

    input_size = input.size()
    t1 = len(input_size) - 1
    trans_input = input.transpose(axis, t1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)

def softmax2D(input, axis=1):
    input_size = input.size()
    t1 = len(input_size) - 1
    trans_input = input.transpose(axis, t1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    return soft_max_2d, input_size, trans_size, axis


def convolutional_block(x, block_idx, n_input_channel, nb_filter,
                        nb_conv, subsample):
    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        print(name)
        if i < nb_conv - 1:
            # x = Convolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
            x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1)(x)
        else:
            # x = Convolution2D(nb_filter, 3, 3, name=name, subsample=subsample, border_mode="same")(x)
            x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1, stride=subsample)(x)
        n_input_channel = nb_filter
        # x = BatchNormalization(mode=2, axis=1)(x)
        x = nn.BatchNorm2d(nb_filter)(x)
        # x = Activation("relu")(x)
        x = F.relu(x)
    return x

def atrous_block(x, block_idx, n_input_channel, nb_filter, nb_conv):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        print(name)
        #x = AtrousConvolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
        #x = nn.Conv2d(n_input_channel, nb_filter, 3, dilation=?)(x)
        x = nn.Conv2d(n_input_channel, nb_filter, 3, padding=1)(x)

        #x = BatchNormalization(mode=2, axis=1)(x)
        x = nn.BatchNorm2d(nb_filter)(x)
        #x = Activation("relu")(x)
        x = F.relu(x)
        n_input_channel = nb_filter
    return x

def ToTensor2(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    to a torch.FloatTensor of shape (C x H x W)
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        #return img.float().div(255)
        return img.float()
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        #return img.float().div(255)
        return img.float()
    else:
        return img


def check_if_uncompression_done(dir_save, foldername_train, foldername_test):

    #base_folder = 'imagenet'
    #fpath = join(dir_save, base_folder)
    if not exists(dir_save):
        return False
    fpath_train = join(dir_save, foldername_train)
    if not exists(fpath_train):
        return False
    fpath_test = join(dir_save, foldername_test)
    if not exists(fpath_test):
        return False
    return True



def check_if_download_done(dir_save, foldername_train, foldername_test):

    if not check_if_uncompression_done(dir_save, foldername_train, foldername_test):
        filename = "imagenet.tar.gz"
        fn = join(dir_save, filename)
        return exists(fn)
    return True

def check_if_image_set_exists(dir_save, li_label, n_im_per_label, ext_img):
    does_exist = True
    for label in li_label:
        dir_label = join(dir_save, label)
        if exists(dir_label):
            li_img_file = [file for file in listdir(dir_label) if file.endswith(ext_img)]
            n_img = len(li_img_file)
            if n_im_per_label != n_img:
                does_exist = False
                break
        else:
            does_exist = False
            break
    return does_exist



def saveTrainImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):#byte_per_image, ):

    #data = {}
    #dataMean = np.zeros((3, ImgSize, ImgSize))
    dir_train = join(dir_save, foldername)
    i_total = 0
    for ifile in range(1, 6):
        fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'data_batch_' + str(ifile))
        with open(fn_batch, 'rb') as f:
            if sys.version_info[0] < 3:
                data = cp.load(f)
            else:
                data = cp.load(f, encoding='latin1')
            for i in range(n_im_per_batch):
                i_total += 1
                idx_label = data['labels'][i]
                name_label = li_label[idx_label]
                dir_label = abspath(join(dir_train, name_label))
                if not exists(dir_label):
                    makedirs(dir_label)
                fname = join(dir_label, ('%05d.%s' % (i + (ifile - 1) * 10000, ext_img)))
                saveImage(fname, data['data'][i, :])
                print('Saved %d th image of %s at %s' % (i_total, name_label, fname))

                #saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    #dataMean = dataMean / (50 * 1000)
    #saveMean('CIFAR-10_mean.xml', dataMean)
    return

def saveTestImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):

    #if not os.path.exists(foldername):
        #os.makedirs(foldername)
    dir_test = join(dir_save, foldername)
    fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'test_batch')
    i_total = 0
    with open(fn_batch, 'rb') as f:
        if sys.version_info[0] < 3:
            data = cp.load(f)
        else:
            data = cp.load(f, encoding='latin1')
        for i in range(n_im_per_batch):
            i_total += 1
            idx_label = data['labels'][i]
            name_label = li_label[idx_label]
            dir_label = abspath(join(dir_test, name_label))
            if not exists(dir_label):
                makedirs(dir_label)
            fname = join(dir_label, ('%05d.%s' % (i, ext_img)))
            saveImage(fname, data['data'][i, :])
            print('Saved %d th image of %s at %s' % (i_total, name_label, fname))



def prepare_imagenet_dataset(dir_save, foldername_train,
                             foldername_test):

    #dir_save = './data'
    #n_im_per_label_train, n_im_per_label_test = 5000, 1000
    #foldername_train, foldername_test = 'train', 'test'
    if not check_if_download_done(dir_save, foldername_train, foldername_test):
        print('The imagenet file has not been downloaded yet')
        sys.exit(1)
    if not check_if_uncompression_done(dir_save, foldername_train, foldername_test):
        print('The imagenet file has not been uncompressed yet')
        sys.exit(1)


#def make_dataloader_custom_file(dir_data, data_transforms, ext_img,                                n_img_per_batch, n_worker):
def make_dataloader_custom_file(dir_data, size_img, ext_img,
                                n_img_per_batch, n_worker):

    foldername_train, foldername_test = 'train', 'val'
    prepare_imagenet_dataset(dir_data, foldername_train,
                                 foldername_test)
    li_set = [foldername_train, foldername_test]
    #data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: ImageNetCustomFile(
        #join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
        join(dir_data, x), size_img, ext_img)
             for x in li_set}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x], batch_size=n_img_per_batch, shuffle=True, num_workers=n_worker) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]

    return trainloader, testloader#, li_class


def categorical_crossentropy_color(y_true, y_pred):

    print(y_true.size())
    print(y_pred.size())

    # Flatten
    n, h, w, q = y_true.shape
    y_true = K.reshape(y_true, (n * h * w, q))
    y_pred = K.reshape(y_pred, (n * h * w, q))

    weights = y_true[:, 313:]  # extract weight from y_true
    weights = K.concatenate([weights] * 313, axis=1)
    y_true = y_true[:, :-1]  # remove last column
    y_pred = y_pred[:, :-1]  # remove last column

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent

def initialize(is_gpu, dir_data, size_img, #di_set_transform,
               ext_img, n_img_per_batch, n_worker, shortcut_type):

    trainloader, testloader =\
        make_dataloader_custom_file(
            dir_data, size_img, #di_set_transform,
            ext_img, n_img_per_batch, n_worker)

    #net = Net().cuda()
    #net = Net(n_class, n_img_per_batch)
    net = Ellie(shortcut_type)
    #t1 = net.cuda()
    criterion = nn.CrossEntropyLoss()
    if is_gpu:
        net.cuda()
        criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience = 8, epsilon=0.00001, min_lr=0.000001) # set up scheduler

    return trainloader, testloader, net, criterion, optimizer, scheduler#, li_class



def validate_epoch(net, n_loss_rising, loss_avg_pre, ax,
                   li_n_img_val, li_loss_avg_val,
                   testloader, criterion, th_n_loss_rising,
                   kolor, n_img_train, sec, is_gpu):
    net.eval()
    shall_stop = False
    sum_loss = 0
    n_img_val = 0
    start_val = time()
    for i, data in enumerate(testloader):
        inputs, labels = data
        n_img_4_batch = labels.size()[0]
        if is_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        #images, labels = images.cuda(), labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.data[0]
        n_img_val += n_img_4_batch

    lap_val = time() - start_val
    loss_avg = sum_loss / n_img_val
    if loss_avg_pre <= loss_avg:
        n_loss_rising += 1
        if n_loss_rising >= th_n_loss_rising:
            shall_stop = True
    else:
        n_loss_rising = max(0, n_loss_rising - 1)
    li_n_img_val.append(n_img_train)
    li_loss_avg_val.append(loss_avg)
    ax.plot(li_n_img_val, li_loss_avg_val, c=kolor)
    plt.pause(sec)
    loss_avg_pre = loss_avg
    return shall_stop, net, n_loss_rising, loss_avg_pre, ax, \
           li_n_img_val, li_loss_avg_val, lap_val, n_img_val


def train_epoch(
        net, trainloader, optimizer, criterion, scheduler, n_img_total,
        n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
        li_n_img, li_loss_avg_train, ax_loss, sec, epoch,
        kolor, interval_train_loss, is_gpu):
    shall_stop = False
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        n_img_4_batch = labels.size()[0]
        # wrap them in Variable
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        if is_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # labels += 10
        #loss = criterion(outputs, labels)
        loss = categorical_crossentropy_color(outputs, labels)
        loss.backward()
        optimizer.step()
        # n_image_total += labels.size()[0]
        # print statistics
        running_loss += loss.data[0]
        #n_image_total += n_img_per_batch
        n_img_total += n_img_4_batch
        n_img_interval += n_img_4_batch

        #if n_image_total % interval_train_loss == interval_train_loss - 1:  # print every 2000 mini-batches
        #if n_image_total % interval_train_loss == 0:  # print every 2000 mini-batches
        if n_img_total > n_img_milestone:  # print every 2000 mini-batches

            # if i % 2000 == 1999:    # print every 2000 mini-batches
            running_loss_avg = running_loss / n_img_interval
            li_n_img.append(n_img_total)
            li_loss_avg_train.append(running_loss_avg)
            ax_loss.plot(li_n_img, li_loss_avg_train, c=kolor)
            plt.pause(sec)
            #i_batch += 1
            print('[%d, %5d] avg. loss per image : %.5f' %
                  (epoch + 1, i + 1, running_loss_avg))
            is_best_changed, is_lr_decayed = scheduler.step(
                running_loss_avg, n_img_total)  # update lr if needed
            running_loss = 0.0
            n_img_interval = 0
            n_img_milestone = n_img_total + interval_train_loss
            #'''
            #if is_lr_just_decayed and (not is_best_changed):
            if is_lr_just_decayed and is_lr_decayed:
                shall_stop = True
                break
            #'''
            is_lr_just_decayed = is_lr_decayed
    return shall_stop, net, optimizer, scheduler, n_img_total, n_img_interval, \
           n_img_milestone, running_loss, li_n_img, li_loss_avg_train, ax_loss_train, \
           is_lr_just_decayed, i + 1








def prepare_display(interval_train_loss, lap_init, color_time, sec):
    #fig = plt.figure(num=None, figsize=(1, 2), dpi=500)
    fig = plt.figure(num=None, figsize=(12, 18), dpi=100)
    plt.ion()
    ax_time = fig.add_subplot(2, 1, 1)
    ax_time.set_title(
        'Elapsed time (sec.) of validation on 10k images vs. epoch. Note that value for epoch 0 is the elapsed time of init.')
    ax_time.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss = fig.add_subplot(2, 1, 2)
    ax_loss.set_title('Avg. train and val. loss per image vs. # train input images')
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    li_n_img_train, li_n_img_val, li_loss_avg_train, li_loss_avg_val = [], [], [], []
    li_lap, li_epoch = [lap_init], [0]
    n_img_milestone = interval_train_loss

    ax_time.plot(li_epoch, li_lap, c = color_time)
    plt.pause(sec)

    return ax_time, ax_loss, \
           li_n_img_train, li_n_img_val, li_loss_avg_train, li_loss_avg_val, \
           li_lap, li_epoch, n_img_milestone




def train(is_gpu, trainloader, testloader, net, criterion, optimizer, scheduler, #li_class,
          n_epoch, lap_init, n_img_per_batch, interval_train_loss):

    sec = 0.01
    is_lr_just_decayed = False
    n_image_total, n_img_interval, running_loss = 0, 0, 0.0
    n_loss_rising, th_n_loss_rising, loss_avg_pre = 0, 3, 100000000000
    di_ax_color = {'time' : np.random.rand(3), 'train' : np.random.rand(3),
                   'val' : np.random.rand(3)}
    ax_time, ax_loss, li_n_img_train, li_n_img_val, \
    li_loss_avg_train, li_loss_avg_val, li_lap, li_epoch, n_img_milestone = \
        prepare_display(interval_train_loss, lap_init, di_ax_color['time'], sec)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print('epoch : %d' % (epoch + 1))
        shall_stop_train, net, optimizer, scheduler, n_image_total, n_img_interval, \
        n_img_milestone, running_loss, li_n_img_train, li_loss_avg_train, ax_loss_train, \
        is_lr_just_decayed, n_batch = train_epoch(
            net, trainloader, optimizer, criterion, scheduler, n_image_total,
            n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
            li_n_img_train, li_loss_avg_train, ax_loss, sec, epoch,
            di_ax_color['train'], interval_train_loss, is_gpu)
        shall_stop_val, net, n_loss_rising, loss_avg_pre, ax_loss_val, \
        li_n_img_val, li_loss_avg_val, lap_val, n_img_val = \
            validate_epoch(
                net, n_loss_rising, loss_avg_pre, ax_loss,
                li_n_img_val, li_loss_avg_val,
                testloader, criterion, th_n_loss_rising, di_ax_color['val'],
                n_image_total, sec,         is_gpu)
        #lap_train = time() - start_train
        n_batch_val = n_img_val / n_img_per_batch
        lap_batch = lap_val / n_batch_val
        li_lap.append(lap_val)
        li_epoch.append(epoch + 1)
        ax_time.plot(li_epoch, li_lap, c=kolor)
        ax_time.legend()
        plt.pause(sec)
        if shall_stop_train or shall_stop_val:
            break
    '''
    ax_time.plot(li_epoch, li_lap, c=kolor, label=legend)
    ax_time.legend()
    ax_loss_train.plot(li_n_img_train, li_loss_avg_train, c=kolor, label=legend)
    ax_loss_train.legend()
    ax_loss_val.plot(li_n_img_val, li_loss_avg_val, c=kolor, label=legend)
    ax_loss_val.legend()
    plt.pause(sec)
    '''
    print('Finished Training')

    return

def main():

    is_gpu = False
    #is_gpu = torch.cuda.device_count() > 0
    #dir_data = './data'
    dir_data = '/mnt/data/data/imagenet'
    ext_img = 'jpeg'
    #n_epoch = 100
    n_epoch = 50
    #n_img_per_batch = 40
    #n_img_per_batch = 60
    n_img_per_batch = 2
    n_worker = 4
    size_img = 256
    #n_class = 300
    shortcut_type = 'B'
    interval_train_loss = int(round(20000 / n_img_per_batch)) * n_img_per_batch

    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    di_set_transform = {'train' : transform, 'test' : transform}
    '''
    start = time()
    trainloader, testloader, net, criterion, optimizer, scheduler =\
        initialize(
            is_gpu, dir_data, size_img, #di_set_transform,
            ext_img, n_img_per_batch, n_worker, shortcut_type)
    lap_init = time() - start
    train(is_gpu, trainloader, testloader, net, criterion, optimizer, scheduler,  # li_class,
          n_epoch, lap_init, n_img_per_batch, interval_train_loss)

    return

if __name__ == "__main__":
    main()
