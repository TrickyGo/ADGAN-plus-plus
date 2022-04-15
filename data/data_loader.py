import torch.utils.data
from data.image_folder import make_dataset
from PIL import Image
import util.util as util
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random

def create_dataloader(opt):
    instance = SemanticImageDataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
        pin_memory=True
    )
    return dataloader



class SemanticImageDataset(data.Dataset):
    def __init__(self):
        super(SemanticImageDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, stage1_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if stage1_paths:
            util.natural_sort(stage1_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        stage1_paths = stage1_paths[:opt.max_dataset_size]


        self.label_paths = label_paths
        self.image_paths = image_paths
        self.stage1_paths = stage1_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.stage1_dir) > 0:
            stage1_dir = opt.stage1_dir
            stage1_paths = make_dataset(stage1_dir, recursive=False, read_cache=True)
        else:
            stage1_paths = []

        return label_paths, image_paths, stage1_paths


    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        label_path = self.label_paths[index]
        label = Image.open(label_path)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = 0 

        if self.opt.stage == '1':
            stage1_tensor = 0
        elif self.opt.stage == '2':
            stage1_path = self.stage1_paths[index]
            stage1 = Image.open(stage1_path)
            stage1 = stage1.convert('RGB')
            transform_stage1 = get_transform(self.opt, params)
            stage1_tensor = transform_stage1(stage1)


        input_dict = {'label': label_tensor,
                      'stage1': stage1_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh) 
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
