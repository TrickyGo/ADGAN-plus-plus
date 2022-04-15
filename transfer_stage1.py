import os
from collections import OrderedDict
import torch
import data.data_loader as data_loader
from options.test_options import TestOptions
from models.ADGANPP import Model
from util.visualizer import Visualizer
from util import html

def select_part(origin_data, part_idx):

    data = origin_data.copy()
    data['label'] = data['label'].long()
    data['label'] = data['label'].cuda(non_blocking=True)
    data['image'] = data['image'].cuda(non_blocking=True)

    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.label_nc 

    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    label = input_label.scatter_(1, label_map, 1.0)
    
    image = data['image']
    label  = label[:, part_idx, :, :]

    if torch.sum(label[:,:,:]) == 0:
        return None

    label  = torch.unsqueeze(label, 1)
    data['label'] = label
    label = label.repeat(1, image.size(1), 1, 1)
    data['image'] = image.mul(label)

    return data

def fill_part(tofill, fill, fill_label):
    if tofill is None:
        return fill
    tofill = tofill.cuda(non_blocking=True)
    fill_mask = fill_label.repeat(1,3,1,1).cuda(non_blocking=True)
    filled = torch.where(fill_mask != 0, fill, tofill)
    return filled

opt = TestOptions().parse()
opt.status = 'test'
dataloader = data_loader.create_dataloader(opt)

model = Model(opt)
model.eval()

visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

transfer_idx = [i for i in range(1, 150)]

for i, data_i in enumerate(dataloader):
    
    if i * opt.batchSize >= opt.how_many:
            break
    if i % 2 == 0:
        data_content = data_i
        img_path_content = data_content['path']
        tofill = data_content['image']
        
    else:
        data_style = data_i
        img_path_style = data_style['path']
        print('process image... %s' % img_path_style) 

        for part_idx in range(1,opt.label_nc):
            opt.part_idx = part_idx
            data_content_part = select_part(data_content, part_idx)
            data_style_part = select_part(data_style, part_idx)
            if data_content_part is None:
                continue
            
            if data_style_part is not None and part_idx in transfer_idx:
                print("transfering part",part_idx)
                generated = model((data_content_part, data_style_part), mode='transfer')

                for b in range(generated.shape[0]):
                    filled = fill_part(tofill, generated, data_content_part['label'])
                    tofill = filled
                    visuals = OrderedDict([('transfered_image_on_test', filled[b])])
                    visualizer.save_images(webpage, visuals, img_path_style[b:b + 1])

webpage.save()