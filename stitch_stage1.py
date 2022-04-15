import os 
import torch
from collections import OrderedDict
import data.data_loader as data_loader
from options.test_options import TestOptions
from models.ADGANPP import Model
from util.visualizer import Visualizer
from util import html
from data.data_loader import get_params,get_transform
import data.data_loader as data_loader

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
    label  = torch.unsqueeze(label, 1)
    data['label'] = label
    label = label.repeat(1, image.size(1), 1, 1)
    data['image'] = image.mul(label)

    return data

def fill_part(tofill, fill, fill_label):
    if tofill is None:
        return fill

    tofill = tofill.convert('RGB')
    params = get_params(opt, tofill.size)
    transform_image = get_transform(opt, params)
    tofill = transform_image(tofill)
    tofill = torch.unsqueeze(tofill, 0).cuda(non_blocking=True)
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

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
            break
    print('process batch... ', i)
    for part_idx in range(1, opt.label_nc):
        opt.part_idx = part_idx
        data_i_part = select_part(data_i, part_idx)   
        generated = model(data_i_part, mode='stage1_inference')

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            tofill = visualizer.load_image(webpage, 'synthesized_image_on_training', img_path[b:b + 1])
            filled = fill_part(tofill, generated, data_i_part['label'])
            visuals = OrderedDict([('synthesized_image_on_training', filled[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()