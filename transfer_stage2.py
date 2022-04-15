import os
from collections import OrderedDict
import data.data_loader as data_loader
from options.test_options import TestOptions
from models.ADGANPP import Model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.status = 'test'
opt.stage = '2'

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

    if i % 2 == 0:
        data_content = data_i
    else:
        data_i['label'] = data_content['label']
        generated = model(data_i, mode='stage2_inference')
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('transfered_image_on_test', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()