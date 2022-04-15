import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data.data_loader as data_loader
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.trainer import Trainer
import torch

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
    label  = label[:, part_idx:part_idx+1, :, :]

    new_image = None
    new_label = None
    for i in range(label.shape[0]):
        if torch.sum(label[i,:,:,:]) != 0:
            if  new_image is None:
                new_image = image[i:i+1,:,:,:]
                new_label = label[i:i+1,:,:,:]
            else:
                new_image = torch.cat([new_image, image[i:i+1,:,:,:]])
                new_label = torch.cat([new_label, label[i:i+1,:,:,:]])

    if new_label is None:
        return None

    while new_image.shape[0] < label.shape[0] /2:
        new_image = torch.cat([new_image, new_image], 0)
        new_label = torch.cat([new_label, new_label], 0)

    data['label'] = new_label
    data['image'] = new_image.mul(new_label.repeat(1, image.size(1), 1, 1))

    return data

opt = TrainOptions().parse()
print(' '.join(sys.argv))

dataloader = data_loader.create_dataloader(opt)

trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))

visualizer = Visualizer(opt) 

            
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)

    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        for part_idx in range(1,opt.label_nc):
            opt.part_idx = part_idx

            data_i_part = select_part(data_i, part_idx)
            if data_i_part is None:
                continue

            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i_part)

            trainer.run_discriminator_one_step(data_i_part)

            if i % 256 == 0 and part_idx <= 10:
                visuals = OrderedDict([('synthesized_image_part'+str(part_idx), trainer.get_latest_generated()),
                                    ('real_image_part'+str(part_idx), data_i_part['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
        
        iter_counter.record_one_iteration()
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    print("\n epoch:", epoch, "losses:", trainer.get_latest_losses())
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
    epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')