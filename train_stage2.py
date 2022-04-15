import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data.data_loader as data_loader
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.trainer import Trainer

opt = TrainOptions().parse()
opt.stage = '2'

print(' '.join(sys.argv))

dataloader = data_loader.create_dataloader(opt)

trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))

visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        trainer.run_discriminator_one_step(data_i)

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
