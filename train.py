from __future__ import absolute_import, division, print_function

from trainer import Trainer
from trainer_day import TrainerDay
from trainer_unpaired import TrainerUnpaired
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.unpaired:
        trainer = TrainerUnpaired(opts)
    elif opts.train_day_only:
        trainer = TrainerDay(opts)
    else:
        trainer = Trainer(opts)
    trainer.train()
