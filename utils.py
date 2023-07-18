import os
import re
import fnmatch
import torch
import logging
import shutil
import matplotlib.pyplot as plt
from matplotlib import animation


def make_dir(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)


def tensor_to_np(x):
    return x.data.cpu().numpy()


class Logger(object):
    """docstring for Logger."""

    def __init__(self, args):
        super(Logger, self).__init__()
        self.log_file = args.log_file
        _LOG_FORMAT = "%(asctime)s - %(levelname)s"
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format=_LOG_FORMAT)
    def log(self,msg):
        logging.info(msg)



def save_checkpoint(state, is_best, step, args, name=''):
    if(step % args.save_model_every_n_steps == 0):
        print("=> saving checkpoint '{}'".format(step))
        torch.save(state, os.path.join(args.save_dir, name + 'checkpoint_%03d.pth.tar' % step))
    if is_best:
        print("=> saving best checkpoint '{}'".format(step))
        torch.save(state, os.path.join(args.save_dir, name + 'model_best_epochs.pth.tar'))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def display_frames_as_gif(frames,dir='./test.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=5)
    anim.save(dir, writer='imagemagick', fps=5)