from utils.dataset import dataset
from utils.common import PSNR
from model import ESPCN
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000,          help='-')
parser.add_argument("--scale",          type=int, default=2,               help='-')
parser.add_argument("--batch-size",     type=int, default=128,             help='-')
parser.add_argument("--save-every",     type=int, default=1000,            help='-')
parser.add_argument("--save-best-only", type=int, default=0,               help='-')
parser.add_argument("--ckpt-dir",       type=str, default="checkpoint/x2", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
scale = FLAG.scale
ckpt_dir = FLAG.ckpt_dir
save_every = FLAG.save_every
model_path = os.path.join(ckpt_dir, f"ESPCN-x{scale}.pt")
ckpt_path = os.path.join(ckpt_dir, f"ckpt.pt")
save_best_only = (FLAG.save_best_only == 1)


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 17
hr_crop_size = 17 * 2
if scale == 3:
    hr_crop_size = 17 * 3
elif scale == 4:
    hr_crop_size = 17 * scale

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    espcn = ESPCN(scale, device)
    espcn.setup(optimizer=torch.optim.Adam(espcn.model.parameters(), lr=2e-4),
                loss=torch.nn.MSELoss(), model_path=model_path,
                ckpt_path=ckpt_path, metric=PSNR)

    espcn.load_checkpoint(ckpt_path)
    espcn.train(train_set, valid_set, 
                steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, 
                save_every=save_every)

if __name__ == "__main__":
    main()
