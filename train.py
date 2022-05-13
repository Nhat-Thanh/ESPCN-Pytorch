from utils.dataset import dataset
from utils.common import PSNR
from model import ESPCN
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000, help='-')
parser.add_argument("--scale",          type=int, default=2,      help='-')
parser.add_argument("--batch-size",     type=int, default=128,    help='-')
parser.add_argument("--save-every",     type=int, default=1000,   help='-')
parser.add_argument("--save-best-only", type=int, default=0,      help='-')
parser.add_argument("--save-log",       type=int, default=0,      help='-')
parser.add_argument("--ckpt-dir",       type=str, default="",     help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
steps = FLAGS.steps
batch_size = FLAGS.batch_size
save_every = FLAGS.save_every
save_best_only = (FLAGS.save_best_only == 1)
save_log = (FLAGS.save_log == 1)

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_dir = FLAGS.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/x{scale}"

model_path = os.path.join(ckpt_dir, f"ESPCN-x{scale}.pt")
ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")


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
    espcn.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()
