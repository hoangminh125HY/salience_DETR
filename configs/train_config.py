from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = "/kaggle/working/outputs"
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "/kaggle/input/ppever2/PPE_v2zip"
train_transform = presets.detr  # see transforms/presets to choose a transform
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train",
    ann_file=f"{coco_path}/annotations/train_annotations.coco.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/valid",
    ann_file=f"{coco_path}/annotations/val_annotations.coco.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "/kaggle/working/salience_DETR/configs/salience_detr/salience_detr_resnet50_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune
resume_from_checkpoint = None  

learning_rate = 1e-4  # initial learning rate

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

# Optimizer (must include param_dicts)
optimizer = optim.AdamW(param_dicts, lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))

# LR Scheduler (must include optimizer)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
