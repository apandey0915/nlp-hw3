import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.trainer import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    return

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if getattr(args, "finetune", False):
        model = T5ForConditionalGeneration.from_pretrained("t5-small")  # :contentReference[oaicite:1]{index=1}
    else:
        cfg = T5Config.from_pretrained("t5-small")
        model = T5ForConditionalGeneration(cfg)
    model.to(DEVICE)
    return model

def mkdir(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    tag = "best" if best else "last"
    out = os.path.join(checkpoint_dir, tag)
    mkdir(out)
    model.save_pretrained(out)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = "ft" if getattr(args, "finetune", False) else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    tag = "best" if best else "last"
    ckpt = os.path.join(checkpoint_dir, tag)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    model.to(DEVICE)
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [n for n in decay_parameters if "bias" not in n]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length: int):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

