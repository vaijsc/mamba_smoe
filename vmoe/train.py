import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time
import torch
import torch.nn as nn
import math
import datetime
import time
from torch.utils.data import DataLoader
from vmoe.config import PARAMS_CONFIG
from vmoe.data import get_train_val_data_imagenet1k
# from vmoe.models import TransformerSeq
from vmoe.models import TransformerVision
from vmoe.trainer import train_iteration, full_eval
import datetime
import wandb
import os
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    freeze_gate_weight,
    Logger,
    set_freq_optimal_search,
)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--hid-sz', type=int, default=768, help="Hidden size")
    # Add other arguments as needed
    return parser.parse_args()

def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    args = parse_args()
    wandb_flag = wandb_params["wandb_flag"]
    if wandb_flag:
        wandb.init(project=wandb_params["project_name"])
        wandb.run.name = wandb_params["job_name"]
        wandb.config.update(model_params)
    # global val
    best_val_loss = None
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]
    resume = trainer_params["resume"]

    if distributed == False or env_params["rank"] == 0:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)
    
    train_loader, val_loader = get_train_val_data_imagenet1k(data_params, 
                                              env_params=env_params,
                                              batch_size=trainer_params['batch_size'],
                                              device=device)

    import ipdb; ipdb.set_trace()
    # Model setup
    model = TransformerVision(
        hidden_size=model_params['hidden_size'],
        nb_heads=model_params['nb_heads'],
        num_classes=1000,  # ImageNet has 1000 classes
        **model_params
    )

    print(model)

    # Distribute the model if necessary (DistributedDataParallel)
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # Optimizer and scheduler setup
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # Logger setup
    logger = Logger()
    folder_path = '/home/ubuntu/workspace/MomentumSMoE/result/log'
    logging = create_exp_dir(f"{folder_path}")
    fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
    logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")

    logging(f"Training Parameters:\n {trainer_params}")
    logging(f"Models Parameters:\n {model_params}")
    logging(str(datetime.datetime.now()))
    logging(str(model))
    logging(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    logging(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Resume from checkpoint if applicable
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume,
    )

    # Training loop
    start_time = time.time()
    best_val_loss = None

    for iter_no in range(0, trainer_params["nb_iter"]):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                val_loss += loss.item()

        # Average loss per epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging(f"Epoch {iter_no}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save model if validation loss is the best so far
        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                trainer_params["checkpoint_path"],
                iter_no,
                model,
                optimizer,
                scheduler,
                logger,
            )

        # Logging metrics (e.g., using wandb)
        if wandb_flag:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': iter_no
            })

    # Final logging
    end_time = time.time()
    logging(f"Training completed in {(end_time - start_time)/3600:.2f} hours.")


if __name__ == "__main__":
    launch(**get_params(params_config=PARAMS_CONFIG))
