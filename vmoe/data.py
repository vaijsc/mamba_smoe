import os, sys
import argparse
import math, random
import torch
import tqdm
from timm.data import ImageDataset, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy,\
    LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler
from scheduler.scheduler_factory import create_scheduler
import shutil
# from utils.datasets import imagenet_lmdb_dataset
from tensorboard import TensorboardLogger
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms
import os

def loader_to_tensor_batches(loader, device):
    """
    Convert a DataLoader into batches of tensors for data and labels.
    Args:
        loader (DataLoader): The data loader to process.
        device (torch.device): Device to move the tensors to (e.g., "cuda" or "cpu").
    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: List of (data_batch, label_batch) tensors.
    """
    batched_data = []

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            # Move to device and store batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batched_data.append((batch_data, batch_labels))

    return batched_data


def get_train_val_data_imagenet1k(data_params, env_params, batch_size, device):
    """
    Configure ImageNet1k data with preprocessing and loaders, returning data in batches.
    Args:
        data_params (dict): Configuration for data paths and preprocessing.
        env_params (dict): Configuration for distributed training.
        batch_size (int): Batch size for data loaders.
        device (torch.device): Device to use (e.g., "cuda" or "cpu").
    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        Batched training and validation data and labels.
    """
    # Data preprocessing configuration
    data_config = {
        "input_size": data_params.get("input_size", (3, 224, 224)),
        "mean": data_params.get("mean", [0.485, 0.456, 0.406]),
        "std": data_params.get("std", [0.229, 0.224, 0.225]),
        "crop_pct": data_params.get("crop_pct", 0.875),
    }

    # Preprocessing transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(data_config["input_size"][1], scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(data_config["input_size"][1] / data_config["crop_pct"])),
        transforms.CenterCrop(data_config["input_size"][1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    # Dataset paths
    train_root = os.path.join(data_params["data_dir"], "train")
    val_root = os.path.join(data_params["data_dir"], "val")

    # Create datasets
    train_dataset = create_dataset(
        name=data_params["data_name"],
        root=train_root,
        split="train",
        is_training=True,
        transform=train_transforms,
    )
    val_dataset = create_dataset(
        name=data_params["data_name"],
        root=val_root,
        split="val",
        is_training=False,
        transform=val_transforms,
    )

    # Configure samplers
    if env_params["distributed"]:
        world_size = env_params["world_size"]
        rank = env_params["rank"]
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # Data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=data_params["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=data_params["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader
