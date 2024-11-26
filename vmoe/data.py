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
# 0.6.12
def _tokenize(text_path, dictionary_to_update):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)

    nb_tokens_in_dictionary = len(dictionary_to_update)

    # Count nb of tokens in text and update the dictionary
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1

    # Assign to each token its identifier
    ids = []
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids


class Corpus:
    def __init__(self, data_path):
        self._dictionary = {}
        self.train = _tokenize(
            text_path=os.path.join(data_path, "train.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.valid = _tokenize(
            text_path=os.path.join(data_path, "valid.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.test = _tokenize(
            text_path=os.path.join(data_path, "test.txt"),
            dictionary_to_update=self._dictionary,
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    # import ipdb; ipdb.set_trace()
    nb_batches = data_tensor.size(0) // batch_size
    # batch_size 16
    # data_tensor.size(0) 103227021
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    """
    ipdb> data_tensor.narrow(0, 0, nb_batches * batch_size).shape
    torch.Size([103227008])
    """
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, env_params, data_name=None):
    # save the corpus to a file so that it's faster next time
    corpus_path = os.path.join(data_path, "corpus.pt")
    if os.path.exists(corpus_path):
        print("Loading an existing corpus file from {}".format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print("Creating a corpus file at {}".format(corpus_path))
        if env_params["distributed"]:
            # only one process need to create a corpus file
            if env_params["rank"] == 0:
                corpus = Corpus(data_path)
                torch.save(corpus, corpus_path)
                # sync with other processes
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                print("Waiting rank0 to create a corpus file.")
                # sync with rank0
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = torch.load(corpus_path)
        else:
            corpus = Corpus(data_path)
            torch.save(corpus, corpus_path)
    return corpus


def _get_train_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size),
    ]



import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist

class ImageNetProcessor:
    def __init__(self, data_path, input_size=224):
        self.data_path = data_path
        self.input_size = input_size
        self._prepare_transforms()
        self.train_dataset = self._load_dataset(split="train")
        self.val_dataset = self._load_dataset(split="val")

    def _prepare_transforms(self):
        # Standard ImageNet preprocessing transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_dataset(self, split):
        dataset_path = os.path.join(self.data_path, split)
        transform = self.train_transform if split == "train" else self.val_transform
        return datasets.ImageFolder(root=dataset_path, transform=transform)

def get_train_val_data(data_path, batch_size, env_params, device, input_size=224):
    processor = ImageNetProcessor(data_path=data_path, input_size=input_size)

    # Data loaders
    train_loader = DataLoader(
        processor.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=env_params.get("num_workers", 16),
        pin_memory=False
    )
    val_loader = DataLoader(
        processor.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=env_params.get("num_workers", 16),
        pin_memory=False
    )

    # Optionally distribute data
    if env_params["distributed"]:
        train_loader = _split_data_loader(train_loader, env_params)
        val_loader = _split_data_loader(val_loader, env_params)

    # Move data to the device
    train_data = _move_to_device(train_loader, device)
    val_data = _move_to_device(val_loader, device)

    return train_data, val_data

def _split_data_loader(data_loader, env_params):
    # Slice dataset for distributed training
    world_size = env_params["world_size"]
    rank = env_params["rank"]
    total_samples = len(data_loader.dataset)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    sampler = torch.utils.data.SubsetRandomSampler(range(start_idx, end_idx))
    return DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        sampler=sampler,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )

def _move_to_device(data_loader, device):
    # Moves a batch of data to the specified device
    data_batches = []
    for images, labels in data_loader:
        data_batches.append((images.to(device), labels.to(device)))
    return data_batches

















import torch
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


def get_train_val_test_data_imagenet1k(data_params, env_params, batch_size, device):
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
        num_workers=data_params.get("num_workers", 24),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=data_params.get("num_workers", 24),
        pin_memory=True,
    )
    return train_loader, val_loader
