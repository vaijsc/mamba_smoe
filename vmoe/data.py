import os, sys
import argparse
import math, random
import torch
import tqdm
from timm.data import ImageDataset, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
# from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy,\
    # LabelSmoothingCrossEntropy
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

from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import os

def get_train_val_test_data_imagenet1k(data_params, env_params, batch_size, device):
    """
    Configure ImageNet1k data with preprocessing and loaders.
    """
    data_config = {
        "input_size": data_params.get("input_size", (3, 224, 224)),
        "mean": data_params.get("mean", [0.485, 0.456, 0.406]),
        "std": data_params.get("std", [0.229, 0.224, 0.225]),
        "crop_pct": data_params.get("crop_pct", 0.875),
    }

    # Preprocessing transformations for training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(data_config["input_size"][1], scale=(0.08, 1.0)),  # Random crop
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),  # Normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(data_config["input_size"][1] / data_config["crop_pct"])),  # Resize keeping aspect ratio
        transforms.CenterCrop(data_config["input_size"][1]),  # Center crop to input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),  # Normalize
    ])

    # Define dataset paths
    train_root = os.path.join(data_params["data_dir"], "train")
    val_root = os.path.join(data_params["data_dir"], "val")

    # Create datasets with transformations applied
    train_dataset = create_dataset(
        name=data_params["data_name"],
        root=train_root,
        split="train",
        is_training=True,
        transform=train_transforms  # Apply training transforms
    )
    val_dataset = create_dataset(
        name=data_params["data_name"],
        root=val_root,
        split="val",
        is_training=False,
        transform=val_transforms  # Apply validation transforms
    )

    # Configure samplers
    if env_params["distributed"]:
        world_size = env_params["world_size"]
        rank = env_params["rank"]
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # Initialize data loaders with preprocessing
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=data_params.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=data_params.get("num_workers", 4),
        pin_memory=True,
    )
    # import ipdb; ipdb.set_trace()
    return train_loader, val_loader
