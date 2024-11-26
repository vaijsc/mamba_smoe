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
from utils.datasets import imagenet_lmdb_dataset
from tensorboard import TensorboardLogger

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


def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus = _build_corpus(**data_params, env_params=env_params)
    data_params["vocab_size"] = corpus.vocab_size
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size
    )

    if env_params["distributed"]:
        # split the data into equal parts
        assert batch_size % env_params["world_size"] == 0
        device_batch_size = batch_size // env_params["world_size"]
        slice_data = slice(
            device_batch_size * env_params["rank"],
            device_batch_size * (env_params["rank"] + 1),
        )
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    """
    ipdb> train_data.shape
    torch.Size([16, 6451688])
    ipdb> val_data.shape
    torch.Size([16, 13602])
    ipdb> test_data.shape
    torch.Size([16, 15348])
    """
    # import ipdb; ipdb.set_trace()
    return train_data, val_data, test_data

def get_vision_data(dataset, data_dir, batch_size):
    dataset_train = create_dataset(
            dataset, root=data_dir + os.path.join('/train'), 
            is_training=True, batch_size=batch_size)

    dataset_eval = create_dataset(
            dataset, root=data_dir + os.path.join('/val'), 
            is_training=False, batch_size=batch_size)
    loader_train = create_loader(
        dataset_train,
        input_size=[3,224,224],
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        re_split=False,
        scale=[0.08, 1.0],
        ratio=[0.75, 4/3],
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        num_aug_repeats=0,
        num_aug_splits=0,
        # interpolation=train_interpolation,
        mean=(0.485, 0.456, 0.406),
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )
    loader_eval = create_loader(
        dataset_eval,
        input_size=[3,224,224],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    