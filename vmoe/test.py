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

# Usage Example
data_params = {"data_path": "/home/ubuntu/workspace/dataset/imagenet1K/"}
env_params = {"distributed": True, "rank": 0, "world_size": 1, "num_workers": 24}
batch_size = 128
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
train_data, val_data = get_train_val_data(
    data_path=data_params["data_path"],
    batch_size=batch_size,
    env_params=env_params,
    device=device
)

print(f"Train Batches: {len(train_data)}, Validation Batches: {len(val_data)}")
