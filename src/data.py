import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from medmnist import ChestMNIST, INFO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: medmnist. Install the project requirements in the "
        "same environment as your notebook kernel. From the notebook, run "
        "`%pip install -r ../requirements.txt`; from the project root, run "
        "`python -m pip install -r requirements.txt`."
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DictChestMNIST(Dataset):
    def __init__(self, dataset, split_name, max_items=None):
        self.dataset = dataset
        if max_items is None:
            n_items = len(dataset)
        elif max_items < 0:
            raise ValueError("max_items must be None or a non-negative integer.")
        else:
            n_items = min(max_items, len(dataset))

        self.indices = list(range(n_items))
        self.split_name = split_name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        image, label = self.dataset[source_idx]

        return {
            "images": image,
            "labels": torch.as_tensor(label, dtype=torch.float32).view(-1),
            "ids": f"{self.split_name}_{source_idx}",
        }


def _class_names():
    label_info = INFO["chestmnist"]["label"]
    if isinstance(label_info, dict):
        return [label_info[key] for key in sorted(label_info, key=lambda x: int(x))]
    return list(label_info)


def get_small_data(n_train=None, n_val=None, batch_size=8, image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = ChestMNIST(
        split="train",
        transform=transform,
        download=True,
        as_rgb=True,
    )
    val_dataset = ChestMNIST(
        split="val",
        transform=transform,
        download=True,
        as_rgb=True,
    )

    train_dataset = DictChestMNIST(train_dataset, split_name="train", max_items=n_train)
    val_dataset = DictChestMNIST(val_dataset, split_name="val", max_items=n_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, _class_names()
