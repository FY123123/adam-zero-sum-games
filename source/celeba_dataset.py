"""CelebA dataset loader via HuggingFace (no Google Drive dependency).

Uses streaming + partial download to avoid excessive disk usage.
"""

import logging
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hf_xet").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from torch.utils.data import Dataset
from datasets import load_dataset


class HFCelebA(Dataset):
    """CelebA dataset downloaded from HuggingFace (flwrlabs/celeba).

    Streams data and keeps only `max_samples` in memory to avoid
    filling the disk when space is limited.
    Returns (image, 0) tuples to match torchvision dataset interface.
    """

    def __init__(self, split='train', transform=None, max_samples=50000):
        print(f"\r⏳ CelebA: Loading {max_samples} images via streaming...",
              end='', flush=True)
        stream = load_dataset(
            'flwrlabs/celeba', split=split, streaming=True)
        self.images = []
        for i, sample in enumerate(stream):
            if i >= max_samples:
                break
            if i % 5000 == 0:
                print(f"\r⏳ CelebA: Loading images... {i}/{max_samples}",
                      end='', flush=True)
            img = sample['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            self.images.append(img)
        self.transform = transform
        print(f"\r✅ CelebA: Loaded {len(self.images)} images              ",
              flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, 0
