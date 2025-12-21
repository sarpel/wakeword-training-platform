"""
Balanced Batch Sampler for controlling class ratios in batches
Ensures each batch has specified proportions of positive, negative, and hard negative samples
"""

from typing import Iterator, List

import numpy as np
import structlog
from torch.utils.data import Sampler

logger = structlog.get_logger(__name__)


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that maintains fixed ratio of pos:neg:hard_neg in each batch

    Ensures balanced representation for effective training with hard negative mining.
    """

    def __init__(
        self,
        idx_pos: List[int],
        idx_neg: List[int],
        idx_hard_neg: List[int],
        batch_size: int,
        ratio: tuple = (1, 1, 1),
        drop_last: bool = True,
    ):
        """
        Initialize balanced batch sampler

        Args:
            idx_pos: Indices of positive samples
            idx_neg: Indices of negative samples
            idx_hard_neg: Indices of hard negative samples
            batch_size: Total batch size
            ratio: Ratio tuple (pos, neg, hard_neg) - default (1, 1, 1)
            drop_last: Drop incomplete batches
        """
        self.idx_pos = np.array(idx_pos)
        self.idx_neg = np.array(idx_neg)
        self.idx_hard_neg = np.array(idx_hard_neg)

        self.batch_size = batch_size
        self.ratio = ratio
        self.drop_last = drop_last

        # Calculate samples per class in each batch
        ratio_sum = sum(ratio)
        self.n_pos = int(batch_size * ratio[0] / ratio_sum)
        self.n_neg = int(batch_size * ratio[1] / ratio_sum)
        self.n_hard_neg = batch_size - self.n_pos - self.n_neg

        # Verify we have enough samples
        if len(self.idx_pos) < self.n_pos:
            logger.warning(
                f"Not enough positive samples ({len(self.idx_pos)}) " f"for batch requirement ({self.n_pos})"
            )

        if len(self.idx_neg) < self.n_neg:
            logger.warning(
                f"Not enough negative samples ({len(self.idx_neg)}) " f"for batch requirement ({self.n_neg})"
            )

        if len(self.idx_hard_neg) < self.n_hard_neg:
            logger.warning(
                f"Not enough hard negative samples ({len(self.idx_hard_neg)}) "
                f"for batch requirement ({self.n_hard_neg})"
            )

        logger.info(
            f"BalancedBatchSampler initialized: "
            f"batch_size={batch_size}, ratio={ratio}, "
            f"samples_per_batch=[pos={self.n_pos}, neg={self.n_neg}, hard_neg={self.n_hard_neg}]"
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches"""
        # Shuffle indices for each class
        pos_perm = np.random.permutation(self.idx_pos)
        neg_perm = np.random.permutation(self.idx_neg)
        hard_neg_perm = np.random.permutation(self.idx_hard_neg)

        # Determine number of batches
        max_batches_pos = len(pos_perm) // self.n_pos if self.n_pos > 0 else float("inf")
        max_batches_neg = len(neg_perm) // self.n_neg if self.n_neg > 0 else float("inf")
        max_batches_hard = len(hard_neg_perm) // self.n_hard_neg if self.n_hard_neg > 0 else float("inf")

        num_batches = int(min(max_batches_pos, max_batches_neg, max_batches_hard))

        # Generate batches
        for i in range(num_batches):
            batch_indices = []

            # Add positive samples
            if self.n_pos > 0:
                start = i * self.n_pos
                end = start + self.n_pos
                batch_indices.extend(pos_perm[start:end].tolist())

            # Add negative samples
            if self.n_neg > 0:
                start = i * self.n_neg
                end = start + self.n_neg
                batch_indices.extend(neg_perm[start:end].tolist())

            # Add hard negative samples
            if self.n_hard_neg > 0:
                start = i * self.n_hard_neg
                end = start + self.n_hard_neg
                batch_indices.extend(hard_neg_perm[start:end].tolist())

            # Shuffle within batch (optional, but helps with GPU parallelism)
            np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self) -> int:
        """Return number of batches"""
        max_batches_pos = len(self.idx_pos) // self.n_pos if self.n_pos > 0 else float("inf")
        max_batches_neg = len(self.idx_neg) // self.n_neg if self.n_neg > 0 else float("inf")
        max_batches_hard = len(self.idx_hard_neg) // self.n_hard_neg if self.n_hard_neg > 0 else float("inf")

        return int(min(max_batches_pos, max_batches_neg, max_batches_hard))


from torch.utils.data import Dataset


def create_balanced_sampler_from_dataset(
    dataset: Dataset, batch_size: int, ratio: tuple = (1, 1, 1), drop_last: bool = True
) -> BalancedBatchSampler:
    """
    Create balanced batch sampler from dataset

    Args:
        dataset: PyTorch dataset with category information
        batch_size: Batch size
        ratio: Ratio (pos, neg, hard_neg)
        drop_last: Drop incomplete batches

    Returns:
        BalancedBatchSampler
    """
    # Collect indices by category
    idx_pos = []
    idx_neg = []
    idx_hard_neg = []

    # Optimized: Access metadata directly from dataset.files if available
    if hasattr(dataset, "files"):
        for i, file_info in enumerate(dataset.files):
            category = file_info.get("category", "")

            if category == "positive":
                idx_pos.append(i)
            elif category == "negative":
                idx_neg.append(i)
            elif category == "hard_negative":
                idx_hard_neg.append(i)
    else:
        # Fallback to iteration
        logger.warning("Dataset does not expose .files, falling back to slow iteration")
        for i in range(len(dataset)):  # type: ignore[arg-type]
            # Try to get metadata without loading full item if possible
            # But dataset[i] usually loads audio.
            # We have to rely on dataset[i] returning (audio, label, metadata)
            try:
                _, _, metadata = dataset[i]
                category = metadata.get("category", "")

                if category == "positive":
                    idx_pos.append(i)
                elif category == "negative":
                    idx_neg.append(i)
                elif category == "hard_negative":
                    idx_hard_neg.append(i)
            except Exception as e:
                logger.warning(f"Failed to get metadata for index {i}: {e}")

    logger.info(f"Collected indices: pos={len(idx_pos)}, " f"neg={len(idx_neg)}, hard_neg={len(idx_hard_neg)}")

    return BalancedBatchSampler(
        idx_pos=idx_pos,
        idx_neg=idx_neg,
        idx_hard_neg=idx_hard_neg,
        batch_size=batch_size,
        ratio=ratio,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Test balanced sampler
    print("BalancedBatchSampler Test")
    print("=" * 60)

    # Create dummy indices
    idx_pos = list(range(0, 100))  # 100 positive samples
    idx_neg = list(range(100, 300))  # 200 negative samples
    idx_hard_neg = list(range(300, 400))  # 100 hard negative samples

    print(f"Test dataset:")
    print(f"  Positive: {len(idx_pos)}")
    print(f"  Negative: {len(idx_neg)}")
    print(f"  Hard Negative: {len(idx_hard_neg)}")

    # Create sampler with 1:1:1 ratio
    batch_size = 24
    sampler = BalancedBatchSampler(
        idx_pos=idx_pos,
        idx_neg=idx_neg,
        idx_hard_neg=idx_hard_neg,
        batch_size=batch_size,
        ratio=(1, 1, 1),
    )

    print(f"\nSampler created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Ratio: (1, 1, 1)")
    print(f"  Samples per batch: pos={sampler.n_pos}, neg={sampler.n_neg}, hard_neg={sampler.n_hard_neg}")
    print(f"  Total batches: {len(sampler)}")

    # Generate a few batches
    print(f"\nGenerating batches:")
    for i, batch_idx in enumerate(sampler):
        if i >= 3:
            break

        # Count samples from each category
        n_pos = sum(1 for idx in batch_idx if idx < 100)
        n_neg = sum(1 for idx in batch_idx if 100 <= idx < 300)
        n_hard = sum(1 for idx in batch_idx if idx >= 300)

        print(f"  Batch {i+1}: size={len(batch_idx)}, pos={n_pos}, neg={n_neg}, hard_neg={n_hard}")

    # Test with different ratio (1:2:1)
    print(f"\nTesting with ratio (1:2:1):")
    sampler2 = BalancedBatchSampler(
        idx_pos=idx_pos,
        idx_neg=idx_neg,
        idx_hard_neg=idx_hard_neg,
        batch_size=batch_size,
        ratio=(1, 2, 1),
    )

    print(f"  Samples per batch: pos={sampler2.n_pos}, neg={sampler2.n_neg}, hard_neg={sampler2.n_hard_neg}")
    print(f"  Total batches: {len(sampler2)}")

    for i, batch_idx in enumerate(sampler2):
        if i >= 2:
            break

        n_pos = sum(1 for idx in batch_idx if idx < 100)
        n_neg = sum(1 for idx in batch_idx if 100 <= idx < 300)
        n_hard = sum(1 for idx in batch_idx if idx >= 300)

        print(f"  Batch {i+1}: size={len(batch_idx)}, pos={n_pos}, neg={n_neg}, hard_neg={n_hard}")

    print("\n✅ BalancedBatchSampler test complete")
