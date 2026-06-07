"""
PyTorch Dataset for In-Context Learning
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import random


class InContextDiseaseDataset(Dataset):
    """
    Dataset for in-context disease prediction

    每个sample包含:
        - context: [(protein_features, disease_id, label), ...]
        - query: (protein_features, disease_id)
        - target: label
    """

    def __init__(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        disease_to_id: Dict[str, int],
        context_size: int = 8,
        sampling_strategy: str = 'mixed',
        is_training: bool = True
    ):
        """
        Args:
            data_dict: Dict mapping disease_code -> (X, y)
                X: [n_samples, n_features]
                y: [n_samples]
            disease_to_id: Dict mapping disease_code -> integer ID
            context_size: Number of examples in context
            sampling_strategy: 'mixed' or 'single'
                - 'mixed': Context can contain multiple diseases
                - 'single': Context contains only same disease as query
            is_training: Training mode (shuffle samples)
        """
        self.data_dict = data_dict
        self.disease_to_id = disease_to_id
        self.context_size = context_size
        self.sampling_strategy = sampling_strategy
        self.is_training = is_training

        # Flatten all samples
        self.all_samples = []
        for disease_code, (X, y) in data_dict.items():
            disease_id = disease_to_id[disease_code]
            for i in range(len(X)):
                self.all_samples.append({
                    'proteins': X[i],
                    'disease_code': disease_code,
                    'disease_id': disease_id,
                    'label': y[i]
                })

        # Precompute disease -> indices mapping for O(1) context sampling
        self.disease_to_indices = {}
        for i, s in enumerate(self.all_samples):
            code = s['disease_code']
            if code not in self.disease_to_indices:
                self.disease_to_indices[code] = []
            self.disease_to_indices[code].append(i)

        print(f"   Dataset created: {len(self.all_samples)} samples")
        print(f"   Diseases: {list(data_dict.keys())}")
        print(f"   Context size: {context_size}, Strategy: {sampling_strategy}")

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one training example

        Returns:
            Dict with keys:
                - context_proteins: [context_size, n_features]
                - context_disease_ids: [context_size]
                - context_labels: [context_size]
                - query_proteins: [n_features]
                - query_disease_id: scalar
                - target_label: scalar
        """
        # Query sample
        query_sample = self.all_samples[idx]

        # Sample context
        if self.sampling_strategy == 'single':
            # Context from same disease
            context_samples = self._sample_same_disease_context(
                query_sample['disease_code'],
                exclude_idx=idx
            )
        else:
            # Context from mixed diseases
            context_samples = self._sample_mixed_context(exclude_idx=idx)

        # Convert to tensors
        context_proteins = torch.tensor(
            np.stack([s['proteins'] for s in context_samples]),
            dtype=torch.float32
        )
        context_disease_ids = torch.tensor(
            [s['disease_id'] for s in context_samples],
            dtype=torch.long
        )
        context_labels = torch.tensor(
            [s['label'] for s in context_samples],
            dtype=torch.float32
        )

        query_proteins = torch.tensor(
            query_sample['proteins'],
            dtype=torch.float32
        )
        query_disease_id = torch.tensor(
            query_sample['disease_id'],
            dtype=torch.long
        )
        target_label = torch.tensor(
            query_sample['label'],
            dtype=torch.float32
        )

        return {
            'context_proteins': context_proteins,
            'context_disease_ids': context_disease_ids,
            'context_labels': context_labels,
            'query_proteins': query_proteins,
            'query_disease_id': query_disease_id,
            'target_label': target_label,
            'query_disease_code': query_sample['disease_code']  # For logging
        }

    def _sample_same_disease_context(
        self,
        disease_code: str,
        exclude_idx: int
    ) -> List[Dict]:
        """Sample context from same disease (O(k) instead of O(N))"""
        indices = [i for i in self.disease_to_indices[disease_code] if i != exclude_idx]
        k = self.context_size
        if len(indices) < k:
            chosen = random.choices(indices, k=k)
        else:
            chosen = random.sample(indices, k)
        return [self.all_samples[i] for i in chosen]

    def _sample_mixed_context(self, exclude_idx: int) -> List[Dict]:
        """Sample context from all diseases (O(k) instead of O(N))"""
        n = len(self.all_samples)
        # Sample from range(n-1) then shift indices >= exclude_idx up by 1
        chosen = random.sample(range(n - 1), self.context_size)
        chosen = [i if i < exclude_idx else i + 1 for i in chosen]
        return [self.all_samples[i] for i in chosen]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    return {
        'context_proteins': torch.stack([b['context_proteins'] for b in batch]),
        'context_disease_ids': torch.stack([b['context_disease_ids'] for b in batch]),
        'context_labels': torch.stack([b['context_labels'] for b in batch]),
        'query_proteins': torch.stack([b['query_proteins'] for b in batch]),
        'query_disease_id': torch.stack([b['query_disease_id'] for b in batch]),
        'target_label': torch.stack([b['target_label'] for b in batch]),
        'query_disease_codes': [b['query_disease_code'] for b in batch]
    }


def compute_class_weights(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """
    Compute class weights for handling imbalance

    Args:
        data_dict: Dict mapping disease_code -> (X, y)

    Returns:
        Dict mapping disease_code -> positive_weight
    """
    weights = {}

    for disease_code, (X, y) in data_dict.items():
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()

        if n_pos > 0:
            # Weight for positive class = n_neg / n_pos
            pos_weight = np.log1p(n_neg / n_pos)
        else:
            pos_weight = 1.0

        weights[disease_code] = pos_weight

    return weights


def main():
    """Test dataset"""
    # Dummy data
    np.random.seed(42)

    data_dict = {
        'I10': (np.random.randn(100, 800), np.random.randint(0, 2, 100)),
        'I11': (np.random.randn(50, 800), np.random.randint(0, 2, 50)),
    }

    disease_to_id = {'I10': 0, 'I11': 1}

    dataset = InContextDiseaseDataset(
        data_dict=data_dict,
        disease_to_id=disease_to_id,
        context_size=8,
        sampling_strategy='mixed'
    )

    # Test __getitem__
    sample = dataset[0]
    print("\nSample shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
