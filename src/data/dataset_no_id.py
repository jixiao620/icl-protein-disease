"""
PyTorch Dataset - No Disease ID Version
Context和Query来自同一个disease
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import random


class InContextDiseaseDatasetNoID(Dataset):

    def __init__(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        context_size: int = 8,
        is_training: bool = True,
        context_pos_ratio: Optional[float] = None,
        query_pos_ratio: Optional[float] = None,
    ):
        """
        context_pos_ratio: fraction of context slots filled with positive examples.
                           None = legacy behaviour (3x prevalence, capped at 0.3).
                           e.g. 0.5 means half the context is always positive.
        query_pos_ratio:   target fraction of positive queries drawn during training.
                           None = original distribution (no oversampling).
                           e.g. 0.5 means queries are 50% positive via oversampling.
                           Only applied when is_training=True.
        """
        self.data_dict = data_dict
        self.context_size = context_size
        self.is_training = is_training
        self.context_pos_ratio = context_pos_ratio
        self.query_pos_ratio = query_pos_ratio

        self.samples_by_disease = {}
        for disease_code, (X, y) in data_dict.items():
            samples = []
            for i in range(len(X)):
                samples.append({
                    'proteins': X[i],
                    'label': y[i],
                    'disease_code': disease_code
                })
            self.samples_by_disease[disease_code] = samples

        # Build query pool: oversample positives if query_pos_ratio is set
        all_samples = []
        for disease_code, samples in self.samples_by_disease.items():
            all_samples.extend(samples)

        if is_training and query_pos_ratio is not None:
            positives = [s for s in all_samples if s['label'] == 1]
            negatives = [s for s in all_samples if s['label'] == 0]
            n_neg = len(negatives)
            # how many positives needed so positives/(positives+negatives) = query_pos_ratio
            n_pos_target = int(n_neg * query_pos_ratio / max(1 - query_pos_ratio, 1e-6))
            n_pos_target = min(n_pos_target, n_neg * 10)  # safety cap: at most 10x negatives
            oversampled_pos = random.choices(positives, k=n_pos_target)
            self.all_samples = negatives + oversampled_pos
            random.shuffle(self.all_samples)
            actual_ratio = n_pos_target / max(len(self.all_samples), 1)
            print(f"   Query oversampling: {len(positives)} -> {n_pos_target} positives "
                  f"({actual_ratio*100:.1f}% of {len(self.all_samples)} total)")
        else:
            self.all_samples = all_samples

        print(f"   Dataset created: {len(self.all_samples)} samples")
        print(f"   Diseases: {list(data_dict.keys())}")
        print(f"   Context size: {context_size}, Single-disease context")
        print(f"   context_pos_ratio: {context_pos_ratio}  query_pos_ratio: {query_pos_ratio}")

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_sample = self.all_samples[idx]
        query_disease = query_sample['disease_code']

        disease_samples = self.samples_by_disease[query_disease]

        # Exclude query by object identity (O(N) not O(N*D) array comparison)
        query_id = id(query_sample)
        available_samples = [s for s in disease_samples if id(s) != query_id]

        positive_samples = [s for s in available_samples if s['label'] == 1]
        negative_samples = [s for s in available_samples if s['label'] == 0]

        # Context positive ratio
        if self.context_pos_ratio is not None:
            target_ratio = float(self.context_pos_ratio)
        else:
            # Legacy: 3x prevalence, capped at 0.3
            prevalence = len(positive_samples) / max(len(available_samples), 1)
            target_ratio = min(prevalence * 3, 0.3)

        n_pos_needed = int(self.context_size * target_ratio)
        n_neg_needed = self.context_size - n_pos_needed

        if len(positive_samples) >= n_pos_needed and len(negative_samples) >= n_neg_needed:
            context_samples = (
                random.sample(positive_samples, n_pos_needed) +
                random.sample(negative_samples, n_neg_needed)
            )
        elif len(available_samples) < self.context_size:
            context_samples = random.choices(available_samples, k=self.context_size)
        else:
            context_samples = random.sample(available_samples, self.context_size)

        context_proteins = torch.tensor(
            np.stack([s['proteins'] for s in context_samples]),
            dtype=torch.float32
        )
        context_labels = torch.tensor(
            [s['label'] for s in context_samples],
            dtype=torch.float32
        )

        query_proteins = torch.tensor(
            query_sample['proteins'],
            dtype=torch.float32
        )
        target_label = torch.tensor(
            query_sample['label'],
            dtype=torch.float32
        )

        return {
            'context_proteins': context_proteins,
            'context_labels': context_labels,
            'query_proteins': query_proteins,
            'target_label': target_label,
            'query_disease_code': query_sample['disease_code']
        }


def collate_fn_no_id(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        'context_proteins': torch.stack([b['context_proteins'] for b in batch]),
        'context_labels': torch.stack([b['context_labels'] for b in batch]),
        'query_proteins': torch.stack([b['query_proteins'] for b in batch]),
        'target_label': torch.stack([b['target_label'] for b in batch]),
        'query_disease_codes': [b['query_disease_code'] for b in batch]
    }
