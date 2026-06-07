
#!/usr/bin/env python -u

import sys

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)



import pickle

import torch

import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score



sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))



from models.transformer_gpt import InContextTransformerGPT

from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print("Sanity Check: O Model on Training Diseases")



checkpoint = torch.load('checkpoints_o_model/best_model.pt', map_location=device)

model = InContextTransformerGPT(

    protein_dim=2941,

    hidden_dim=768,

    n_layers=12,

    n_heads=12,

    dropout=0.2,

    temperature=1.0

)

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

model.eval()



with open('processed_data_o/train_data.pkl', 'rb') as f:

    train_data = pickle.load(f)



for disease in ['O80', 'O70']:

    print(f"\n{disease}")

    

    X, y = train_data[disease]

    print(f"Total: {len(y)}, Positives: {int(y.sum())}")

    

    dataset = InContextDiseaseDatasetNoID(

        data_dict={disease: (X, y)},

        context_size=64,

        is_training=False

    )

    

    loader = DataLoader(dataset, batch_size=8, shuffle=False, 

                       collate_fn=collate_fn_no_id, num_workers=0)

    

    all_probs = []

    all_targets = []

    

    with torch.no_grad():

        for i, batch in enumerate(loader):

            if i % 1000 == 0:

                print(f"  Batch {i}/{len(loader)}")

            

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 

                    for k, v in batch.items()}

            

            logits = model(batch)

            probs = torch.sigmoid(logits).cpu().numpy()

            

            all_probs.extend(probs)

            all_targets.extend(batch['target_label'].cpu().numpy())

    

    all_probs = np.array(all_probs)

    all_targets = np.array(all_targets)

    

    auroc = roc_auc_score(all_targets, all_probs)

    print(f"  AUROC: {auroc:.4f}")

    print(f"  Prediction min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}")

    print(f"  Positives predicted: {(all_probs > 0.5).sum()}")

