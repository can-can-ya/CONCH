import json

import os
from pathlib import Path
root = Path('../').resolve()
os.chdir(root)

from conch.open_clip_custom import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero
from conch.downstream.wsi_datasets import WSIEmbeddingDataset

import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# display all jupyter output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# This notebook provides an example for performing zero-shot classification by ensembling multiple prompts and prompt templates for WSIs.

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/home/gjx/can_pretrained-model/conch/pytorch_model.bin'
model, _ = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path, device=device)
model.eval()

index_col = 'pathology_id' # column with the slide ids
target_col = 'subtype' # column with the target labels
patient_id = 'patient_id' # column with patient id
datasets = ['lung', 'brca', 'rcc', 'esca'] # four datasets
all_label_map = {
    'lung': {'LUAD': 0, 'LUSC': 1},
    'brca': {'IDC': 0, 'ILC': 1},
    'rcc': {'CCRCC': 0, 'PRCC': 1},
    'esca': {'ESAD': 0, 'ESCC': 1}
} # maps values in target_col to integers
dataset_label_shift = {
    'lung': 0,
    'brca': 2,
    'rcc': 4,
    'esca': 6
}

def read_datasplit_npz(path: str):
    data_npz = np.load(path, allow_pickle=True)

    pids_train = [str(s) for s in data_npz['train_patients']]
    if 'val_patients' in data_npz:
        pids_val = [str(s) for s in data_npz['val_patients']]
    else:
        pids_val = None
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

for fold in range(1, 11):
    for dataset_name in datasets:
        label_map = all_label_map[dataset_name]

        # assuming the csv has a column for slide_id (index_col) and OncoTreeCode (target_col), adjust above as needed
        df = pd.read_csv(f'/home/gjx/can_dataset/tcga_{dataset_name}/table/TCGA_{dataset_name.upper()}_path_subtype_x10_processed.csv')
        # path to the extracted embeddings, assumes the embeddings are saved as .pt files, 1 file per slide
        data_source = f'/home/gjx/can_dataset/tcga_{dataset_name}/feats-l1-s256_CONCH/'

        datasplit_path = f'/home/gjx/can_dataset/tcga_{dataset_name}/datasplit/fold_{fold}.npz'
        pids_train, pids_val, pids_test = read_datasplit_npz(datasplit_path)

        df = df[df[patient_id].isin(pids_test)].reset_index(drop=True)

        dataset = WSIEmbeddingDataset(data_source = data_source,
                                      df=df,
                                      index_col=index_col,
                                      target_col=target_col,
                                      label_map=label_map)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4)

        idx_to_class = {0: 'LUAD', 1: 'LUSC', 2: 'IDC', 3: 'ILC', 4: 'CCRCC', 5: 'PRCC', 6: 'ESAD', 7: 'ESCC'}
        print("num samples: ", len(dataloader.dataset))
        print(idx_to_class)

        prompt_file = '/home/gjx/Code/CONCH/CONCH/prompts/prompts_all_per_class.json'
        with open(prompt_file) as f:
            prompts = json.load(f)['0']
        classnames = prompts['classnames']
        templates = prompts['templates']
        n_classes = len(classnames)
        classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
        for class_idx, classname in enumerate(classnames_text):
            print(f'{class_idx}: {classname}')

        zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)
        print(zeroshot_weights.shape)

        results = run_mizero(model, zeroshot_weights, dataloader, device, eight_sutype=True, label_shift=dataset_label_shift[dataset_name])

        best_j_idx = np.argmax(list(results['acc'].values()))
        best_j = list(results['acc'].keys())[best_j_idx]
        for metric, metric_dict in results.items():
            print(f"{fold}-{dataset_name}-{metric}: {metric_dict[best_j]:.6f}")