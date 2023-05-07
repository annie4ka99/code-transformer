"""
Evaluates a stored snapshot on the valid or test partition. Performance is measured in micro-F1 score.
Usage: python -m scripts.evaluate {model} {run_id} {snapshot_iteration} {partition}
The actual language is inferred from the trained model.
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import os

from code_transformer.modeling.constants import PAD_TOKEN, UNKNOWN_TOKEN, NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager import GreatModelManager, XLNetModelManager
from code_transformer.modeling.modelmanager.code_transformer import CodeTransformerModelManager
from code_transformer.preprocessing.datamanager.base import batch_to_device, batch_filter_distances
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.ablation import CTCodeSummarizationOnlyASTDataset
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDataset, \
    CTCodeSummarizationDatasetEdgeTypes, CTCodeSummarizationDatasetNoPunctuation
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import TokenDistancesTransform

from code_transformer.env import DATA_PATH_STAGE_2


def save_lengths(model_type, run_id, snapshot_iteration, save_path, partition='valid', batch_size=8, limit_tokens = 1000, no_gpu=False):
    if model_type == 'code_transformer':
        model_manager = CodeTransformerModelManager()
    elif model_type == 'great':
        model_manager = GreatModelManager()
    elif model_type == 'xl_net':
        model_manager = XLNetModelManager()
    else:
        raise ValueError(f"Unknown model type `{model}`")

    model = model_manager.load_model(run_id, snapshot_iteration, gpu=not no_gpu)
    model = model.eval()
    if not no_gpu:
        model = model.cuda()

    config = model_manager.load_config(run_id)
    data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2,
                                         config['data_setup']['language'],
                                         partition=partition,
                                         shuffle=False)  

    token_distances = None
    if TokenDistancesTransform.name in config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['binning']['num_bins']
        distance_binning_config = config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))

    use_pointer_network = config['data_setup']['use_pointer_network']
    if model_type in {'great'}:
        dataset_type = 'great'
    elif 'use_only_ast' in config['data_setup'] and config['data_setup']['use_only_ast']:
        dataset_type = 'only_ast'
    elif 'use_no_punctuation' in config['data_setup'] and config['data_setup']['use_no_punctuation']:
        dataset_type = 'no_punctuation'
    else:
        dataset_type = 'regular'

    print(
        f"Evaluating model snapshot-{snapshot_iteration} from run {run_id} on {config['data_setup']['language']} partition {partition}")
    print(f"gpu: {not no_gpu}")
    print(f"dataset_type: {dataset_type}")
    print(f"model: {model}")
    print(f"use_pointer_network: {use_pointer_network}")

    if dataset_type == 'great':
        dataset = CTCodeSummarizationDatasetEdgeTypes(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                      use_pointer_network=use_pointer_network,
                                                      token_distances=token_distances, max_num_tokens=limit_tokens)
    elif dataset_type == 'regular':
        dataset = CTCodeSummarizationDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                             use_pointer_network=use_pointer_network, max_num_tokens=limit_tokens,
                                             token_distances=token_distances)
    elif dataset_type == 'no_punctuation':
        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager,
                                                          num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                          use_pointer_network=use_pointer_network,
                                                          max_num_tokens=limit_tokens,
                                                          token_distances=token_distances)
    elif dataset_type == 'only_ast':
        dataset = CTCodeSummarizationOnlyASTDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                    use_pointer_network=use_pointer_network,
                                                    max_num_tokens=limit_tokens, token_distances=token_distances)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size)

    relative_distances = config['data_transforms']['relative_distances']

    lengths = []
    progress = tqdm(enumerate(dataloader), total=int(data_manager.approximate_total_samples() / batch_size))

    for _, batch in progress:
        batch = batch_filter_distances(batch, relative_distances)
        
        if not no_gpu:
            batch = batch_to_device(batch)
        
        lengths.extend(batch.sequence_lengths.tolist())

        progress.set_description()
        del batch
        gc.collect()
        torch.cuda.empty_cache()

    data_manager.shutdown()

    print(f"Storing metrics into {save_path}")
    with open(save_path, 'wb') as f:     
        np.save(f, np.array(lengths))    



if __name__ == '__main__':
    snapshot = 'latest'
    batch_size = 128
    limit_tokens = 512
    models = [dict(model='code_transformer',    label='CT', run_ids=range(5, 9)), 
            dict(model='great',               label='GT', run_ids=range(1, 5)),
            dict(model='xl_net',              label='XL', run_ids=range(1, 5))]
    results_dir = 'samples_info'

    csn_langs = ['python', 'javascript', 'ruby', 'go']
    os.makedirs(results_dir, exist_ok=True)
    for model_info in models:
        model = model_info['model']
        label = model_info['label']
        for run_id in model_info['run_ids']:
            model_id = f'{label}-{run_id}'
            lang = csn_langs[(run_id - 1) % 4]
            for partition in ['train', 'valid']:
                save_path = os.path.join(results_dir, f"lengths-{model}-{lang}-{partition}.npy")
                if os.path.exists(save_path):
                    print(f'{model} {model_id} for {lang} on {partition} already evaluated')
                else:
                    save_lengths(model, model_id, snapshot, save_path, partition, batch_size, limit_tokens, no_gpu=True)
                print('------------------------------------------------')

