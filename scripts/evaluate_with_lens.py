"""
Evaluates a stored snapshot on the valid or test partition. Performance is measured in micro-F1 score.
Usage: python -m scripts.evaluate {model} {run_id} {snapshot_iteration} {partition}
The actual language is inferred from the trained model.
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import gc
from sacrebleu.metrics import CHRF

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
from code_transformer.utils.metrics import get_tp_fp_fn, compute_rouge, get_best_non_unk_predictions
from code_transformer.utils.inference import decode_predicted_tokens

from code_transformer.env import DATA_PATH_STAGE_2


def evaluate(model_type, run_id, snapshot_iteration, save_path, partition='valid', batch_size=8, limit_tokens = 1000, no_gpu=False):
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
    vocabularies = data_manager.load_vocabularies()
    if len(vocabularies) == 3:
        word_vocab, _, _ = vocabularies
    else:
        word_vocab, _, _, _ = vocabularies

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

    pad_id = word_vocab[PAD_TOKEN]
    unk_id = word_vocab[UNKNOWN_TOKEN]

    predictions = []
    labels = []
    losses = []
    lengths = []
    tps = []
    fps = []
    fns = []
    chrf_scores = []
    progress = tqdm(enumerate(dataloader), total=int(data_manager.approximate_total_samples() / batch_size))
    chrf_metric = CHRF()

    for i, batch in progress:
        batch = batch_filter_distances(batch, relative_distances)
        
        if not no_gpu:
            batch = batch_to_device(batch)

        label = batch.labels.detach().cpu()

        with torch.no_grad():
            output = model.forward_batch(batch).cpu()
        losses.append(output.loss.item())
        tp, fp, fn = get_tp_fp_fn(output.logits, label, pad_id=pad_id, unk_id=unk_id)
        
        lengths.extend(batch.sequence_lengths.tolist())
        tps.extend(tp)
        fps.extend(fp)
        fns.extend(fn)

        batch_logits = output.logits.detach().cpu()
        predictions.extend(batch_logits.argmax(-1).squeeze(1))
        label = label.squeeze(1)
        labels.extend(label)

        best_non_unk_predictions = get_best_non_unk_predictions(output.logits, unk_id=unk_id).squeeze(1)
        # print(best_non_unk_predictions.shape)
        for i in range(len(best_non_unk_predictions)):
            predicted_method_name = decode_predicted_tokens(best_non_unk_predictions[i], batch, data_manager)
            if len(predicted_method_name) == 0:
                predicted_method_name = 'EMPTY'
            else:
                predicted_method_name = ' '.join(predicted_method_name)
            
            gt_method_name = ' '.join(decode_predicted_tokens(label[i], batch, data_manager))
            if len(gt_method_name) == 0:
                gt_method_name = 'EMPTY'
            else:
                gt_method_name = ' '.join(gt_method_name)

            chrf_scores.append(chrf_metric.sentence_score(predicted_method_name, [gt_method_name]))

        progress.set_description()
        del batch
        gc.collect()
        torch.cuda.empty_cache()

    data_manager.shutdown()

    predictions = torch.stack(predictions)
    labels = torch.stack(labels)

    scores = compute_rouge(predictions, labels, pad_id=pad_id, predictions_provided=True, per_sample=True)

    print(f"Storing metrics into {save_path}")
    with open(save_path, 'w', newline='') as csvfile:     
        writer = csv.writer(csvfile)
        writer.writerow(['length', 'tp', 'fp', 'fn', 'rouge1-f', 'rouge2-f', 'rougeL-f', 'chrf'])

        for i in range(len(lengths)):
            writer.writerow([lengths[i], 
                             tps[i], fps[i], fns[i],
                             scores[i]['rouge-1']['f'], scores[i]['rouge-2']['f'], scores[i]['rouge-l']['f'],
                             chrf_scores[i]
                             ])



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model",
                        choices=['code_transformer', 'xl_net', 'great'])
    parser.add_argument("run_id", type=str)
    parser.add_argument("snapshot_iteration", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("partition", type=str, choices=['train', 'valid', 'test'], default='valid')
    parser.add_argument("batch_size", type=int, default=8)
    parser.add_argument("limit_tokens", type=int, default=1000)
    parser.add_argument("--no-gpu", action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args.model, args.run_id, args.snapshot_iteration, args.save_path, args.partition, args.batch_size, args.limit_tokens, args.no_gpu)

