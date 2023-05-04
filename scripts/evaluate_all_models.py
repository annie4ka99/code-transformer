import os

from .evaluate_with_lens import evaluate

snapshot = 'latest'
partition = 'test'
batch_size = 32
limit_tokens = 512
models = [dict(model='code_transformer',    label='CT', run_ids=range(5, 9)), 
          dict(model='great',               label='GT', run_ids=range(1, 5)),
          dict(model='xl_net',              label='XL', run_ids=range(1, 5))]
results_dir = 'experiments_results'


csn_langs = ['python', 'javascript', 'ruby', 'go']
os.makedirs(results_dir, exist_ok=True)
for model_info in models:
    model = model_info['model']
    label = model_info['label']
    for run_id in model_info['run_ids']:
        model_id = f'{label}-{run_id}'
        lang = csn_langs[(run_id - 1) % 4]
        save_path = os.path.join(results_dir, f"sample-metrics-{model}-{lang}-{partition}.csv")
        if os.path.exists(save_path):
            print(f'{model} {model_id} for {lang} on {partition} already evaluated')
        else:
            evaluate(model, model_id, snapshot, save_path, partition, batch_size, limit_tokens)   
        print('------------------------------------------------')