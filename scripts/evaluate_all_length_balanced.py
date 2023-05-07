import os

from .evaluate_with_lens import evaluate

snapshot = 'latest'
partition = 'test'
batch_size = 32
limit_tokens = 512
model='code_transformer'
label='CT'
models = [dict(run_id=9,  desc='balanced_sampling', snapshots=['latest']),
          dict(run_id=12, desc='balanced_normilized_weigts_1', snapshots=['latest']),
          dict(run_id=13, desc='balanced_normilized_weigts_2', snapshots=['latest']),
          dict(run_id=11, desc='balanced_weights', snapshots=['10000', '30000', '60000']),
          ]

results_dir = 'experiments_results_len_balanced'

lang = 'python'
os.makedirs(results_dir, exist_ok=True)
for model_info in models:
    run_id = model_info['run_id']
    desc = model_info['desc']
    for snapshot in model_info['snapshots']:
        model_id = f'{label}-{run_id}'
        save_path = os.path.join(results_dir, f"sample-metrics-{model}-{lang}-{partition}-{desc}.csv")
        if os.path.exists(save_path):
            print(f'{model} {model_id} for {lang} on {partition} already evaluated')
        else:
            evaluate(model, model_id, snapshot, save_path, partition, batch_size, limit_tokens)   
        print('------------------------------------------------')