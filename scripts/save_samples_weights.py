import os
from collections import defaultdict
import numpy as np

def get_samples_weights(model='code_transformer', lang='python', 
                        partition='valid',
                        results_dir='./samples_info', max_tokens = 512):
    os.makedirs(results_dir, exist_ok=True)

    lens_path = os.path.join(results_dir, f"lengths-{model}-{lang}-{partition}.npy")
    save_path = os.path.join(results_dir, f"weights-{model}-{lang}-{partition}.npy")
    if not os.path.exists(lens_path):
        print(f'cant find file: {lens_path}')
        return
    lengths = np.load(lens_path)

    cnts = defaultdict(int)
    for l in lengths:
        cnts[l] += 1

    for l in range(0, max_tokens+2):
        if l not in cnts:
            cnts[l] = 1

    n = sum(cnts.values())

    weights = np.array([(n/cnts[l]) for l in range(0, max_tokens+2)])
    weights /= len(weights)

    print("total sum:", sum([cnts[l]*weights[l] for l in cnts]))

    with open(save_path, 'wb') as f:
        np.save(f, weights)


get_samples_weights()