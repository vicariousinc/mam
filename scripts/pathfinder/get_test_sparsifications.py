import os

import imageio
import joblib
import numpy as np
import typer
from mam import BASE
from mam.sparsify.bmf import sparsify
from tqdm import tqdm


def process_case(case):
    W = joblib.load(os.path.join(BASE, 'pathfinder/logs/weights_learned_16_7x7.joblib'))
    p10 = 0.0019  # prob(see a 1 when model says there's a 0)
    p01 = 0.07  # prob(see a 0 when model says there's a 1)
    c1, c0 = np.log(1.0 - p01) - np.log(p10), np.log(p01) - np.log(1.0 - p10)
    score_scale = c1 - c0
    min_score = c0
    td = -20  # invsigmoid(prob of a sparsification being on) -> best set empirically
    max_iter = 300
    if case not in ['6', '9', '14']:
        dataset_folder = os.path.join(
            BASE,
            f'pathfinder_2021/data/{case}/curv_contour_length_20_31_distractor_length_6_16',
        )
    else:
        dataset_folder = os.path.join(
            BASE, f'pathfinder/data/batch_1/curv_contour_length_{case}'
        )

    sparsification_attempts = 10

    def process_img(dataset_folder, img_idx):
        img = imageio.imread(f'{dataset_folder}/imgs/sample_{img_idx}_original.png')
        feature_activations_list = []
        for _ in range(sparsification_attempts):
            S, beliefs_dense = sparsify(
                (img > 26)[..., None],
                W,
                max_iter,
                min_score,
                score_scale,
                td,
                damping=0.9,
                parallel_bp=False,
                retries=10,
            )
            feature_activations = S[:, [2, 0, 1]]
            feature_activations_list.append(feature_activations)

        joblib.dump(
            feature_activations_list,
            f'{dataset_folder}/imgs/sample_{img_idx}_sparsifications_learned_parts.joblib',
            compress=3,
        )

    joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(process_img)(dataset_folder, img_idx)
        for img_idx in tqdm(range(25000))
    )


if __name__ == '__main__':
    typer.run(process_case)
