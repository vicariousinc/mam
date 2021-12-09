import os
from collections import defaultdict
from timeit import default_timer as timer
from typing import Tuple

import imageio
import jax
import joblib
import networkx as nx
import numba
import numpy as np
import sacred
import skimage.draw
import typer
from mam import BASE
from mam.pathfinder import binary_lateral
from mam.pathfinder.sparse import get_interaction_graph_from_feature_activations
from mam.sparsify.bmf import sparsify
from sacred.observers import FileStorageObserver
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm


@numba.jit(nopython=True, cache=True)
def get_img_reconstruction_with_labels(feature_activations, W):
    img_with_labels = np.zeros((300, 300), dtype=np.int32)
    for idx in range(feature_activations.shape[0]):
        frc = feature_activations[idx]
        img_with_labels[frc[1] : frc[1] + W.shape[1], frc[2] : frc[2] + W.shape[2]] = (
            idx + 1
        )

    return img_with_labels


def run_experiment(
    n_batches_for_training: str,
    train_difficulty: str,
    test_difficulty: str,
    start_end_indices: Tuple[int, int] = (0, 25000),
    log_folder: str = None,
    record_results: bool = True,
):
    ex = sacred.Experiment('test_accuracy_binary_lateral_interface')

    @ex.config
    def config():
        n_batches_for_training = '2'
        train_difficulty = '14'
        test_difficulty = '14'
        start_end_indices = (0, 25000)

    @ex.main
    def run(
        n_batches_for_training,
        train_difficulty,
        test_difficulty,
        start_end_indices,
        _run,
        _seed,
    ):
        if len(_run.observers) > 0:
            log_path = _run.observers[0].dir
        else:
            log_path = None

        laterals_counter_fname = os.path.join(
            BASE,
            f'pathfinder/logs/mam_train_laterals_{n_batches_for_training}_batches_case_{train_difficulty}_with_counts_learned_parts.joblib',
        )
        print(f'Using laterals set from {laterals_counter_fname}')
        laterals_counter = joblib.load(laterals_counter_fname)
        threshold = np.sum(list(laterals_counter.values())) * 1.7e-7
        laterals_set = set(
            [key for key in laterals_counter if laterals_counter[key] >= threshold]
        )
        W = joblib.load(
            os.path.join(BASE, 'pathfinder/logs/weights_learned_16_7x7.joblib')
        )
        patch_M, patch_N = W.shape[1:3]
        n_features = W.shape[0]
        laterals_identifiers_list = [defaultdict(list) for _ in range(n_features)]
        for lateral in tqdm(laterals_set):
            f1, f2, dr, dc = list(lateral)[0]
            points0 = np.argwhere(W[f1, ..., 0])
            points1 = np.array([dr, dc]) + np.argwhere(W[f2, ..., 0])
            id0, id1 = np.unravel_index(
                np.argmin(cdist(points0, points1)), (points0.shape[0], points1.shape[0])
            )
            laterals_identifiers_list[f1][id0].append((f2, dr, dc))
            laterals_identifiers_list[f2][id1].append((f1, -dr, -dc))

        kmeans = KMeans(n_clusters=2)
        pools_to_laterals_list = []
        for f1 in range(n_features):
            identifiers = np.array(list(laterals_identifiers_list[f1].keys()))
            indices = np.argwhere(W[f1, ..., 0])[identifiers]
            labels = kmeans.fit_predict(indices)
            pools_to_laterals = []
            for pool_idx in range(2):
                pool_to_laterals = dict()
                for identifier in identifiers[labels == pool_idx]:
                    for f2, dr, dc in laterals_identifiers_list[f1][identifier]:
                        pool_to_laterals[
                            frozenset([(f1, f2, dr, dc), (f2, f1, -dr, -dc)])
                        ] = np.array([f2, dr, dc])

                pools_to_laterals.append(pool_to_laterals)

            pools_to_laterals_list.append(pools_to_laterals)

        templates_list = [
            [
                [0],
                [1],
                [0, 1],
            ]
            for _ in range(n_features)
        ]

        if test_difficulty not in ['6', '9', '14']:
            dataset_folder = os.path.join(
                BASE,
                f'pathfinder_2021/data/{test_difficulty}/curv_contour_length_20_31_distractor_length_6_16',
            )
        else:
            dataset_folder = os.path.join(
                BASE,
                f'pathfinder/data/batch_1/curv_contour_length_{test_difficulty}',
            )

        meta = np.load(
            os.path.join(dataset_folder, 'metadata/metadata.npy'),
            allow_pickle=True,
            encoding='latin1',
        ).tolist()
        damping = 0.5
        boundary_conditions = -1000.0
        jitted_infer = jax.jit(
            jax.partial(
                binary_lateral.infer,
                n_bp_iter=30,
                boundary_conditions=boundary_conditions,
                damping=damping,
            )
        )
        n_correct = 0
        counts = 0
        incorrect_indices = []
        for img_idx in range(start_end_indices[0], start_end_indices[1]):
            start = timer()
            sparsifications_fname = f'{dataset_folder}/imgs/sample_{img_idx}_sparsifications_learned_parts.joblib'
            if not os.path.exists(sparsifications_fname):
                p10 = 0.0019  # prob(see a 1 when model says there's a 0)
                p01 = 0.07  # prob(see a 0 when model says there's a 1)
                c1, c0 = np.log(1.0 - p01) - np.log(p10), np.log(p01) - np.log(
                    1.0 - p10
                )
                score_scale = c1 - c0
                min_score = c0
                td = (
                    -20
                )  # invsigmoid(prob of a sparsification being on) -> best set empirically
                max_iter = 300
                img = imageio.imread(
                    f'{dataset_folder}/imgs/sample_{img_idx}_original.png'
                )
                #  print(f'Loading image took {timer() - start}')
                #  start = timer()
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
            else:
                feature_activations = joblib.load(sparsifications_fname)[
                    np.random.randint(10)
                ]

            #  print(f'Sparsification took {timer() - start}')
            #  start = timer()
            interaction_graph = get_interaction_graph_from_feature_activations(
                feature_activations, pools_to_laterals_list, templates_list
            )
            wiring = binary_lateral.get_wiring_from_interaction_graph(interaction_graph)
            #  print(f'Wring construction took {timer() - start}')
            #  start = timer()
            logw = np.zeros((feature_activations.shape[0], 3))
            logw[:, :2] = -1.6
            messages = binary_lateral.initialize_messages(
                input=1000.0,
                boundary_conditions=boundary_conditions,
                wiring=wiring,
            )
            #  print(f'Messages init took {timer() - start}')
            #  start = timer()
            messages = jax.device_get(
                jitted_infer(
                    jax.device_put(messages),
                    jax.device_put(wiring),
                    jax.device_put(logw),
                )
            )
            #  print(f'Inference took {timer() - start}')
            #  start = timer()
            G = binary_lateral.get_laterals_connectivity(
                messages, wiring, feature_activations
            )
            #  print(f'Connectivity graph took {timer() - start}')
            #  start = timer()
            img_with_labels = get_img_reconstruction_with_labels(feature_activations, W)
            #  print(f'img_with_labels took {timer() - start}')
            #  start = timer()
            marker_centers = meta[f'imgs/sample_{img_idx}.png']['marker_centers']
            fallback_assignments = [
                np.argmin(
                    np.linalg.norm(
                        (np.array(center) - np.array([6, 6])).reshape((1, -1))
                        - feature_activations[:, 1:],
                        axis=1,
                    )
                )
                for center in marker_centers
            ]
            true_label = meta[f'imgs/sample_{img_idx}.png']['label']
            marker_assignments = []
            for center, fallback_assignment in zip(
                marker_centers, fallback_assignments
            ):
                rr, cc = skimage.draw.disk(center, radius=5)
                marker_assignment = None
                if np.sum(img_with_labels[rr, cc]) > 0:
                    assignments, assignments_counts = np.unique(
                        img_with_labels[rr, cc][img_with_labels[rr, cc] > 0] - 1,
                        return_counts=True,
                    )
                    for assignment in assignments[np.argsort(-assignments_counts)]:
                        if G.degree(tuple(feature_activations[assignment])) == 1:
                            marker_assignment = int(assignment)
                            break

                if marker_assignment is None:
                    marker_assignment = int(fallback_assignment)

                marker_assignments.append(marker_assignment)

            prediction = nx.has_path(
                G,
                tuple(feature_activations[marker_assignments[0]]),
                tuple(feature_activations[marker_assignments[1]]),
            )
            #  print(f'Making prediction took {timer() - start}')
            is_correct = bool(true_label) == prediction
            info = f'Testing on {test_difficulty} with {n_batches_for_training} batches training images on {train_difficulty}.\n'
            info += (
                f'{dataset_folder}/imgs/sample_{img_idx}.png took {timer() - start}s. '
            )
            if is_correct:
                n_correct += 1
                info += 'Correct prediction. '
            else:
                incorrect_indices.append(img_idx)
                info += 'Incorrect prediction. '

            counts += 1
            info += f'GT: {bool(true_label)}, pred: {prediction}. {n_correct}/{counts} correct so far. Accuracy {n_correct / counts}'
            print(info)

        if log_path is not None:
            joblib.dump(
                dict(
                    n_correct=n_correct,
                    counts=counts,
                    incorrect_indices=incorrect_indices,
                ),
                os.path.join(log_path, 'accuracy_statistics.joblib'),
            )

    if log_folder is None:
        log_folder = os.path.join(
            BASE,
            'pathfinder/logs/pathfinder_results_for_aaai2022',
        )

    if record_results:
        ex.observers = [
            FileStorageObserver.create(
                os.path.join(
                    log_folder,
                    f'train_on_{n_batches_for_training}_batches_{train_difficulty}_test_on_{test_difficulty}',
                )
            )
        ]

    ex.run(
        config_updates=dict(
            n_batches_for_training=n_batches_for_training,
            train_difficulty=train_difficulty,
            test_difficulty=test_difficulty,
            start_end_indices=start_end_indices,
        )
    )


if __name__ == '__main__':
    typer.run(run_experiment)
