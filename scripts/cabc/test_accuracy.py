import itertools
import os
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import joblib
import networkx as nx
import numpy as np
import sacred
import skimage.draw
import typer
from mam import BASE
from mam.cabc import backtrace, forward_pass
from mam.cabc.data import assign_marker_to_letter, get_marker_mask, load_image
from sacred.observers import FileStorageObserver
from skimage.transform import resize
from tqdm import tqdm


def backtracer_post_init(messages, context):
    preproc = context['preproc']
    messages.input.from_bottom[:] = preproc[..., 1]
    messages.internal.incoming[:] += 0.01 * np.random.logistic(
        size=messages.internal.incoming.shape
    )
    return messages


def run_experiment(
    experiment: str,
    log_folder: str = None,
    record_results: bool = True,
):
    ex = sacred.Experiment('test_accuracy')

    @ex.config
    def config():
        model_path = None
        train_difficulty = 'ix2'
        test_difficulty = 'ix2'
        n_elastic_graphs = 90000
        im_size = 9
        pool_size = 25
        threshold = 0.05
        evidence_for_0 = -0.89
        evidence_for_1 = 1.0
        fp_norm_exponent = 0.76
        n_winners_per_loc = 3
        n_instances_to_backtrace = 140
        damping = 0.4
        n_bp_iter = 60
        batch_img_indices = [('all', list(range(45000, 50000)))]
        overlap_penalty = -0.33

    @ex.main
    def run(
        model_path,
        train_difficulty,
        test_difficulty,
        n_elastic_graphs,
        im_size,
        pool_size,
        threshold,
        evidence_for_0,
        evidence_for_1,
        fp_norm_exponent,
        n_winners_per_loc,
        n_instances_to_backtrace,
        damping,
        n_bp_iter,
        batch_img_indices,
        overlap_penalty,
    ):
        fp_pool_size = 15
        step_size = 13
        elastic_graphs_list = joblib.load(model_path.format(train_difficulty))[
            :n_elastic_graphs
        ]
        elastic_graphs_list = [
            elastic_graph
            for elastic_graph in elastic_graphs_list
            if len(elastic_graph['edges']) > 0
        ]
        frcs_list = [elastic_graph['frcs'] for elastic_graph in elastic_graphs_list]
        perturb_radiuses_set = set()
        for elastic_graph in tqdm(elastic_graphs_list):
            perturb_radiuses_set.update([edge[1] for edge in elastic_graph['edges']])

        perturbation_configurations_dict = {
            perturb_radius: backtrace.get_perturbation_configurations(
                perturb_radius, pool_size
            )
            for perturb_radius in tqdm(perturb_radiuses_set)
        }
        padding = 1
        M, N = 256, 256
        im_size_without_padding = im_size - 2 * padding
        upper_shape = (M - im_size_without_padding, N - im_size_without_padding)
        proto_template = np.zeros((im_size, im_size))
        rr, cc = skimage.draw.circle(
            im_size // 2, im_size // 2, radius=im_size // 2, shape=(im_size, im_size)
        )
        proto_template[rr, cc] = 1
        features_array = proto_template[None, None, :, :, None].astype(np.int32)
        features_array = (
            forward_pass.get_features_array_with_pooling_from_binary_features_array(
                features_array
            )
        )
        or_layer_wiring = jax.device_put(
            forward_pass.get_or_layer_wiring_with_pooling(features_array)
        )
        get_preproc = jax.jit(
            forward_pass.make_get_preproc_func(or_layer_wiring, upper_shape)
        )
        fp = forward_pass.ForwardPass(
            pool_size=fp_pool_size,
            step_size=step_size,
            frcs_list=frcs_list,
            preproc_shape=(1,) + upper_shape,
            fp_norm_exponent=fp_norm_exponent,
        )
        forward_pass_func = jax.jit(fp.make_forward_pass_func())
        combinations = jnp.array(
            list(itertools.combinations(range(n_instances_to_backtrace), 2))
        )
        jitted_do_recounting = jax.jit(backtrace.do_recounting_single_feature)
        n_correct = 0
        counts = 0
        dataset_base = os.path.join(BASE, 'cabc/data')
        for batch_idx, img_indices in batch_img_indices:
            if batch_idx == 'all':
                meta = np.load(
                    os.path.join(
                        dataset_base, f'{test_difficulty}/metadata/combined.npy'
                    )
                )
            else:
                meta = np.load(
                    os.path.join(
                        dataset_base, f'{test_difficulty}/metadata/{batch_idx}.npy'
                    )
                )

            for ind in img_indices:
                start = timer()
                if batch_idx == 'all':
                    img_fname = os.path.join(
                        os.path.join(dataset_base, test_difficulty),
                        meta[ind][0].decode('utf-8'),
                        meta[ind][2].decode('utf-8'),
                    )
                    print(f'Working on image {img_fname}')
                    data_with_marker = load_image(img_fname)
                else:
                    print(f'Working on batch {batch_idx}, image {ind}')
                    data_with_marker = load_image(
                        os.path.join(
                            dataset_base,
                            f'{test_difficulty}/imgs/{batch_idx}/sample_{ind}.png',
                        )
                    )

                img_with_marker = resize(
                    data_with_marker['img'], (M, N), anti_aliasing=True
                )
                img = img_with_marker > threshold
                evidences = np.zeros((1, M, N, 2))
                evidences[0, img] = np.array([0, evidence_for_1])
                evidences[0, ~img] = np.array([0, evidence_for_0])
                preproc = get_preproc(jax.device_put(evidences))
                fp_scores = forward_pass_func(
                    preproc[..., 1], jax.device_put(fp.wiring)
                )
                top_values, top_indices = jax.lax.top_k(fp_scores.T, n_winners_per_loc)
                top_locs, top_templates = np.unravel_index(
                    np.argsort(top_values.ravel())[-n_instances_to_backtrace:],
                    top_values.shape,
                )
                top_templates = top_indices[top_locs, top_templates]
                top_indices = np.stack([top_templates, top_locs], axis=1)
                elastic_graph_dict = {}
                for idx in tqdm(np.unique(top_indices[:, 0])):
                    elastic_graph = nx.Graph()
                    elastic_graph.add_nodes_from(
                        [
                            (node_idx, dict(frc=frc))
                            for node_idx, frc in enumerate(
                                elastic_graphs_list[idx]['frcs']
                            )
                        ]
                    )
                    elastic_graph.add_edges_from(
                        [
                            (edge[0][0], edge[0][1], dict(perturb_radius=edge[1]))
                            for edge in elastic_graphs_list[idx]['edges']
                        ]
                    )
                    elastic_graph_dict[idx] = elastic_graph

                wiring_dict = {
                    idx: backtrace.get_backtracer_wiring_from_elastic_graph(
                        elastic_graph_dict[idx],
                        pool_size=pool_size,
                        perturbation_configurations_dict=perturbation_configurations_dict,
                    )
                    for idx in elastic_graph_dict
                }
                wiring_loc_list = [
                    (wiring_dict[template_idx], fp.wiring.locs[loc_idx].copy())
                    for template_idx, loc_idx in top_indices
                ]
                wiring = backtrace.concatenate_backtracer_wiring(wiring_loc_list)
                backtracer = backtrace.Backtracer(damping=damping)
                context = dict(preproc=preproc)
                messages = backtracer.initialize_messages(
                    shape=(1,) + upper_shape,
                    wiring=wiring,
                    post_init=backtracer_post_init,
                    context=context,
                )
                jitted_infer_backtracer = jax.jit(
                    jax.partial(
                        backtracer.make_infer_func(),
                        n_bp_iter=n_bp_iter,
                    )
                )
                messages = jitted_infer_backtracer(messages, jax.device_put(wiring))
                recount_scores, all_backtraced_locs = jitted_do_recounting(
                    combinations,
                    evidences,
                    messages,
                    jax.device_put(wiring),
                    or_layer_wiring.features_description,
                    overlap_penalty,
                )
                inst_indices = combinations[jnp.argmax(recount_scores)]
                backtraced_locs_list = [
                    np.unique(
                        all_backtraced_locs[idx][
                            np.sum(all_backtraced_locs[idx] == -1, axis=-1) == 0
                        ],
                        axis=0,
                    )
                    for idx in inst_indices
                ]
                backtraced_locs_list = [
                    backtraced_locs[
                        np.logical_and(
                            backtraced_locs[:, 1] < M, backtraced_locs[:, 2] < N
                        )
                    ]
                    for backtraced_locs in backtraced_locs_list
                ]
                letter_mask_list = [
                    np.zeros_like(img) for _ in range(len(inst_indices))
                ]
                for ii in range(len(inst_indices)):
                    letter_mask_list[ii][
                        backtraced_locs_list[ii][:, 1], backtraced_locs_list[ii][:, 2]
                    ] = 1

                marker_mask_list = get_marker_mask(
                    data_with_marker['img'], target_shape=(M, N)
                )
                assignment = assign_marker_to_letter(marker_mask_list, letter_mask_list)
                gt_is_on_same_letter = int(meta[ind][4]) == 1
                is_correct = (len(np.unique(assignment)) == 1) == gt_is_on_same_letter
                info = f'Processing batch {batch_idx}, image {ind} took {timer() - start}s. '
                if is_correct:
                    n_correct += 1
                    info += 'Correct prediction. '
                else:
                    info += 'Incorrect prediction. '

                counts += 1
                info += f'{n_correct}/{counts} correct so far. Accuracy {n_correct / counts}'
                print(info)

    if log_folder is None:
        log_folder = os.path.join(BASE, 'cabc/logs/test_accuracy')

    if record_results:
        ex.observers = [
            FileStorageObserver.create(os.path.join(log_folder, experiment))
        ]

    if experiment == 'baseline-':
        # baseline- experiments
        for n_elastic_graphs in [
            90000,
            40000,
            20000,
            10000,
            4000,
            2000,
            1000,
            400,
            200,
            100,
        ]:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "baseline-",
                    "test_difficulty": "baseline-",
                    "n_elastic_graphs": n_elastic_graphs,
                }
            )
    elif experiment == 'baseline-_cross':
        for test_difficulty in ['ix1-', 'ix2']:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "baseline-",
                    "test_difficulty": test_difficulty,
                    "n_elastic_graphs": 90000,
                }
            )
    elif experiment == 'ix1-':
        # ix1- experiments
        for n_elastic_graphs in [
            90000,
            40000,
            20000,
            10000,
            4000,
            2000,
            1000,
            400,
            200,
            100,
        ]:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "ix1-",
                    "test_difficulty": "ix1-",
                    "n_elastic_graphs": n_elastic_graphs,
                }
            )
    elif experiment == 'ix1-_cross':
        for test_difficulty in ['baseline-', 'ix2']:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "ix1-",
                    "test_difficulty": test_difficulty,
                    "n_elastic_graphs": 90000,
                }
            )
    elif experiment == 'ix2':
        # ix2 experiments
        for n_elastic_graphs in [
            90000,
            40000,
            20000,
            10000,
            4000,
            2000,
            1000,
            400,
            200,
            100,
        ]:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "ix2",
                    "test_difficulty": "ix2",
                    "n_elastic_graphs": n_elastic_graphs,
                }
            )
    elif experiment == 'ix2_cross':
        for test_difficulty in ['baseline-', 'ix1-']:
            ex.run(
                config_updates={
                    "model_path": os.path.join(
                        BASE,
                        'cabc/logs/elastic_graphs_for_cabc_{}_learned_part.joblib',
                    ),
                    "train_difficulty": "ix2",
                    "test_difficulty": test_difficulty,
                    "n_elastic_graphs": 90000,
                }
            )
    else:
        raise ValueError(f'Unsupported experiment {experiment}.')


if __name__ == "__main__":
    typer.run(run_experiment)
