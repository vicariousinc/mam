import os
from collections import Counter
from typing import Tuple

import joblib
import networkx as nx
import numpy as np
import typer
from mam import BASE
from mam.sparsify.bmf import sparsify
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from tqdm import tqdm

root_folder = os.path.join(BASE, 'pathfinder/data/')


def get_successor(start, successors_dict, allowed_nodes):
    curr = start
    while curr not in allowed_nodes:
        if curr in successors_dict:
            #  assert len(successors_dict[curr]) == 1
            curr = successors_dict[curr][0]
        else:
            curr = None
            break

    return curr


def process_batch(
    batch_idx: int,
    cases: str = '6,9,14',
    start_end_idx: Tuple[int, int] = (0, 25000),
    sparsification_attempts: int = 10,
):
    cases = cases.split(',')
    p10 = 0.0019  # prob(see a 1 when model says there's a 0)
    p01 = 0.07  # prob(see a 0 when model says there's a 1)
    c1, c0 = np.log(1.0 - p01) - np.log(p10), np.log(p01) - np.log(1.0 - p10)
    score_scale = c1 - c0
    min_score = c0
    td = -20  # invsigmoid(prob of a sparsification being on) -> best set empirically
    max_iter = 300

    W = joblib.load(os.path.join(BASE, 'pathfinder/logs/weights_learned_16_7x7.joblib'))

    def worker_fun(dataset_folder, idx):
        fname = os.path.join(
            dataset_folder, 'imgs', f'sample_{idx}_separate_images.joblib'
        )
        separate_images = joblib.load(fname)
        graphs_list = []
        laterals_counter = Counter()
        for attempt_idx in range(sparsification_attempts):
            for ind in range(len(separate_images['original'])):
                img = separate_images['original'][ind]
                skele = skeletonize(separate_images['filled_in'][ind] > 0.2)
                S, beliefs_dense = sparsify(
                    (img > 26.0 / 255)[..., None],
                    W,
                    max_iter,
                    min_score,
                    score_scale,
                    td,
                    damping=0.9,
                    parallel_bp=False,
                    retries=10,
                )
                frcs = S[:, [2, 0, 1]]
                nodes = np.argwhere(skele)
                graph = nx.Graph()
                graph.add_nodes_from([tuple(node) for node in nodes])
                distance = np.max(np.abs(nodes[:, None] - nodes[None, :]), axis=-1)
                graph.add_edges_from(
                    [
                        (tuple(nodes[edge[0]]), tuple(nodes[edge[1]]))
                        for edge in np.argwhere(
                            np.logical_and(distance > 0, distance <= 1)
                        )
                    ]
                )
                assignments = np.argmin(cdist(frcs[:, 1:], nodes), axis=1)
                assigned_nodes = [tuple(nodes[node]) for node in assignments]
                sparse_graph = nx.Graph()
                sparse_graph.add_nodes_from(
                    [
                        (assigned_node, dict(frc=frcs[node_idx]))
                        for node_idx, assigned_node in enumerate(assigned_nodes)
                    ]
                )
                for node in assigned_nodes:
                    successors_dict = nx.dfs_successors(graph, node)
                    immediate_successors = successors_dict[node]
                    for start in immediate_successors:
                        successor = get_successor(
                            start, successors_dict, set(assigned_nodes)
                        )
                        if successor is not None:
                            sparse_graph.add_edge(node, successor)

                graphs_list.append(sparse_graph)
                for edge in sparse_graph.edges():
                    frc0, frc1 = (
                        sparse_graph.nodes[edge[0]]['frc'],
                        sparse_graph.nodes[edge[1]]['frc'],
                    )
                    f1, f2, dr, dc = (
                        frc0[0],
                        frc1[0],
                        frc1[1] - frc0[1],
                        frc1[2] - frc0[2],
                    )
                    laterals_counter.update(
                        [frozenset([(f1, f2, dr, dc), (f2, f1, -dr, -dc)])]
                    )

        joblib.dump(
            laterals_counter,
            os.path.join(
                dataset_folder, 'imgs', f'sample_{idx}_laterals_learned_parts.joblib'
            ),
            compress=3,
        )

    for case in cases:
        dataset_folder = os.path.join(
            root_folder, f'batch_{batch_idx}', f'curv_contour_length_{case}'
        )
        print('Working on {}'.format(dataset_folder))
        joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(worker_fun)(dataset_folder, idx)
            for idx in tqdm(range(start_end_idx[0], start_end_idx[1]))
        )


if __name__ == '__main__':
    typer.run(process_batch)
