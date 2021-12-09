import os

import joblib
import numpy as np
import sacred
from mam import BASE
from mam.cabc.data import load_image
from mam.sparsify import underconstraint
from mam.sparsify.bmf import sparsify
from sacred.observers import FileStorageObserver
from skimage.transform import resize
from tqdm import tqdm

log_folder = os.path.join(BASE, 'cabc/logs/learn_elastic_graphs_learned_part')
ex = sacred.Experiment('learn_elastic_graphs')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    M = 256
    N = 256
    threshold = 0.0
    perturb_factor = 7.0
    max_cxn_length = 200
    tolerance = 2
    difficulty = 'ix2'
    p10 = 0.05  # prob(see a 1 when model says there's a 0)
    p01 = 0.45  # prob(see a 0 when model says there's a 1)
    td = -20  # invsigmoid(prob of a sparsification being on) -> best set empirically
    max_iter = 300


@ex.automain
def run(
    M,
    N,
    threshold,
    perturb_factor,
    max_cxn_length,
    tolerance,
    difficulty,
    p10,
    p01,
    td,
    max_iter,
    _run,
):
    if len(_run.observers) > 0:
        log_path = _run.observers[0].dir
    else:
        log_path = None

    dataset_base = os.path.join(BASE, 'cabc/data')
    meta = np.load(os.path.join(dataset_base, f'{difficulty}/metadata/combined.npy'))
    proto_template = joblib.load(
        os.path.join(BASE, 'cabc/logs/learned_object_part.joblib')
    )
    W = proto_template[None, ..., None].astype(bool)
    c1, c0 = np.log(1.0 - p01) - np.log(p10), np.log(p01) - np.log(1.0 - p10)
    score_scale = c1 - c0
    min_score = c0

    def process_img(elastic_graph_ind):
        img_ind = elastic_graph_ind // 2
        img_fname = os.path.join(
            os.path.join(dataset_base, f'{difficulty}'),
            meta[img_ind][1].decode('utf-8'),
            meta[img_ind][2].decode('utf-8'),
        )
        data_segs = load_image(img_fname)
        img = [
            resize(data_segs['img'][..., ii], (M, N), anti_aliasing=True)
            for ii in range(2)
        ]
        S, beliefs_dense = sparsify(
            (img[elastic_graph_ind % 2] > threshold)[..., None],
            W,
            max_iter,
            min_score,
            score_scale,
            td,
            damping=0.9,
            parallel_bp=False,
            retries=10,
        )
        frcs = np.argwhere(beliefs_dense > 0)[:, [2, 0, 1]]
        graph = underconstraint.add_underconstraint_perturb_cxns(
            frcs,
            max_cxn_length=max_cxn_length,
            tolerance=tolerance,
            perturb_factor=perturb_factor,
            min_perturb_radius=1,
        )
        underconstraint.adjust_perturbation_distances(
            graph, perturb_factor=perturb_factor
        )
        elastic_graph = {
            'frcs': np.array(
                [graph.nodes[node]['frc'] for node in range(graph.number_of_nodes())]
            ),
            'edges': np.array(
                [(edge, graph.edges[edge]['perturb_radius']) for edge in graph.edges],
                dtype=object,
            ),
        }
        if log_path is not None:
            joblib.dump(
                elastic_graph,
                os.path.join(
                    log_path,
                    f'elastic_graphs_list_{difficulty}_{elastic_graph_ind}.joblib',
                ),
                compress=3,
            )

    joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(process_img)(elastic_graph_ind)
        for elastic_graph_ind in tqdm(range(90000))
    )
