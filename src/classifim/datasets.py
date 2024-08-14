import classifim.bench.fidelity
import classifim.io
import datasets
import numpy as np
import re

def _process_dataset(sm_name, column_names, dataset, res):
    """
    Process dataset.

    Args:
        sm_name (str): Name of the statistical manifold.
        column_names (set): Set of column names to process.
            processed features are removed from the set.
        dataset (datasets.Dataset): Dataset.
        res (dict): Dictionary to store the processed data.
    """
    f = _DATASET_PROCESSORS.get(sm_name, None)
    if f:
        f(column_names, dataset, res)

def _process_ising400(column_names, dataset, res):
    """
    Custom processing step for Ising400 and IsNNN400 statistical manifolds.
    """
    assert 'sample' in column_names, f"{column_names}"
    res['height'] = height = 20
    res['width'] = width = 20
    samples = classifim.io.bytes_to_samples2d(
            classifim.io.pa_to_np(dataset.data['sample']),
            height=height, width=width)
    # width <= 31, so we can store the samples in a uint32
    # and not lose information when int32 is used in pytorch.
    res['samples'] = samples.astype(np.uint32)
    column_names.remove('sample')

def _process_xxz300(column_names, dataset, res):
    """
    Custom processing step for XXZ300Z and XXZ300P statistical manifolds.
    """
    assert 'sample' in column_names, f"{column_names}"
    res['num_sites'] = num_sites = 300
    samples = classifim.io.pa_to_np(dataset.data['sample'])
    samples = classifim.io.samples1d_bytes_to_uint8(samples, num_sites)
    res['samples'] = samples
    column_names.remove('sample')

_DATASET_PROCESSORS = {
    'ising_400': _process_ising400,
    'isnnn_400': _process_ising400,
    'xxz_300_z': _process_xxz300,
    'xxz_300_p': _process_xxz300
}

def dataset_huggingface_to_dict(dataset):
    """
    Convert HuggingFace dataset to a dictionary of np.arrays.
    """
    column_names = set(dataset.column_names)
    assert 'lambda0' in column_names
    config_name = dataset.config_name.split('.')
    assert len(config_name) == 2
    res = {}
    seed_match = re.match(r'^seed(\d+)$', config_name[1])
    assert seed_match is not None, config_name
    res['seed'] = int(seed_match.group(1))
    lambdas = [dataset.data['lambda0'].to_numpy()]
    column_names.remove('lambda0')
    num_lambdas = 1
    while True:
        key = f'lambda{num_lambdas}'
        if key not in dataset.column_names:
            break
        lambdas.append(dataset.data[key].to_numpy())
        column_names.remove(key)
        num_lambdas += 1
    res['lambdas'] = np.stack(lambdas, axis=1)
    num_rows, num_lambdas1 = res['lambdas'].shape
    assert num_lambdas1 == num_lambdas

    _process_dataset(config_name[0], column_names, dataset, res)

    for column_name in column_names:
        res[column_name + 's'] = classifim.io.pa_to_np(
                dataset.data[column_name])
    return res

def _load_gt_fim(gt_fim_dataset):
    """
    Reads the ground truth FIM from the HuggingFace dataset object.
    """
    gt_fim = classifim.bench.fidelity.as_data_frame(
        gt_fim_dataset.to_pandas(), decode=True)

    if 'lambda0' not in gt_fim.columns:
        raise ValueError(
            "The GT FIM dataset does not contain the 'lambda0' column.")

    if 'lambda1' not in gt_fim.columns:
        # 1D statistical manifold:
        return gt_fim

    if 'lambda2' in gt_fim.columns:
        raise NotImplementedError(
            "Only 1D and 2D statistical manifolds are currently supported.")

    # 2D statistical manifold:
    gt_fim_mgrid = classifim.bench.fidelity.meshgrid_transform_2D_fim(gt_fim)
    return gt_fim, gt_fim_mgrid

def load_gt_fim(config_name=None, sm_name=None):
    if (config_name is None) == (sm_name is None):
        raise ValueError(
            "Exactly one of config_name and sm_name must be specified.")
    if config_name is None:
        config_name = f"{sm_name}.gt_fim"
    # Test split should be the only split: in the training data
    # there is no gt_fim, only the samples.
    gt_fim_dataset = datasets.load_dataset(
        'fiktor/FIM-Estimation', config_name, split='test')
    return _load_gt_fim(gt_fim_dataset)

def load_gt_fims(config_names=None, sm_name=None):
    if config_names is None:
        if sm_name is None:
            raise ValueError(
                "Either config_names or sm_name must be specified.")
        all_config_names = datasets.get_dataset_config_names(
                'fiktor/FIM-Estimation')
        config_names = {}
        re_pattern = re.compile(rf"^{sm_name}\.seed(\d+)\.gt_fim$")
        for config_name in all_config_names:
            match = re_pattern.match(config_name)
            if match:
                seed = int(match.group(1))
                config_names[seed] = config_name
        if not config_names and f"{sm_name}.gt_fim" in all_config_names:
            warnings.warn(
                f"Per-seed gt_fim is not available for {sm_name}, "
                "but there is a single gt_fim. Did you mean to "
                "call `load_gt_fim` instead of `load_gt_fims`?")
    else:
        assert isinstance(config_names, dict)
    gt_fims = {}
    gt_fim_mgrids = {}
    for seed, config_name in config_names.items():
        gt_fims[seed], gt_fim_mgrids[seed] = load_gt_fim(config_name)
    return gt_fims, gt_fim_mgrids

