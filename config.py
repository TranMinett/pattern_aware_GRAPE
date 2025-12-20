"""Configuration."""

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

METHODS = [
    'GRAPE-NoPattern',
    'GRAPE-MeanImpute',
    'GRAPE-MedianImpute',
    'GRAPE-KNNImpute',
    'GRAPE-MICEImpute',
    'GRAPE-RandomPattern',
    'GRAPE-LearnedPattern',
    'GRAPE-StatisticalPattern',
    'GRAPE-HierarchicalPattern',
]

DATASETS = [
    'annealing', 'hepatitis', 'soybean', 'thyroid',
    'voting', 'physionet_sepsis', 'nhanes',
]

FAST = {
    'datasets': ['annealing', 'hepatitis', 'soybean'],
    'methods': ['GRAPE-NoPattern', 'GRAPE-MeanImpute', 'GRAPE-KNNImpute',
                'GRAPE-RandomPattern', 'GRAPE-LearnedPattern'],
    'seeds': [42, 123, 456],
    'epochs': 100,
}

FULL = {
    'datasets': DATASETS,
    'methods': METHODS,
    'seeds': [1, 2, 3, 4, 5],
    'epochs': 150,
}

MODEL = {
    'hidden_dim': 64,
    'batch_size': 32,
    'patience': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.2,
    'edge_dropout': 0.3,
}


def get_config(fast=False):
    preset = FAST if fast else FULL
    return {**MODEL, **preset, 'device': DEVICE, 'save_dir': 'results'}
