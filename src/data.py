"""
Dataset loading and bipartite graph construction.

Datasets have naturally occurring missing values from UCI and other sources.
"""

import os
import glob
import subprocess
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

try:
    from ucimlrepo import fetch_ucirepo
    _HAS_UCI = True
except ImportError:
    _HAS_UCI = False

try:
    from nhanes.load import load_NHANES_data
    _HAS_NHANES = True
except ImportError:
    _HAS_NHANES = False


class DatasetLoader:
    """Load datasets with naturally occurring missingness."""

    DATASETS = ['annealing', 'hepatitis', 'soybean', 'thyroid',
                'voting', 'physionet_sepsis', 'nhanes']

    @staticmethod
    def load(name):
        """Load dataset, returns (X, y) with X potentially containing NaN."""
        loaders = {
            'annealing': DatasetLoader._annealing,
            'hepatitis': DatasetLoader._hepatitis,
            'voting': DatasetLoader._voting,
            'soybean': DatasetLoader._soybean,
            'thyroid': DatasetLoader._thyroid,
            'physionet_sepsis': DatasetLoader._physionet,
            'nhanes': DatasetLoader._nhanes,
        }
        if name not in loaders:
            raise ValueError(f'unknown dataset: {name}')
        X, y = loaders[name]()
        miss = np.isnan(X).mean()
        print(f'{name}: {X.shape[0]} samples, {X.shape[1]} features, '
              f'{miss:.1%} missing')
        return X, y

    @staticmethod
    def _encode_cat(df):
        """Encode categoricals, preserve NaN."""
        le = LabelEncoder()
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == 'object':
                mask = out[c].notna()
                if mask.any():
                    out.loc[mask, c] = le.fit_transform(out.loc[mask, c].astype(str))
                out[c] = pd.to_numeric(out[c], errors='coerce')
        return out

    @staticmethod
    def _annealing():
        """Steel annealing, ~65% missing."""
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/'
        cols = ['family', 'product-type', 'steel', 'carbon', 'hardness',
                'temper_rolling', 'condition', 'formability', 'strength',
                'non-ageing', 'surface-finish', 'surface-quality', 'enamelability',
                'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond',
                'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean',
                'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len',
                'oil', 'bore', 'packing', 'class']
        df1 = pd.read_csv(url + 'anneal.data', names=cols, na_values='?')
        df2 = pd.read_csv(url + 'anneal.test', names=cols, na_values='?')
        df = pd.concat([df1, df2], ignore_index=True)
        X = DatasetLoader._encode_cat(df.drop('class', axis=1))
        return X.values.astype(float), LabelEncoder().fit_transform(df['class'])

    @staticmethod
    def _hepatitis():
        """Hepatitis survival, ~5.7% missing."""
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
        cols = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE',
                'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE',
                'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK_PHOSPHATE',
                'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
        df = pd.read_csv(url, names=cols, na_values='?')
        X = DatasetLoader._encode_cat(df.drop('Class', axis=1))
        return X.values.astype(float), LabelEncoder().fit_transform(df['Class'])

    @staticmethod
    def _soybean():
        """Soybean disease, ~6.6% missing."""
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
        cols = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist',
                'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth',
                'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size',
                'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem', 'lodging',
                'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external-decay',
                'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit-spots',
                'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling',
                'roots', 'class']
        df = pd.read_csv(url, names=cols, na_values='?')
        X = DatasetLoader._encode_cat(df.drop('class', axis=1))
        return X.values.astype(float), LabelEncoder().fit_transform(df['class'])

    @staticmethod
    def _thyroid():
        """Thyroid disease, ~78% missing."""
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick-euthyroid.data'
        df = pd.read_csv(url, header=None)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1].replace('?', np.nan)
        # filter rare classes
        vc = y.value_counts()
        valid = vc[vc >= 10].index
        mask = y.isin(valid)
        X, y = X[mask], y[mask]
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        return X.values.astype(float), LabelEncoder().fit_transform(y)

    @staticmethod
    def _voting():
        """Congressional voting, ~5.6% missing. Requires ucimlrepo."""
        if not _HAS_UCI:
            raise ImportError('pip install ucimlrepo')
        data = fetch_ucirepo(id=105)
        X, y = data.data.features.copy(), data.data.targets
        le = LabelEncoder()
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = X[c].replace('?', np.nan)
                mask = X[c].notna()
                if mask.any():
                    X.loc[mask, c] = le.fit_transform(X.loc[mask, c].astype(str))
                X[c] = pd.to_numeric(X[c], errors='coerce')
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        return X.values.astype(float), le.fit_transform(y)

    @staticmethod
    def _physionet():
        """PhysioNet 2019 sepsis challenge, ~28% missing."""
        data_dir = 'physionet_sepsis_data'
        setA = os.path.join(data_dir, 'training_setA')
        setB = os.path.join(data_dir, 'training_setB')
        os.makedirs(setA, exist_ok=True)
        os.makedirs(setB, exist_ok=True)

        if not glob.glob(f'{setA}/*.psv') or not glob.glob(f'{setB}/*.psv'):
            print('  downloading physionet data...')
            for s in [setA, setB]:
                subprocess.run([
                    'aws', 's3', 'sync', '--no-sign-request',
                    f's3://physionet-open/challenge-2019/1.0.0/training/{os.path.basename(s)}/',
                    f'{s}/'
                ], check=True, capture_output=True, timeout=600)

        files = (glob.glob(f'{setA}/*.psv') + glob.glob(f'{setB}/*.psv'))[:5000]
        patients, labels = [], []
        for fp in files:
            try:
                df = pd.read_csv(fp, sep='|', na_values=['', ' ', 'NaN'])
                labels.append(int(df['SepsisLabel'].max()))
                df = df.drop(['SepsisLabel', 'HospAdmTime', 'ICULOS'],
                             axis=1, errors='ignore')
                patients.append(df.ffill().bfill().iloc[-1].values)
            except Exception:
                continue
        return np.array(patients, dtype=float), np.array(labels, dtype=int)

    @staticmethod
    def _nhanes():
        """NHANES health survey, ~39% missing. Requires nhanes package."""
        if not _HAS_NHANES:
            raise ImportError('pip install nhanes')
        df = load_NHANES_data()
        target = 'GeneralHealthCondition'
        df = df.dropna(subset=[target])
        good = ['Good', 'Very good', 'Excellent']
        y = df[target].isin(good).astype(int).values
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c not in ['SEQN', target]]
        return df[feat_cols].values.astype(float), y


def sample_to_graph(x, y, mask):
    """
    Convert single sample to bipartite graph.

    This is the GRAPE representation from [1]: graph has 1 observation
    node + d feature nodes, edges exist only for observed (non-NaN) features.
    """
    d = len(x)
    n_nodes = 1 + d

    # GRAPE: node features are one-hot identity vectors
    node_feat = np.zeros((n_nodes, d))
    node_feat[0, :] = 1.0  # observation node
    for i in range(d):
        node_feat[i + 1, i] = 1.0  # feature nodes

    # GRAPE: edges carry observed values as attributes
    edges, attrs = [], []
    for i in range(d):
        if not np.isnan(x[i]):
            edges.extend([[0, i + 1], [i + 1, 0]])
            attrs.extend([x[i], x[i]])

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32).unsqueeze(1)

    return Data(
        x=torch.FloatTensor(node_feat),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.LongTensor([y]),
        mask=torch.FloatTensor(mask.astype(float)),  # our extension: missingness pattern
        num_features=d)


class BipartiteGraphDataset(Dataset):
    """Wraps tabular data as bipartite graphs."""

    def __init__(self, X, y, masks):
        self.X, self.y, self.masks = X, y, masks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return sample_to_graph(self.X[idx], self.y[idx], self.masks[idx])


def collate_graphs(batch):
    """Collate bipartite graphs, handling mask concatenation."""
    batched = Batch.from_data_list(batch)
    if batch and hasattr(batch[0], 'num_features'):
        batched.num_features = batch[0].num_features
    if hasattr(batched, 'mask') and batched.mask is not None:
        masks = [g.mask for g in batch if hasattr(g, 'mask')]
        if masks:
            batched.mask = torch.cat(masks, dim=0)
    return batched
