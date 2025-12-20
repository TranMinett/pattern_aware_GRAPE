# Pattern-Aware GRAPE

This is an extensions to [GRAPE](https://snap.stanford.edu/grape/) that encode missingness patterns for tabular data with missing values.

Standard approaches either drop incomplete samples or impute missing values, ignoring the fact that which features are missing can itself be informative. For example, in clinical data, the tests a physician orders depend on patient presentation therefore the pattern of missingness could possibly carries diagnostic signal.

This code adds additional pattern encoders and tests to GRAPE's bipartite graph representation.

## Methods

| Method | Description |
|--------|-------------|
| `GRAPE-NoPattern` | Baseline, no pattern info |
| `GRAPE-*Impute` | Imputation + baseline |
| `GRAPE-RandomPattern` | Frozen random projection |
| `GRAPE-LearnedPattern` | Trainable MLP |
| `GRAPE-StatisticalPattern` | Handcrafted features |
| `GRAPE-HierarchicalPattern` | Bottleneck MLP |

## General Results

Averaged over 7 UCI datasets with natural missingness:

- +17% balanced accuracy vs baseline
- +22% F1-macro
- Random embeddings ≈ learned (0.650 vs 0.663)

Results vary by dataset—annealing sees +80%, voting only +4%. Again, sucessful application/improvement with this extension assumes that the additonal of the missingness pattern improves the data or contributes to prediction.

## Quick Install for Use

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py              # full eval
python main.py --fast       # quick test
python main.py --datasets annealing --methods GRAPE-LearnedPattern
```

```python
from src import GRAPEPatternAware, DatasetLoader

X, y = DatasetLoader.load('annealing')
model = GRAPEPatternAware(X.shape[1], hidden_dim=64, num_classes=5,
                          pattern_encoder='statistical')
```

## Structure

```
src/
  models.py   # GRAPEBase, GRAPEPatternAware
  data.py     # loaders, graph construction
  train.py    # training loop
  utils.py
config.py
main.py
```

## Datasets

All from UCI with natural missingness: annealing (65%), hepatitis (5.7%), soybean (6.6%), thyroid (78%), voting (5.6%), physionet_sepsis (28%), nhanes (39%).

## primary References

- You et al. "Handling Missing Data with Graph Representation Learning." NeurIPS 2020.
- Veličković et al. "Graph Attention Networks." ICLR 2018.
