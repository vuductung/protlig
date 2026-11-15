import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator

def load_dataset() -> pd.DataFrame:
    from skfp.datasets.moleculenet import load_esol
    return load_esol(data_dir='../data', as_frame=True)
# some health checks
def parse_check(df: pd.DataFrame):
    bad = [i for i, s in enumerate(df['SMILES']) if Chem.MolFromSmiles(s) is None]
    print(f"Unparseable SMILES: {len(bad)}")
    return bad


def duplicate_check(df: pd.DataFrame):
    canon = []

    for s in df['SMILES']:
        mol = Chem.MolFromSmiles(s)
        canon.append(mol if mol is not None else None)
    canon = pd.Series(canon)
    print(f'Number of duplicates: {canon.duplicated().sum()}')


def mean_predictor_rmse(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """RMSE of predicting the TRAINING mean for every test example."""
    c = y_train.mean()
    return float(np.sqrt(np.mean((y_test - c) ** 2)))

def make_generator(radius, n_bits):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

def featurize(smiles: list[str], generator):
    rows, keep = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        keep.append(mol is not None)
        if mol is not None:
            rows.append(generator.GetCountFingerprintAsNumPy(mol))

    rows, keep = np.stack(rows), np.stack(keep)
    return rows, keep
