import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import torch
import torch.nn as nn


ELEMENTS = ["C", "N", "O", "F", "S", "Cl", "Br", "I", "P"]
N_FEATURES = len(ELEMENTS) + 1 + 2 # one-hot + other + degree + aromatic
DTYPE = torch.float32

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

def mol_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    rows = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        onehot = [float(sym == e) for e in ELEMENTS]
        onehot.append(float(sym not in ELEMENTS))
        rows.append(onehot + [atom.GetDegree(), atom.GetIsAromatic()])
    x = torch.tensor(rows, dtype=DTYPE)

    src, dst = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, edge_index, torch.tensor([y], dtype=DTYPE)


class GCNLayerSparse(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dtype=DTYPE):
        super().__init__()
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.empty(in_channels, out_channels, dtype=dtype))
        if bias:
            self.b = nn.Parameter(torch.empty(out_channels, dtype=dtype))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.b is not None:
            nn.init.zeros_(self.b)

    def forward(self, x, edge_index):
        N = x.size(0)

        # 1. Remove existing self loops, then add exactly one per node.
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        loops = torch.arange(N, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loops], dim=1)
        src, dst = edge_index

        # 2. In-degree at the receiving node, including the self loop (deg >= 1).
        deg = torch.zeros(N, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        d_inv_sqrt = deg.pow(-0.5)

        # 3. Symmetric normalization coefficient for each edge.
        coef = d_inv_sqrt[src] * d_inv_sqrt[dst]

        # 4. Transform, build messages, and sum them at the receiving node.
        h = x @ self.W
        msg = h[src] * coef.unsqueeze(-1)
        out = torch.zeros(N, self.out_channels, dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, msg)

        if self.b is not None:
            out = out + self.b
        return out


class GCNLayerDense(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_channels, out_channels, dtype=dtype))
        if bias:
            self.b = nn.Parameter(torch.empty(out_channels, dtype=dtype))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.b is not None:
            nn.init.zeros_(self.b)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index

        # Row = receiver, column = sender (same direction as the sparse layer).
        A = torch.zeros(N, N, dtype=x.dtype, device=x.device)
        A[dst, src] = 1.0
        A.fill_diagonal_(1.0)  # exactly one self loop per node -> A_tilde

        deg = A.sum(dim=1)  # in-degree including self loop
        d_inv_sqrt = deg.pow(-0.5)
        S = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

        out = S @ (x @ self.W)
        if self.b is not None:
            out = out + self.b
        return out
    
def collate(graphs):

    xs, eis, ys, batch = [], [], [], []
    offset = 0

    for g_idx, (x, ei, y) in enumerate(graphs):
        n = x.size(0)
        xs.append(x)
        eis.append(ei + offset)
        ys.append(y)
        batch.append(torch.full((n, ), g_idx, dtype=torch.long))
        offset += n
    return (
        torch.cat(xs, dim=0),
        torch.cat(eis, dim=1),
        torch.cat(batch, dim=0),
        torch.cat(ys, dim=0),
    )

def global_add_pool(h, batch, n_graphs):
    out = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype)
    return out.index_add_(0, batch, h)

def iterate_minibatches(graphs, batch_size, shuffle=True, seed=0):
    idx = np.arange(len(graphs))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(graphs), batch_size):
        yield collate([graphs[i] for i in idx[start: start + batch_size]])

class GCNConv(nn.Module):
    def __init__(self, in_dims, hidden_dims, n_layers):
        super().__init__()
        dims = [in_dims] + [hidden_dims] * n_layers

        self.convs = nn.ModuleList(
            GCNLayerSparse(dims[i], dims[i + 1]) for i in range(n_layers))

        self.read_out = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x, edge_index, batch, n_graphs):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        g = global_add_pool(h, batch, n_graphs)
        return self.read_out(g).squeeze(-1)
    
def train_full_batch(data, model, epochs=3000, lr=1e-2, log_every=250):
    x, ei, batch, y = data
    n_graphs = int(max(batch)) + 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for e in range(1, epochs + 1):
        opt.zero_grad()
        out = model(x, ei, batch, n_graphs)
        loss = ((y - out) ** 2).mean()
        loss.backward()
        opt.step()

        history.append(loss.detach().item())
        if e % log_every == 0 or e == 1:
            print(f"  epoch {e:4d}   mse {history[-1]:.4e}   "
                  f"rmse {history[-1] ** 0.5:.4e}")
    return history

