from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn

ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']
DEGREES  = [0, 1, 2, 3, 4, 5]
CHARGES  = [-2, -1, 0, 1, 2]
NUM_HS   = [0, 1, 2, 3, 4]
HYBRIDS  = [Chem.HybridizationType.SP,  Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2]
BONDTYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
STEREOS   = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ,
             Chem.BondStereo.STEREOE]

def one_hot(value, vocab):
    """One-hot with a trailing 'other' slot. Length = len(vocab) + 1."""
    v = [0.0] * (len(vocab) + 1)
    v[vocab.index(value) if value in vocab else len(vocab)] = 1.0
    return v

def get_atom_features(atom):
    f = one_hot(atom.GetSymbol(), ELEMENTS)
    f += one_hot(atom.GetDegree(), DEGREES)
    f += one_hot(atom.GetFormalCharge(), CHARGES)
    f += one_hot(atom.GetTotalNumHs(), NUM_HS)
    f += one_hot(atom.GetHybridization(), HYBRIDS)
    f += [float(atom.GetIsAromatic())]
    f += [float(atom.IsInRing())]
    q = atom.GetDoubleProp('_GasteigerCharge')
    f += [0.0 if (q != q or abs(q) > 1e3) else q]     #  1  (NaN/inf guard)
    return f

ATOM_DIM = 39

def get_bond_features(bond):
    f = one_hot(bond.GetBondType(), BONDTYPES)
    f += [float(bond.GetIsConjugated())]
    f += [float(bond.IsInRing())]
    f += one_hot(bond.GetStereo(), STEREOS)
    return f

BOND_DIM = 11

def mol_to_graph(smiles):
    """Return dict with X (N,ATOM_DIM), A (N,N), E (N,N,BOND_DIM), n."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDkit cannot parse {smiles}")
    AllChem.ComputeGasteigerCharges(mol)
    n = mol.GetNumAtoms()
    X = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float64)
    A = torch.zeros(n, n, dtype=torch.float64)
    E = torch.zeros(n, n, BOND_DIM, dtype=torch.float64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0
        bf = torch.tensor(get_bond_features(b), dtype=torch.float64)
        E[i, j] = E[j, i] = bf
    return {'X': X, 'A': A, 'E': E, 'n': n, 'smiles': smiles}


def normalized_adjacency(A):
    A_hat = A + torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * A_hat * d_inv_sqrt.unsqueeze(-2)


def collate(graphs):
    B, Nmax = len(graphs), max(g['n'] for g in graphs)
    X = torch.zeros(B, Nmax, ATOM_DIM, dtype=torch.float64)
    A = torch.zeros(B, Nmax, Nmax, dtype=torch.float64)
    mask = torch.zeros(B, Nmax, dtype=torch.float64)

    for b, g in enumerate(graphs):
        n = g['n']
        X[b, :n] = g['X']
        A[b, :n, :n] = g['A']
        mask[b, :n] = 1.0

    return X, normalized_adjacency(A), mask


def gcn_forward(S, H, W, b):
    M = S @ H
    Z = (M @ W) + b
    Hout = torch.relu(Z)
    return Hout, (S, H, W, M, Z)

def gcn_backward(dHout, cache):
    """Hand-derived gradients of §3. dHout is dL/dHout, shape (B,N,F')."""
    S, H, W, M, Z = cache
    dZ = dHout * (Z > 0).to(Z.dtype)              # Delta = G * relu'(Z)
    dW = torch.einsum('bnf,bng->fg', M, dZ)       # (SH)^T Delta, summed over batch
    db = dZ.sum(dim=(0, 1))                       # column sum of Delta
    dM = dZ @ W.transpose(-1, -2)                 # Delta W^T
    dH = S.transpose(-1, -2) @ dM                 # S^T Delta W^T
    return dH, dW, db

class GCNRegressor(nn.Module):
    def __init__(self, in_dims=ATOM_DIM, hidden_dims=64, mlp_hidden=64):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dims, hidden_dims, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.zeros(hidden_dims, dtype=torch.float64))

        self.W2 = nn.Parameter(torch.empty(hidden_dims, hidden_dims, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(hidden_dims, dtype=torch.float64))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims, mlp_hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1, dtype=torch.float64)
        )

    def forward(self, X, S, mask):
        H, _ = gcn_forward(S, X, self.W1, self.b1)
        H, _ = gcn_forward(S, H, self.W2, self.b2)
        H = H * mask.unsqueeze(-1)
        z = H.sum(1)/mask.sum(1, keepdim=True)
        return self.head(z).squeeze(-1)
    