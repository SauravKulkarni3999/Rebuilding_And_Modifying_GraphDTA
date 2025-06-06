# -*- coding: utf-8 -*-
"""protein_utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f9admyaTtbBqoOoKjyNCBZcl2B797sGR
"""

# src/utils/protein_utils.py
import torch
from Bio.PDB import PDBParser
import numpy as np
from torch_geometric.data import Data

# Amino acid vocabulary for sequence encoding (GraphDTA baseline)
AMINO_ACIDS_SEQ = 'ACDEFGHIKLMNPQRSTVWY' # 20 standard amino acids
AA_TO_IDX_SEQ = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS_SEQ)} # 0 for padding/unknown
MAX_SEQ_LEN = 1000 # As used in notebooks

# Amino acid vocabulary for PDB parsing / graph node features (GraphDTA-3D)
# Mapping 3-letter PDB residue names to 1-letter codes and then to integer
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'UNK': 'X' # Unknown or non-standard
}
# For GraphDTA-3D, node features are often based on these 20 standard AAs
AMINO_ACIDS_GRAPH = 'ARNDCQEGHILKMFPSTWYV' # 20 standard
AA_TO_IDX_GRAPH = {aa: idx for idx, aa in enumerate(AMINO_ACIDS_GRAPH)} # 0 to 19


def sequence_to_tensor(seq, max_len=MAX_SEQ_LEN, aa_to_idx_map=None):
    """
    Converts protein sequence to an integer tensor with padding/truncation.
    Used for GraphDTA baseline (1D CNN on protein sequences).
    """
    if aa_to_idx_map is None:
        aa_to_idx_map = AA_TO_IDX_SEQ

    seq = seq[:max_len] # Trimming
    idxs = [aa_to_idx_map.get(aa, 0) for aa in seq] # Unknown AAs become 0 (padding index)

    # Padding
    # padded_idxs = idxs + [0] * (max_len - len(idxs))
    # The dataloader in GraphDTA_Baseline.ipynb does padding dynamically with pad_sequence.
    # So, we return variable length tensors here.
    return torch.tensor(idxs, dtype=torch.long)


def pdb_to_graph(pdb_path, dist_threshold=8.0, aa_to_idx_map=None):
    """
    Converts a PDB file to a PyTorch Geometric graph object for GraphDTA-3D.
    Nodes are C-alpha atoms of residues.
    Node features are one-hot encoded amino acid types (20 features).
    Edges are created if C-alpha distance is below dist_threshold.
    """
    if aa_to_idx_map is None:
        aa_to_idx_map = AA_TO_IDX_GRAPH

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_path}: {e}")
        return None

    ca_atoms = []
    node_features = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in THREE_TO_ONE: # Standard amino acids
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
                        one_letter_aa = THREE_TO_ONE[residue.get_resname()]
                        aa_idx = aa_to_idx_map.get(one_letter_aa, len(AMINO_ACIDS_GRAPH)) # Map to an 'unknown' index if not standard

                        # Create one-hot encoded feature vector (20 features for standard AAs)
                        feat = [0] * len(AMINO_ACIDS_GRAPH)
                        if one_letter_aa in aa_to_idx_map:
                            feat[aa_idx] = 1
                        node_features.append(feat)
        break # Assuming single model PDB

    if not ca_atoms:
        return None

    num_residues = len(ca_atoms)
    coords = np.array([atom.get_coord() for atom in ca_atoms])

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index_list = []
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < dist_threshold:
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])

    if not edge_index_list:
      edge_index = torch.empty((2,0), dtype=torch.long)
    else:
      edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, num_nodes=num_residues)


if __name__ == '__main__':
    # Test sequence_to_tensor
    seq = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQV"
    tensor_seq = sequence_to_tensor(seq)
    print(f"Sequence: {seq}")
    print(f"Tensor: {tensor_seq}")
    print(f"Tensor shape: {tensor_seq.shape}")

    # Placeholder for PDB to graph test if a sample PDB is available
    # Create a dummy PDB file for testing
    dummy_pdb_content = """
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      13.000  10.000  10.000  1.00  0.00           O
ATOM      5  N   GLY A   2      14.000  10.000  10.000  1.00  0.00           N
ATOM      6  CA  GLY A   2      15.000  10.000  10.000  1.00  0.00           C
ATOM      7  C   GLY A   2      16.000  10.000  10.000  1.00  0.00           C
ATOM      8  O   GLY A   2      17.000  10.000  10.000  1.00  0.00           O
ATOM      9  CA  CYS A   3      10.500  10.500  10.500  1.00  0.00           C
    """
    dummy_pdb_path = "dummy_protein.pdb"
    with open(dummy_pdb_path, "w") as f:
        f.write(dummy_pdb_content)

    protein_graph = pdb_to_graph(dummy_pdb_path, dist_threshold=8.0)
    if protein_graph:
        print(f"\nGenerated protein graph from dummy PDB: {protein_graph}")
        print(f"Node feature shape: {protein_graph.x.shape}") # Should be [num_residues, 20]
        print(f"Edge index shape: {protein_graph.edge_index.shape}")

    import os
    os.remove(dummy_pdb_path) # Clean up dummy file