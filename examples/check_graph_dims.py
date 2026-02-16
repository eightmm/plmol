"""Quick validation script for graph feature dimensionality."""

from rdkit import Chem

from plmol.featurizers.graph_featurizer import MoleculeGraphFeaturizer


def main() -> None:
    smiles_list = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
    ]

    featurizer = MoleculeGraphFeaturizer()

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            continue

        node, edge, adj = featurizer.featurize(mol)
        node_shape = tuple(node["node_feats"].shape)
        edge_shape = tuple(edge["edge_feats"].shape)
        adj_shape = tuple(adj.shape)

        print(
            f"{smiles} -> node_feats {node_shape}, edge_feats {edge_shape}, adj {adj_shape}"
        )


if __name__ == "__main__":
    main()
