"""
Ligand Views Demo

Demonstrates the container-style interface for ligand representations:
graph, fingerprint, surface, and SMILES.
"""

from plmol.mol.ligand import Ligand


def main() -> None:
    ligand = Ligand.from_smiles("CCO")
    ligand.generate_conformer()

    print(f"SMILES: {ligand.smiles}")

    graph = ligand.featurize("graph")["graph"]
    print(f"Graph nodes: {graph['node']['node_feats'].shape}")
    print(f"Graph edges: {graph['edge']['edge_feats'].shape}")

    fingerprint = ligand.featurize("fingerprint")["fingerprint"]
    print(f"Morgan fingerprint: {fingerprint.shape}")

    try:
        surface = ligand.featurize(
            "surface",
            surface_kwargs={"grid_density": 2.5, "threshold": 0.5, "sharpness": 1.5},
            generate_conformer=True,
        )["surface"]
        if surface is None:
            print("Surface extraction failed (returned None).")
        else:
            print(f"Surface verts: {surface['verts'].shape}")
            print(f"Surface faces: {surface['faces'].shape}")
    except Exception as exc:
        print(f"Surface extraction skipped: {exc}")


if __name__ == "__main__":
    main()

