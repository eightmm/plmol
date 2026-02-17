"""
Protein/Ligand demo showing shared surface + graph interfaces.
"""
from pathlib import Path

from plmol import Ligand, Protein


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    protein_path = base_dir / "10gs_protein.pdb"
    ligand_path = base_dir / "10gs_ligand.sdf"

    protein = Protein.from_pdb(str(protein_path))
    ligand = Ligand.from_sdf(str(ligand_path))

    print("--- Graph Interfaces ---")
    protein_graph = protein.graph
    ligand_graph = ligand.graph
    print(f"Protein graph keys: {sorted(protein_graph.keys())}")
    print(f"Ligand graph keys: {sorted(ligand_graph.keys())}")

    print("\n--- Surface Interfaces ---")
    try:
        protein_surface = protein.surface
        if protein_surface is None:
            print("Protein surface: None (surface generation failed)")
        else:
            print(
                "Protein surface keys:",
                sorted([k for k in protein_surface.keys() if k in {"points", "normals", "verts"}]),
            )
    except ImportError as exc:
        print(f"Protein surface unavailable: {exc}")

    try:
        ligand_surface = ligand.surface
        if ligand_surface is None:
            print("Ligand surface: None (surface generation failed)")
        else:
            print(
                "Ligand surface keys:",
                sorted([k for k in ligand_surface.keys() if k in {"points", "normals", "verts"}]),
            )
    except (ImportError, ValueError) as exc:
        print(f"Ligand surface unavailable: {exc}")


if __name__ == "__main__":
    main()
