"""Runtime/default constants for APIs, IO and execution settings."""

# Ligand IO
IO_SUPPORTED_LIGAND_EXTENSIONS = [".sdf", ".mol2", ".mol", ".pdb"]

# Protein standardization
PTM_HANDLING_MODES = ("base_aa", "unk", "preserve", "remove")

# Graph defaults
DEFAULT_ATOM_GRAPH_DISTANCE_CUTOFF = 4.0
DEFAULT_RESIDUE_GRAPH_DISTANCE_CUTOFF = 8.0

# Surface defaults
SURFACE_DEFAULT_CURVATURE_SCALES = (1.0, 2.0, 3.0, 5.0, 10.0)
SURFACE_DEFAULT_KNN_ATOMS = 16

# Point cloud defaults
SURFACE_DEFAULT_POINTS_PER_ATOM = 100
SURFACE_DEFAULT_PROBE_RADIUS = 1.4
SURFACE_MIN_POINTS_PER_ATOM = 12
SURFACE_MAX_POINTS_RATIO = 3

# Voxel defaults
VOXEL_DEFAULT_RESOLUTION = 1.0          # Angstrom per voxel
VOXEL_DEFAULT_BOX_SIZE = 24             # grid dimension (24 = 24A cube at 1A resolution)
VOXEL_DEFAULT_PADDING = 4.0             # Angstrom padding around molecule
VOXEL_DEFAULT_SIGMA_SCALE = 1.0         # sigma = VdW_radius * sigma_scale
VOXEL_DEFAULT_CUTOFF_SIGMA = 2.5        # Gaussian cutoff at 2.5 sigma

# Cavity detection (LIGSITE-style grid scan)
CAVITY_GRID_RESOLUTION = 1.0            # Angstrom per grid point
CAVITY_PADDING = 4.0                    # Angstrom of empty grid around the structure
CAVITY_PROBE_RADIUS = 1.4               # solvent probe; a point within vdw + this is protein
CAVITY_SCAN_LENGTH = 8.0                # Angstrom a ray looks for the far wall
CAVITY_PSP_THRESHOLD = 5                # of 7 directions enclosed; 3 also catches open clefts
CAVITY_MIN_POINTS = 10                  # smaller clusters are noise, not pockets
CAVITY_LINING_MARGIN = 1.0              # Angstrom past vdw + probe for a lining atom

# Backbone graph defaults
DEFAULT_BACKBONE_KNN_NEIGHBORS = 30

# Pocket extraction internals
# Pure heavy atoms only (TRP = 14); differs from amino_acids.MAX_ATOMS_PER_RESIDUE (15)
# which includes a virtual sidechain centroid slot.
POCKET_MAX_ATOMS_PER_RESIDUE = 14
