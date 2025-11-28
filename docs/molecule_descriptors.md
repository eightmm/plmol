# Molecular Descriptors

## Overview
The molecular featurizer extracts 40 descriptors covering physicochemical, topological, and structural properties.

## Descriptor Categories

### 📊 Basic Physicochemical Properties (12 descriptors)
| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `mw` | Molecular weight | /1000.0 |
| `logp` | Octanol-water partition coefficient | (+5)/10.0 |
| `tpsa` | Topological polar surface area | /200.0 |
| `n_rotatable_bonds` | Number of rotatable bonds | /20.0 |
| `flexibility` | Ratio of rotatable bonds to total bonds | 0-1 |
| `hbd` | Hydrogen bond donors | /10.0 |
| `hba` | Hydrogen bond acceptors | /15.0 |
| `n_atoms` | Total number of atoms | /100.0 |
| `n_bonds` | Total number of bonds | /120.0 |
| `n_rings` | Number of rings | /10.0 |
| `n_aromatic_rings` | Number of aromatic rings | /8.0 |
| `heteroatom_ratio` | Ratio of heteroatoms to total atoms | 0-1 |

### 🔬 Topological & Complexity Indices (9 descriptors)
| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `balaban_j` | Balaban's J index (topological) | /5.0 |
| `bertz_ct` | Bertz complexity index | /2000.0 |
| `chi0` | Connectivity index 0 | /50.0 |
| `chi1` | Connectivity index 1 | /30.0 |
| `chi0n` | Normalized connectivity index | /50.0 |
| `hall_kier_alpha` | Hall-Kier alpha value | /5.0 |
| `kappa1` | Kappa shape index 1 | /50.0 |
| `kappa2` | Kappa shape index 2 | /20.0 |
| `kappa3` | Kappa shape index 3 | /10.0 |

### ⚡ Electronic & Surface Properties (4 descriptors)
| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `mol_mr` | Molar refractivity | /200.0 |
| `labute_asa` | LabuteASA (accessible surface area) | /500.0 |
| `num_radical_electrons` | Number of radical electrons | /5.0 |
| `num_valence_electrons` | Number of valence electrons | /500.0 |

### 🌐 Ring & Structural Features (8 descriptors)
| Descriptor | Description | Normalization |
|------------|-------------|---------------|
| `num_saturated_rings` | Number of saturated rings | /10.0 |
| `num_aliphatic_rings` | Number of aliphatic rings | /10.0 |
| `num_saturated_heterocycles` | Saturated heterocycles | /8.0 |
| `num_aliphatic_heterocycles` | Aliphatic heterocycles | /8.0 |
| `num_aromatic_heterocycles` | Aromatic heterocycles | /8.0 |
| `num_heteroatoms` | Total heteroatoms | /30.0 |
| `formal_charge` | Sum of formal charges | (+5)/10.0 |
| `n_ring_systems` | Number of ring systems | /8.0 |
| `max_ring_size` | Maximum ring size | /12.0 |
| `avg_ring_size` | Average ring size | /8.0 |

### 💊 Drug-likeness Properties (5 descriptors)
| Descriptor | Description | Range |
|------------|-------------|-------|
| `lipinski_violations` | Lipinski rule of 5 violations | 0-1 |
| `passes_lipinski` | Passes Lipinski rule | 0 or 1 |
| `qed` | Quantitative Estimate of Drug-likeness | 0-1 |
| `num_heavy_atoms` | Number of heavy atoms | /50.0 |
| `frac_csp3` | Fraction of sp3 carbons | 0-1 |

## Usage Example

```python
from plfeature import MoleculeFeaturizer

# Initialize featurizer
featurizer = MoleculeFeaturizer()

# Extract descriptors
features = featurizer.get_feature("CCO")  # ethanol
descriptors = features['descriptor']  # torch.Tensor of shape [40]

# Access specific descriptor values (after normalization)
mw = descriptors[0]  # Molecular weight
logp = descriptors[1]  # LogP
complexity = descriptors[13]  # BertzCT complexity
```

## Normalization Strategy

Descriptors are normalized to approximately [0, 1] range:
- **Division normalization**: Most descriptors divided by typical maximum values
- **Range shifting**: LogP and formal charge shifted to positive range before normalization
- **Direct ratios**: Some descriptors (flexibility, heteroatom_ratio) are naturally in [0,1]

