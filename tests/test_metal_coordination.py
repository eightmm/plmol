"""
Tests for metal coordination detection and feature encoding.
"""

import numpy as np
import pytest

from plmol.interaction.metal_coordination import (
    MetalSite,
    detect_metal_sites,
    classify_coordination_geometry,
    encode_metal_features,
)
from plmol.constants.interactions import (
    METAL_COORDINATION_CUTOFF,
    COMMON_COORDINATION_GEOMETRIES,
    METAL_PREFERRED_DONORS,
)


# =============================================================================
# Fixtures: synthetic metal sites
# =============================================================================

def make_tetrahedral_zinc():
    """Zinc at origin with 4 donors at ~2.1 Å in tetrahedral arrangement."""
    # Tetrahedral vertices around origin
    d = 2.1
    donors = np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ], dtype=float)
    donors = donors / np.linalg.norm(donors[0]) * d

    metal_coord = np.array([0.0, 0.0, 0.0])
    all_coords = np.vstack([[metal_coord], donors])  # index 0 = Zn

    metal_meta = {'atom_name': 'ZN', 'res_name': 'ZN', 'chain_id': 'A', 'element': 'ZN'}
    donor_metas = [
        {'atom_name': 'NE2', 'res_name': 'HIS', 'chain_id': 'A', 'element': 'N'},
        {'atom_name': 'ND1', 'res_name': 'HIS', 'chain_id': 'A', 'element': 'N'},
        {'atom_name': 'OD1', 'res_name': 'ASP', 'chain_id': 'A', 'element': 'O'},
        {'atom_name': 'OE1', 'res_name': 'GLU', 'chain_id': 'A', 'element': 'O'},
    ]
    metadata = [metal_meta] + donor_metas
    return all_coords, metadata, [0]


def make_octahedral_magnesium():
    """Magnesium at origin with 6 oxygen donors at 2.1 Å in octahedral arrangement."""
    d = 2.1
    donors = np.array([
        [ d,  0,  0],
        [-d,  0,  0],
        [ 0,  d,  0],
        [ 0, -d,  0],
        [ 0,  0,  d],
        [ 0,  0, -d],
    ], dtype=float)

    metal_coord = np.array([0.0, 0.0, 0.0])
    all_coords = np.vstack([[metal_coord], donors])

    metal_meta = {'atom_name': 'MG', 'res_name': 'MG', 'chain_id': 'A', 'element': 'MG'}
    donor_metas = [
        {'atom_name': f'O{i}', 'res_name': 'HOH', 'chain_id': 'A', 'element': 'O'}
        for i in range(6)
    ]
    metadata = [metal_meta] + donor_metas
    return all_coords, metadata, [0]


# =============================================================================
# Tests: classify_coordination_geometry
# =============================================================================

class TestClassifyCoordinationGeometry:
    def test_linear_2_donors(self):
        metal = np.array([0.0, 0.0, 0.0])
        donors = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        assert classify_coordination_geometry(donors, metal) == "linear"

    def test_trigonal_planar_3_donors(self):
        metal = np.array([0.0, 0.0, 0.0])
        angle = 2 * np.pi / 3
        donors = np.array([
            [2.0 * np.cos(0), 2.0 * np.sin(0), 0],
            [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0],
            [2.0 * np.cos(2 * angle), 2.0 * np.sin(2 * angle), 0],
        ])
        assert classify_coordination_geometry(donors, metal) == "trigonal_planar"

    def test_tetrahedral_4_donors(self):
        metal = np.array([0.0, 0.0, 0.0])
        d = 2.1
        donors = np.array([
            [ 1,  1,  1],
            [-1, -1,  1],
            [-1,  1, -1],
            [ 1, -1, -1],
        ], dtype=float)
        donors = donors / np.linalg.norm(donors[0]) * d
        result = classify_coordination_geometry(donors, metal)
        assert result == "tetrahedral"

    def test_square_planar_4_donors(self):
        metal = np.array([0.0, 0.0, 0.0])
        d = 2.0
        donors = np.array([
            [ d,  0, 0],
            [-d,  0, 0],
            [ 0,  d, 0],
            [ 0, -d, 0],
        ], dtype=float)
        result = classify_coordination_geometry(donors, metal)
        assert result == "square_planar"

    def test_octahedral_6_donors(self):
        metal = np.array([0.0, 0.0, 0.0])
        d = 2.1
        donors = np.array([
            [ d,  0,  0],
            [-d,  0,  0],
            [ 0,  d,  0],
            [ 0, -d,  0],
            [ 0,  0,  d],
            [ 0,  0, -d],
        ], dtype=float)
        result = classify_coordination_geometry(donors, metal)
        assert result == "octahedral"

    def test_trigonal_bipyramidal_5_donors(self):
        """5 donors: 3 equatorial at 120° + 2 axial at 180°."""
        metal = np.array([0.0, 0.0, 0.0])
        d = 2.0
        angle = 2 * np.pi / 3
        donors = np.array([
            [d * np.cos(0),       d * np.sin(0),       0],
            [d * np.cos(angle),   d * np.sin(angle),   0],
            [d * np.cos(2*angle), d * np.sin(2*angle), 0],
            [0, 0,  d],
            [0, 0, -d],
        ], dtype=float)
        result = classify_coordination_geometry(donors, metal)
        assert result == "trigonal_bipyramidal"

    def test_single_donor_unknown(self):
        metal = np.array([0.0, 0.0, 0.0])
        donors = np.array([[2.0, 0.0, 0.0]])
        result = classify_coordination_geometry(donors, metal)
        assert result == "unknown"


# =============================================================================
# Tests: detect_metal_sites
# =============================================================================

class TestDetectMetalSites:
    def test_tetrahedral_zinc_detection(self):
        coords, metadata, metal_indices = make_tetrahedral_zinc()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        assert len(sites) == 1
        site = sites[0]
        assert site.metal_element == 'ZN'
        assert site.coordination_number == 4
        assert site.geometry == 'tetrahedral'

    def test_octahedral_magnesium_detection(self):
        coords, metadata, metal_indices = make_octahedral_magnesium()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        assert len(sites) == 1
        site = sites[0]
        assert site.metal_element == 'MG'
        assert site.coordination_number == 6
        assert site.geometry == 'octahedral'

    def test_coordinating_atom_info(self):
        coords, metadata, metal_indices = make_tetrahedral_zinc()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        site = sites[0]
        for atom in site.coordinating_atoms:
            assert 'atom_name' in atom
            assert 'res_name' in atom
            assert 'chain_id' in atom
            assert 'coords' in atom
            assert 'distance' in atom
            assert atom['distance'] < METAL_COORDINATION_CUTOFF + 0.1

    def test_empty_inputs(self):
        sites = detect_metal_sites(np.zeros((0, 3)), [], [])
        assert sites == []

    def test_no_metal_indices(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        metadata = [
            {'atom_name': 'CA', 'res_name': 'ALA', 'chain_id': 'A', 'element': 'C'},
            {'atom_name': 'N', 'res_name': 'ALA', 'chain_id': 'A', 'element': 'N'},
        ]
        sites = detect_metal_sites(coords, metadata, [])
        assert sites == []

    def test_distance_cutoff_respected(self):
        """Donors at 5.0 Å should not be detected with default 2.8 Å cutoff."""
        d = 5.0
        coords = np.array([
            [0.0, 0.0, 0.0],
            [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0],
        ])
        metadata = [
            {'atom_name': 'ZN', 'res_name': 'ZN', 'chain_id': 'A', 'element': 'ZN'},
        ] + [
            {'atom_name': 'N', 'res_name': 'HIS', 'chain_id': 'A', 'element': 'N'}
        ] * 4
        sites = detect_metal_sites(coords, metadata, [0])
        assert len(sites) == 0

    def test_preferred_donor_filtering(self):
        """MG should only accept O donors, not N."""
        d = 2.1
        coords = np.array([
            [0.0, 0.0, 0.0],   # MG
            [d, 0, 0],          # O - accepted
            [-d, 0, 0],         # N - rejected for MG
        ])
        metadata = [
            {'atom_name': 'MG', 'res_name': 'MG', 'chain_id': 'A', 'element': 'MG'},
            {'atom_name': 'O1', 'res_name': 'HOH', 'chain_id': 'A', 'element': 'O'},
            {'atom_name': 'N1', 'res_name': 'HIS', 'chain_id': 'A', 'element': 'N'},
        ]
        sites = detect_metal_sites(coords, metadata, [0])
        assert len(sites) == 1
        assert sites[0].coordination_number == 1


# =============================================================================
# Tests: encode_metal_features
# =============================================================================

class TestEncodeMetalFeatures:
    def test_empty_sites(self):
        features = encode_metal_features([], n_residues=10)
        assert features['metal_type_onehot'].shape == (0, len(METAL_PREFERRED_DONORS))
        assert features['coordination_number'].shape == (0,)
        assert features['geometry_encoding'].shape == (0, len(COMMON_COORDINATION_GEOMETRIES))
        assert features['distance_stats'].shape == (0, 3)
        assert features['n_metal_sites'] == 0

    def test_zinc_site_features(self):
        coords, metadata, metal_indices = make_tetrahedral_zinc()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        features = encode_metal_features(sites, n_residues=50)

        assert features['n_metal_sites'] == 1
        assert features['metal_type_onehot'].shape == (1, len(METAL_PREFERRED_DONORS))
        assert features['coordination_number'].shape == (1,)
        assert features['geometry_encoding'].shape == (1, len(COMMON_COORDINATION_GEOMETRIES))
        assert features['distance_stats'].shape == (1, 3)

        # Check ZN one-hot
        metal_types = list(METAL_PREFERRED_DONORS.keys())
        zn_idx = metal_types.index('ZN')
        assert features['metal_type_onehot'][0, zn_idx] == 1.0
        assert features['metal_type_onehot'][0].sum() == 1.0

    def test_geometry_encoding_is_one_hot(self):
        coords, metadata, metal_indices = make_octahedral_magnesium()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        features = encode_metal_features(sites, n_residues=50)

        geom_enc = features['geometry_encoding'][0]
        assert geom_enc.sum() == 1.0
        oct_idx = COMMON_COORDINATION_GEOMETRIES.index('octahedral')
        assert geom_enc[oct_idx] == 1.0

    def test_coordination_number_correct(self):
        coords, metadata, metal_indices = make_tetrahedral_zinc()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        features = encode_metal_features(sites, n_residues=50)
        assert features['coordination_number'][0] == 4

    def test_distance_stats_shapes_and_ordering(self):
        coords, metadata, metal_indices = make_tetrahedral_zinc()
        sites = detect_metal_sites(coords, metadata, metal_indices)
        features = encode_metal_features(sites, n_residues=50)

        dmin, dmax, dmean = features['distance_stats'][0]
        assert dmin <= dmean <= dmax
        assert dmin > 0.0

    def test_multiple_sites(self):
        """Two metal sites in a single encode call."""
        coords_zn, meta_zn, _ = make_tetrahedral_zinc()
        coords_mg, meta_mg, _ = make_octahedral_magnesium()

        # Offset MG site so it doesn't overlap
        offset = np.array([20.0, 0.0, 0.0])
        coords_mg = coords_mg + offset

        all_coords = np.vstack([coords_zn, coords_mg])
        all_meta = meta_zn + meta_mg
        zn_idx = [0]
        mg_idx = [len(coords_zn)]

        sites_zn = detect_metal_sites(coords_zn, meta_zn, zn_idx)
        sites_mg = detect_metal_sites(coords_mg, meta_mg, [0])
        all_sites = sites_zn + sites_mg

        features = encode_metal_features(all_sites, n_residues=100)
        assert features['n_metal_sites'] == 2
        assert features['metal_type_onehot'].shape[0] == 2
        assert features['coordination_number'].shape[0] == 2


# =============================================================================
# Tests: constants
# =============================================================================

class TestConstants:
    def test_cutoff_value(self):
        assert METAL_COORDINATION_CUTOFF == 2.8

    def test_geometries_list(self):
        expected = [
            "linear", "trigonal_planar", "tetrahedral", "square_planar",
            "trigonal_bipyramidal", "square_pyramidal", "octahedral"
        ]
        assert COMMON_COORDINATION_GEOMETRIES == expected

    def test_preferred_donors_keys(self):
        expected_metals = {'ZN', 'MG', 'CA', 'FE', 'CU', 'MN', 'CO', 'NI', 'NA', 'K'}
        assert set(METAL_PREFERRED_DONORS.keys()) == expected_metals

    def test_preferred_donors_are_sets(self):
        for metal, donors in METAL_PREFERRED_DONORS.items():
            assert isinstance(donors, set), f"{metal} donors should be a set"
            assert all(isinstance(d, str) for d in donors)
