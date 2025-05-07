# -*- coding: utf-8 -*-

from pathlib import Path

from simmate.database.base_data_types import (
    Calculation,
    DatabaseTable,
    Structure,
    table_column,
)

class ElfAnalysis(Structure, Calculation):
    """
    This table contains results from an ELF topology analysis.
    """
    separate_spin = table_column.BooleanField(blank=True, null=True)
    """
    Whether or not this calculation tried to separate spin-up and spin-down
    electrons.
    """
    
    spin_polarized = table_column.BooleanField(blank=True, null=True)
    """
    If separate spin was requested, whether or not the ELF was found to 
    be spin polarized.
    """
    
    ignore_low_pseuodopotentials = table_column.BooleanField(blank=True, null=True)
    """
    Whether or not this calculation ignored systems with missing core
    electrons.
    """
    
    downscale_resolution = table_column.FloatField(blank=True, null=True)
    """
    The resolution the ELF and charge-density grids were reduced to if
    they were a higher resolution. None if no downscaling was used.
    """
    
    min_covalent_charge = table_column.FloatField(blank=True, null=True)
    """
    The minimum charge required for a site to be considered a covalent
    bond. This prevents metallic features nearly along atom bonds with low
    charges from being classified as covalent.
    """
    
    min_covalent_angle = table_column.FloatField(blank=True, null=True)
    """
    The minimum angle between neighboring atoms and a non-atomic feature
    above which the feature will be considered covalent.
    """
    
    min_covalent_bond_ratio = table_column.FloatField(blank=True, null=True)
    """
    The minimum ratio along a bond between neighbors, above which a feature
    will be considered covalent. Features below this ratio are typically
    assigned as lone-pairs or shells.
    """
    
    shell_depth = table_column.FloatField(blank=True, null=True)
    """
    The depth under which an atomic feature is considered to be an
    atom shell.
    """
    
    electride_elf_min = table_column.FloatField(blank=True, null=True)
    """
    The minimum ELF required for a non-atomic feature to be considered
    an electride
    """
    
    electride_depth_min = table_column.FloatField(blank=True, null=True)
    """
    The minimum depth required for a non-atomic feature to be considered
    an electride
    """
    
    electride_charge_min = table_column.FloatField(blank=True, null=True)
    """
    The minimum charge required for a non-atomic feature to be considered
    an electride
    """
    
    electride_volume_min = table_column.FloatField(blank=True, null=True)
    """
    The minimum volume required for a non-atomic feature to be considered
    an electride
    """
    
    electride_radius_min = table_column.FloatField(blank=True, null=True)
    """
    The minimum radius required for a non-atomic feature to be considered
    an electride. The radius is defined as the distance between the feature
    and the nearest atom minus that atoms ELF radius.
    """
    
    radius_refine_method = table_column.CharField(
        blank=True,
        null=True,
        max_length=75,
    )
    """
    The interpolation method used to determine the radii of atoms in
    the system.
    """
    
    bifurcation_graph_up = table_column.JSONField(blank=True, null=True)
    """
    The bifurcation graph representing where features appear and connect
    in the spin-up ELF
    """

    bifurcation_graph_down = table_column.JSONField(blank=True, null=True)
    """
    The bifurcation graph representing where features appear and connect
    in the spin-down ELF
    """
    
    labeled_structure_up = table_column.JSONField(blank=True, null=True)
    """
    The labeled structure containing information only for the spin-up
    ELF and charge density. Features are represented with the following
    dummy atoms
    
    E: electride, Le: non-electridic bare electron, Lp: lone-pair, 
    Z: covalent bond, M: metallic feature
    """

    labeled_structure_down = table_column.JSONField(blank=True, null=True)
    """
    The labeled structure containing information only for the spin-down
    ELF and charge density. Features are represented with the following
    dummy atoms
    
    E: electride, Le: non-electridic bare electron, Lp: lone-pair, 
    Z: covalent bond, M: metallic feature
    """
    
    def write_output_summary(self, directory: Path):
        super().write_output_summary(directory)

    def update_from_directory(self, directory):
        """
        The base database workflow will try and register data from the local
        directory. As part of this it checks for a vasprun.xml file and
        attempts to run a from_vasp_run method. Since this is not defined for
        this model, an error is thrown. To account for this, I just create an empty
        update_from_directory method here.
        """
        pass

    def update_elf_features(self, features: list):
        # pull all the data together and save it to the database. We
        # are saving this to an ElfIonicRadii datatable. To access this
        # model, we need to use "elf_features.model".
        feature_model = self.elf_features.model
        # Let's iterate through the ELF features and save these to the database.
        for feature in features:
            if feature.get("split", None) is not None:
                continue # skips reducible domains
            new_row = feature_model(
                spin = feature.get("spin", None),
                basin_type = feature.get("type", None),
                basin_subtype = feature.get("subtype", None),
                reducible = feature.get("reducible", None),
                basins = feature.get("basins", None),
                frac_coords = feature.get("frac_coords", None),
                max_elf = feature.get("max_elf", None),
                depth = feature.get("depth", None),
                depth_3d = feature.get("3d_depth", None),
                charge = feature.get("charge", None),
                volume = feature.get("volume", None),
                nearest_atom = feature.get("nearest_atom", None),
                nearest_atom_type = feature.get("nearest_atom_type", None),
                atom_distance = feature.get("atom_distance", None),
                feature_radius = feature.get("feature_radius", None),
                dist_beyond_atom = feature.get("dist_beyond_atom", None),
                bare_electron_indicator = feature.get("bare_electron_indicator", None),
                bare_electron_scores = feature.get("bare_electron_scores", None),
                coord_number = feature.get("coord_num"),
                coord_atom_indices = feature.get("coord_indices"),
                coord_atom_types = feature.get("coord_atoms"),
                feature_structure_index = feature.get("structure_index"),
                elf_analysis = self, # links to elf analysis calc
            )
            new_row.save()

class ElfFeatures(DatabaseTable):
    """
    This table contains the elf features calculated during an elf analysis
    calculation
    """
    elf_analysis = table_column.ForeignKey(
        "ElfAnalysis",
        on_delete=table_column.CASCADE,
        related_name="elf_features",
    )
    
    ###########################################################################
    # Columns for all basin types including atomic and valent
    ###########################################################################
    
    spin = table_column.CharField(
        blank=True,
        null=True,
        max_length=75,
    )
    """
    Which spin system this feature was found in. None if the spin was
    not separated
    """
    
    basin_type = table_column.CharField(
        blank=True,
        null=True,
        max_length=75,
    )
    """
    The type of basin, either atomic or valence
    """
    
    basin_subtype = table_column.CharField(
        blank=True,
        null=True,
        max_length=75,
    )
    """
    The subtype of the basin. For example atomic basins might be cores
    or shells. Valence basins might be covalent, metallic, lone-pairs,
    or a bare electron.
    """
    
    reducible = table_column.JSONField(blank=True, null=True)
    """
    Only for atom shells. True if this shell splits into multiple basins 
    near its maximum ELF
    """
    
    basins = table_column.JSONField(blank=True, null=True)
    """
    The basins that are in this feature. This is often more than one
    for features such as atom shells.
    """
    
    frac_coords = table_column.JSONField(blank=True, null=True)
    """
    The fractional coordinates of this feature. For features that are
    not point attractors, this is just one of the possible fractional
    coordinates. For example, atom shells often contain multiple maxima
    in the grid, and this only reflects one of these.
    """
    
    max_elf = table_column.FloatField(blank=True, null=True)
    """
    The max elf value in this basin
    """
    
    depth = table_column.FloatField(blank=True, null=True)
    """
    The depth of this feature defined as the difference in the maximum ELF
    to the ELF value at which the feature bifurcated from a larger domain.
    """
    
    depth_3d = table_column.FloatField(blank=True, null=True)
    """
    The depth of this feature defined as the difference between the
    maximum ELF of the feature to the ELF at which it connects to an
    ELF domain extending infinitely
    """
    
    charge = table_column.FloatField(blank=True, null=True)
    """
    The charge contained in this feature
    """
    
    volume = table_column.FloatField(blank=True, null=True)
    """
    The volume of this feature
    """
    
    nearest_atom = table_column.FloatField(blank=True, null=True)
    """
    The index of the nearest atom to this feature
    """
    
    nearest_atom_type = table_column.CharField(
        blank=True,
        null=True,
        max_length=75,
    )
    """
    The type of atom that is closest to this feature
    """
    
    atom_distance = table_column.FloatField(blank=True, null=True)
    """
    The distance from this feature to the nearest atom
    """
    
    
    ###########################################################################
    # Columns only filled out for valence features
    ###########################################################################
    
    feature_structure_index = table_column.IntegerField(blank=True, null=True)
    """
    The index of the labeled structure that this feature corresponds to.
    """
    
    feature_radius = table_column.FloatField(blank=True, null=True)
    """
    The distance from the maximum of this feature to the nearest point
    on the partitioning surface.
    """
    
    dist_beyond_atom = table_column.FloatField(blank=True, null=True)
    """
    The distance from this feature to the neighboring atom minus that atoms
    radius determined from the ELF.
    """
    
    bare_electron_indicator = table_column.FloatField(blank=True, null=True)
    """
    A measurement of how "bare" an electron is based on the features
    ELF max, depth, charge, volume, and radius.
    """
    
    bare_electron_scores = table_column.JSONField(blank=True, null=True)
    """
    The scores for each descriptor of how "bare" an electron is. The
    descriptors each range from 0 to 1 and are related to the ELF max,
    depth, charge, volume, and radius.
    
    Each one is compared to an ideal value
    """
    
    coord_number = table_column.IntegerField(blank=True, null=True)
    """
    The coordination number of this feature
    """
    
    coord_atom_indices = table_column.JSONField(blank=True, null=True)
    """
    The structure indices of each of the coordinated atoms
    """
    
    coord_atom_types = table_column.JSONField(blank=True, null=True)
    """
    The symbol of each of the coordinated atoms
    """
    