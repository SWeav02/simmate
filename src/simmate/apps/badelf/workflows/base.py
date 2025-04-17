# -*- coding: utf-8 -*-

import os
import shutil

# This will be added back once I go through and handle warnings within context
# import warnings
from pathlib import Path
from typing import Literal

from simmate.apps.badelf.core.badelf import SpinBadElfToolkit, ElfAnalyzerToolkit
from simmate.engine import Workflow
from simmate.toolkit import Structure

class BadElfBase(Workflow):
    """
    Controls a Badelf analysis on a pre-ran VASP calculation.
    This is the base workflow that all analyses that run BadELF
    are built from. Note that for more in depth analysis, it may be more
    useful to use the BadElfToolkit class.
    """

    use_database = False

    @classmethod
    def run_config(
        cls,
        source: dict = None,
        directory: Path = None,
        find_electrides: bool = True,
        labeled_structure_up=None,
        labeled_structure_down=None,
        separate_spin=True,
        algorithm: Literal["badelf", "voronelf", "zero-flux"] = "badelf",
        shared_feature_algorithm: Literal["zero-flux", "voronoi"] = "zero-flux",
        shared_feature_separation_method: Literal[
            "plane", "pauling", "equal"
        ] = "pauling",
        elf_analyzer_kwargs: dict = dict(
            resolution=0.02,
            include_lone_pairs=False,
            include_shared_features=True,
            metal_depth_cutoff=0.1,
            min_covalent_angle=135,
            min_covalent_bond_ratio=0.4,
            shell_depth=0.05,
            electride_elf_min=0.5,
            electride_depth_min=0.2,
            electride_charge_min=0.5,
            electride_volume_min=10,
            electride_radius_min=0.3,
            radius_refine_method="linear",
        ),
        threads: int = None,
        ignore_low_pseudopotentials: bool = False,
        write_electride_files: bool = False,
        write_ion_radii: bool = True,
        write_labeled_structures: bool = True,
        run_id: str = None,
        **kwargs,
    ):
        # get cleaned labeled structures
        if labeled_structure_up is not None:
            labeled_structure_up = Structure.from_dynamic(labeled_structure_up)
        if labeled_structure_down is not None:
            labeled_structure_down = Structure.from_dynamic(labeled_structure_down)
        # make a new directory to run badelf algorithm in and copy necessary files.
        badelf_directory = directory / "badelf"
        try:
            os.mkdir(badelf_directory)
        except:
            pass
        files_to_copy = ["CHGCAR", "ELFCAR", "POTCAR"]
        for file in files_to_copy:
            shutil.copy(directory / file, badelf_directory)

        # Get the badelf toolkit object for running badelf.
        badelf_tools = SpinBadElfToolkit.from_files(
            directory=badelf_directory,
            find_electrides=find_electrides,
            algorithm=algorithm,
            separate_spin=separate_spin,
            labeled_structure_up=labeled_structure_up,
            labeled_structure_down=labeled_structure_down,
            threads=threads,
            shared_feature_algorithm=shared_feature_algorithm,
            ignore_low_pseudopotentials=ignore_low_pseudopotentials,
            elf_analyzer_kwargs=elf_analyzer_kwargs,
        )
        # run badelf.
        results = badelf_tools.results
        # write results
        if write_electride_files:
            badelf_tools.write_species_file()
            badelf_tools.write_species_file(file_type="CHGCAR")
        # grab the calculation table linked to this workflow run and save ionic
        # radii
        search_datatable = cls.database_table.objects.get(run_id=run_id)
        search_datatable.update_ionic_radii(badelf_tools.all_atom_elf_radii)
        # write ionic radii
        if write_ion_radii:
            badelf_tools.write_atom_elf_radii()
        if write_labeled_structures:
            badelf_tools.write_labeled_structures()
        badelf_tools.write_results_csv()
        # remove the ELFCAR and CHGCAR copies for space
        for file in files_to_copy:
            os.remove(directory / badelf_directory / file)
        
        return results

class ElfAnalysisBase(Workflow):
    """
    Controls an ELF topology analysis on a pre-ran VASP calculation.
    This is the base workflow that all analyses that run ELF analysis
    are built from. Note that for more in depth analysis, it may be more
    useful to use the ElfAnalyzerToolkit class in python.
    """

    use_database = False

    @classmethod
    def run_config(
        cls,
        source: dict = None,
        directory: Path = None,
        separate_spin=True,
        ignore_low_pseudopotentials: bool = False,
        resolution=0.01,
        downscale_resolution=1200,
        include_lone_pairs=False,
        include_shared_features=True,
        metal_depth_cutoff=0.1,
        min_covalent_angle=135,
        min_covalent_bond_ratio=0.4,
        shell_depth=0.05,
        electride_elf_min=0.5,
        electride_depth_min=0.2,
        electride_charge_min=0.5,
        electride_volume_min=10,
        electride_radius_min=0.3,
        radius_refine_method="linear",
        write_files: bool = False,
        run_id = None,
        **kwargs,
    ):
        # make a new directory to run badelf algorithm in and copy necessary files.
        analysis_directory = directory / "elf_analysis"
        try:
            os.mkdir(analysis_directory)
        except:
            pass
        files_to_copy = ["CHGCAR", "ELFCAR"]
        for file in files_to_copy:
            shutil.copy(directory / file, analysis_directory)

        # Get the badelf toolkit object for running badelf.
        analysis_tools = ElfAnalyzerToolkit.from_files(
            directory=analysis_directory,
            separate_spin=separate_spin,
            ignore_low_pseudopotentials=ignore_low_pseudopotentials,
            downscale_resolution=downscale_resolution,
        )
        # run elf analysis.
        summary = analysis_tools.get_full_analysis(
            include_lone_pairs=include_lone_pairs,
            include_shared_features=include_shared_features,
            metal_depth_cutoff=metal_depth_cutoff,
            min_covalent_angle=min_covalent_angle,
            min_covalent_bond_ratio=min_covalent_bond_ratio,
            shell_depth=shell_depth,
            electride_elf_min=electride_elf_min,
            electride_depth_min=electride_depth_min,
            electride_charge_min=electride_charge_min,
            electride_volume_min=electride_volume_min,
            electride_radius_min=electride_radius_min,
            radius_refine_method=radius_refine_method,
            write_files=write_files,
            )
        
        # get results
        results = {}
        results["structure"] = analysis_tools.structure
        results["separate_spin"] = separate_spin 
        results["spin_polarized"] = analysis_tools.spin_polarized
        results["ignore_low_pseuodopotentials"] = ignore_low_pseudopotentials
        results["downscale_resolution"] = downscale_resolution
        results["resolution"] = resolution
        results["metal_depth_cutoff"] = metal_depth_cutoff
        results["min_covalent_angle"] = min_covalent_angle
        results["min_covalent_bond_ratio"] = min_covalent_bond_ratio
        results["shell_depth"] = shell_depth
        results["electride_elf_min"] = electride_elf_min
        results["electride_depth_min"] = electride_depth_min
        results["electride_charge_min"] = electride_charge_min
        results["electride_volume_min"] = electride_volume_min
        results["electride_radius_min"] = electride_radius_min
        results["radius_refine_method"] = radius_refine_method
        if separate_spin:
            graph_up = summary["graph_up"]
            graph_down = summary["graph_down"]
            results["bifurcation_graph_up"] = graph_up.to_json()
            results["bifurcation_graph_down"] = graph_down.to_json() 
            results["labeled_structure_up"] = summary["structure_up"].to_json()
            results["labeled_structure_down"] = summary["structure_down"].to_json()
            # get the nodes from each graph
            nodes = []
            for graph, spin in zip([graph_up, graph_down], ["up", "down"]):
                for node_idx in graph.nodes:
                    features = graph.nodes[node_idx]
                    features["spin"] = spin
                    nodes.append(features)
        else:
            graph = summary["graph"]
            results["bifurcation_graph_up"] = graph
            results["labeled_structure_up"] = summary["structure"].to_json()
            # get nodes from graph
            nodes = []
            for node_idx in graph.nodes:
                features = graph.nodes[node_idx]
                features["spin"] = None
                nodes.append(features)
        # Update the table of features
        search_datatable = cls.database_table.objects.get(run_id=run_id)
        search_datatable.update_elf_features(nodes)
        # remove the ELFCAR and CHGCAR copies for space
        for file in files_to_copy:
            os.remove(directory / analysis_directory / file)
        
        return results