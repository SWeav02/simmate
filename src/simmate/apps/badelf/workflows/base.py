# -*- coding: utf-8 -*-

import os
import shutil

# This will be added back once I go through and handle warnings within context
# import warnings
from pathlib import Path

from simmate.apps.badelf.core.badelf import BadElfToolkit
from simmate.engine import Workflow

# This file contains workflows for performing Bader and BadELF. Parts of the code
# use the Henkelman groups algorithm for Bader analysis:
# (http://theory.cm.utexas.edu/henkelman/code/bader/).


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
        electride_finder_cutoff: float = 0.5,  # This is somewhat arbitrarily set
        algorithm: str = "badelf",
        check_for_covalency: bool = True,
        write_electride_files: bool = False,
        write_ion_radii: bool = True,
        run_id: str = None,
        **kwargs,
    ):
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
        badelf_tools = BadElfToolkit.from_files(
            directory=badelf_directory,
            find_electrides=find_electrides,
            algorithm=algorithm,
        )
        # Set options and run badelf.
        if not check_for_covalency:
            badelf_tools.check_for_covalency = False
        badelf_tools.electride_finder_cutoff = electride_finder_cutoff
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
        badelf_tools.write_results_csv()
        return results
