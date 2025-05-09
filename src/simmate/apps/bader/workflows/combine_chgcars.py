# -*- coding: utf-8 -*-

from simmate.workflows.base_flow_types import S3Workflow


# TODO: The chgsum.pl script will be replaced with a simple python function
# that just sums the two files. It might not be as fast but it removes one
# executable file from having to be in the user's path. So in the future, this
# Task will be depreciated/removed into the BaderAnalysis.setup method.
class PopulationAnalysis__Bader__CombineChgcars(S3Workflow):
    """
    This tasks simply sums two charge density files into a new file. It uses
    a script from the Henkleman group.
    """

    command = "chgsum.pl AECCAR0 AECCAR2 > chgsum.out"
    monitor = False
    required_files = ["CHGCAR", "AECCAR0", "AECCAR2"]
    use_database = False
    # Note -- POTCAR is copied over so that downstream workflows can grab it
    use_previous_directory = ["CHGCAR", "AECCAR0", "AECCAR2", "POTCAR"]
    parent_workflows = ["population-analysis.vasp-bader.bader-matproj"]
