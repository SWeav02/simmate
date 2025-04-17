# -*- coding: utf-8 -*-

from simmate.apps.badelf.models import ElfAnalysis
from simmate.apps.badelf.workflows.base import ElfAnalysisBase
from simmate.database import connect


class ElfAnalysis__Badelf__ElfTopologyAnalysis(ElfAnalysisBase):
    """
    This workflow performs a topology analysis on the ELF and charge density
    from a static energy calculation. The directory it is run in must 
    already have an ELFCAR and CHGCAR with the same grid size.
    
    This workflow must be run with a specified directory that already exists!
    """

    use_database = True
    database_table = ElfAnalysis
