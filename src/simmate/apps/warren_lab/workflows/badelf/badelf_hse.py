# -*- coding: utf-8 -*-

from simmate.apps.warren_lab.workflows.badelf.badelf import BadElf__Badelf__Badelf
from simmate.apps.warren_lab.workflows.badelf.base import VaspBadElfBase
from simmate.apps.warren_lab.workflows.badelf.prebadelf_dft import (
    StaticEnergy__Vasp__WarrenLabPrebadelfHse,
)


class BadElf__Badelf__BadelfHse(VaspBadElfBase):
    """
    Runs a static energy calculation using an extra-fine FFT grid using vasp
    and then carries out Badelf and Bader analysis on the resulting charge density.
    Uses the Warren lab settings HSE settings.
    """

    static_energy_workflow = StaticEnergy__Vasp__WarrenLabPrebadelfHse
    badelf_workflow = BadElf__Badelf__Badelf
