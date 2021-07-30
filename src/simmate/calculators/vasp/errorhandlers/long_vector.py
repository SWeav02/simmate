# -*- coding: utf-8 -*-

import os

from simmate.workflows.core.tasks.errorhandler import ErrorHandler
from simmate.calculators.vasp.inputs.incar import Incar


class LongVector(ErrorHandler):
    """
    This a simple error handler that is active when VASP finds an issue with the
    rotation matrix.
    """

    # run this while the VASP calculation is still going
    is_monitor = True

    # we assume that we are checking the vasp.out file
    filename_to_check = "vasp.out"

    # These are the error messages that we are looking for in the file
    possible_error_messages = ["One of the lattice vectors is very long (>50 A), but AMIN"]

    def correct(self, error, dir):

        # load the INCAR file to view the current settings
        incar_filename = os.path.join(dir, "INCAR")
        incar = Incar.from_file(incar_filename)

        # make the fix
        incar["AMIN"] = 0.01
        correction = "switched AMIN to 0.01"

        # rewrite the INCAR with new settings
        incar.to_file(incar_filename)

        return correction