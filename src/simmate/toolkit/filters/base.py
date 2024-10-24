# -*- coding: utf-8 -*-

from rich.progress import track

from simmate.toolkit import Molecule


class Filter:
    """
    Abstract base class for filtering a list of molecules based on some criteria.
    """

    is_atomic: bool = False
    """
    Whether the filter can be applied to individual entries.
    
    For some cases, a filter may depend on the makeup of all the molecules 
    provided -- rather than applying a check to individual molecules. These 
    cases should have 'is_atomic' set to False.
    
    For example, a filter that 'keep molecules under a weight of 100 g/mol' 
    is atomic, but a filter that says 'take the top 10% of molecules ranked 
    by weight' is NOT atomic.
    """

    @classmethod
    def check(cls, molecule: Molecule, **kwargs) -> bool:
        """
        "Filters" a single molecule which is really just a check that returns
        a true or false.
        """
        # by default we assume there is a custom _check_serial method and call that
        if not cls.is_atomic:
            raise Exception(
                "This is NOT an atomic filter, which means you can not check individual"
                "molecules. Use the 'filter' method instead."
            )
        return cls._check_serial(molecules=[molecule], **kwargs)[0]

    @classmethod
    def filter(
        cls,
        molecules: list[Molecule],
        return_mode: str = "molecules",  # other options are "booleans" and "count"
        return_type: str = "passed",  # other options are "failed" and "both"
        parallel: bool = False,
        **kwargs,
    ) -> list:
        """
        Filters a list of molecules in a serial or parallel manner.
        """
        if not parallel:
            return cls._check_serial(molecules, **kwargs)
        else:
            if not cls.allow_parallel:
                raise Exception("This filtering method cannot be ran in parallel")
            return cls._check_parallel(molecules, **kwargs)

    @classmethod
    def _check_serial(
        cls,
        molecules: list[Molecule],
        progress_bar: bool = True,
        **kwargs,
    ) -> list[bool]:
        """
        Filters a list of molecules in serial
        (so one at a time on a single core).
        """
        features_list = []
        for molecule in track(molecules, disable=not progress_bar):
            features = cls.check(molecule, **kwargs)
            features_list.append(features)
        return features_list

    @classmethod
    def _check_parallel(
        cls,
        molecules: list[Molecule],
        batch_size: int = 10000,
        use_serial_batches: bool = False,
        batch_size_serial: int = 500,
        **kwargs,
    ) -> list[bool]:
        """
        Filters a list of molecules in parallel.
        """
        # Use this method to help. Maybe write a utility function for batch
        # submitting and serial-batch submitting to dask too.
        # https://github.com/jacksund/simmate/blob/17d926fe5ee8f183240a4526982b4d7fd5d7042b/src/simmate/toolkit/creators/structure/base.py#L67
        raise NotImplementedError("This method is still being written")

    @classmethod
    def from_preset(cls, preset: str, molecules: list[Molecule], **kwargs):
        if preset == "many-smarts":
            from simmate.toolkit import Molecule
            from simmate.toolkit.filters import ManySmarts as ManySmartsFilter

            smarts_strs = kwargs.pop("smarts_strs", None)
            if not smarts_strs:
                raise ValueError(
                    "smiles_strs is required as an input for this method. "
                    "Provide a list of SMARTS strings."
                )
            smiles_mols = [Molecule.from_smarts(s) for s in smarts_strs]
            return ManySmartsFilter.filter(
                molecules=molecules,
                smarts_list=smiles_mols,
                **kwargs,
            )
        else:
            raise Exception(f"Unknown present provided: {preset}")