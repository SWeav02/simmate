# -*- coding: utf-8 -*-

from dscribe.descriptors import SOAP
from .base import DScribeFeaturizer
from pymatgen.io.ase import AseAtomsAdaptor

class SOAPFeaturizer(DScribeFeaturizer):
    
    def __init__(
            self,
            species: list,
            r_cut: float = 6,
            n_max: int = 8,
            l_max: int = 8,
            sigma: float = 0.4,
            rbf: str = "gto",            
            ):
        self.species=species
        self.r_cut=r_cut
        self.n_max=n_max
        self.l_max=l_max
        self.sigma=sigma
        self.rbf=rbf
        self.periodic=True
        self.compression={"mode":"off"}
        self.average="off"
        self.sparse=False
        self.adaptor = AseAtomsAdaptor()
        super().__init__()
    
    def create_featurizer(self):
        return SOAP(
            species=self.species,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            rbf=self.rbf,
            periodic=self.periodic,
            compression=self.compression,
            average=self.average,
            sparse=self.sparse,
            )
                

                