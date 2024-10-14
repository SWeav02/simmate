# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from simmate.toolkit import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np


class Featurizer(ABC):
    """
    An abstract class for extracting local features (arrays) representing local
    environments. All featurizer classes should define the methods "create_featurizer"
    and "featurize_single" and should include super().__init__() in the __init__ method.
    
    The "create_featurizer" method should be a wraparound for an external package
    such as DScribe or matminer which instantiates a class from the external package
    for featurizing an atoms local environment.
    
    The "featurize_single" method should take in a structure and index and use
    the featurizer created by create_featurizer to generate a 1D array.
    
    This abstract class formalizes helps to bring all featurizers from different
    packages into one syntax aimed at the specific goal of learning FFs
    """
    def __init__(self):
        """
        This is where settings specific to each featurizer should be defined in
        inheriting subclasses. Subclasses should also call super().__init__()
        so that the line below is called.
        """
        self.featurizer = self.create_featurizer()
    
    @abstractmethod
    def create_featurizer(self):
        """
        This is where the featurizer class from whichever third party package
        should be instantiated
        """
        raise NotImplementedError
    
    @abstractmethod
    def featurize_single(self, structure: Structure, sites: list[int]) -> np.array:
        """
        This is where a single call to the featurizer class should be called
        """
        raise NotImplementedError
    
    def featurize_all(self, structures: list[Structure]) -> (dict, list):
        """
        Takes in a list of structures and returns the features for each unique
        local environment in the structures. The features for each element type
        are returned in dictionaries and the corresponding original structure is
        noted in a list
        """
        # create dict to store features for each element type
        features = {}
        indices = {}
        for specie in self.species:
            features[specie] = []
            indices[specie] = []

        for i, structure in enumerate(structures):
            # get unique sites
            unique_sites = SpacegroupAnalyzer(structure).get_symmetry_dataset()[
                "equivalent_atoms"
            ]
            unique_sites = np.unique(unique_sites)
            # for i, site in enumerate(structure):
            for site in unique_sites:
                specie = structure[site].species_string
                feature = self.featurize_single(structure, site)
                features[specie].append(np.squeeze(feature))
                indices[specie].append(i)
        return features, indices


# DScribe, Matminer, and others use different syntax when using featurizers. We
# define the featurize_single methods here to account for this. We still leave
# create_featurizer blank as this will depend on a case by case for each type of
# featurizer.
class DScribeFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def create_featurizer(self):
        raise NotImplementedError
    
    def featurize_single(self, structure: Structure, site: int) -> np.array:
        # convert structure to atoms as required by DScribe
        atoms = self.adaptor.get_atoms(structure)
        return self.featurizer.create(system=atoms, centers=[site])
    
    

class MatminerFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def create_featurizer(self):
        raise NotImplementedError
    
    def featurize_single(self, structure: Structure, site: int) -> np.array:
        return self.featurizer.featurize(struct=structure, idx=site)