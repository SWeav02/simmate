# -*- coding: utf-8 -*-

import copy
import numpy as np
from simmate.toolkit import Structure
from simmate.apps.bader.toolkit import Grid
from simmate.apps.badelf.utilities import (
    get_connected_maxima,
    get_steepest_pointers,
    get_edges,
    get_basin_charge_volume_from_label,
    get_basin_charge_volume_from_weights,
    get_near_grid_assignments,
    get_weighted_voxel_assignments,
    get_neighbor_flux,
    get_basin_weights,
    # get_hybrid_basin_weights,
    # refine_near_grid_edges,
    )
from itertools import product
from numpy.typing import NDArray
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Literal
import time

class Bader:
    """
    Class for running Bader analysis on a regular grid
    """
    
    def __init__(
            self,
            charge_grid: Grid,
            reference_grid: Grid,
            method: Literal["ongrid", "weight"] = None,
            directory: Path = Path(".")
            ):
        self.charge_grid = charge_grid
        self.reference_grid = reference_grid
        if method is not None:
            self.method = method
        else:
            self.method = "weight"
        self.directory = directory
        
        # define hidden class variables. This allows us to cache properties and
        # still be able to recalculate them
        # Assigned by run_bader
        self._basin_labels = None
        self._basin_maxima_frac = None
        self._basin_charges = None
        self._basin_volumes = None
        self._basin_weights = None
        self._basin_weight_voxels = None
        self._basin_weight_voxels_1d = None
        self._basin_surface_distances = None
        # Assigned by run_atom_assignment
        self._basin_atoms = None
        self._basin_atom_dists = None
        self._atom_labels = None
        self._atom_charges = None
        self._atom_volumes = None
        self._atom_surface_distances = None
        self._structure = None
        
    
    @property
    def basin_labels(self) -> NDArray[np.int64]:
        """
        A 3D array of the same shape as the reference grid with entries
        representing the basin the voxel belongs to. Note that for some
        methods (e.g. weight) the voxels have weights for each basin.
        These will be stored in the basin_weights property.
        """
        if self._basin_labels is None:
            self.run_bader()
        return self._basin_labels
    
    @property
    def basin_maxima_frac(self) -> NDArray[np.float64]:
        """
        The fractional coordinates of each attractor
        """
        if self._basin_maxima_frac is None:
            self.run_bader()
        return self._basin_maxima_frac
    
    @property
    def basin_charges(self) -> NDArray[np.float64]:
        """
        The charges assigned to each basin
        """
        if self._basin_charges is None:
            self.run_bader()
        return self._basin_charges
    
    @property
    def basin_volumes(self) -> NDArray[np.float64]:
        """
        The fractional coordinates of each attractor
        """
        if self._basin_volumes is None:
            self.run_bader()
        return self._basin_volumes
    
    @property
    def basin_weights(self) -> NDArray[np.float64]:
        """
        A 2D array where each entry represents a weight and the indices
        i, j represent the voxel index and basin index respectively.
        """
        if self._basin_weights is None and self.method == "weight":
            self.run_bader()
        return self._basin_weights
    
    @property
    def basin_weight_voxels(self) -> (NDArray[np.int64], NDArray[np.int64]):
        """
        A 2D array the same length as the basin_weights representing
        the 3D and 1D voxel indices that each row belongs to
        """
        if self._basin_volumes is None and self.method == "weight":
            self.run_bader()
        return self._basin_weight_voxels, self._basin_weight_voxels_1d
    
    @property
    def basin_surface_distances(self) -> NDArray[np.float64]:
        """
        The distance from each basin maxima to the nearest point on
        the basins surface
        """
        if self._basin_surface_distances is None:
            self._get_basin_surface_distances()
        return self._basin_surface_distances
    
    @property
    def basin_atoms(self) -> NDArray[np.int64]:
        """
        The index of each atom each basin is assigned to
        """
        if self._basin_atoms is None:
            self.run_atom_assignment()
        return self._basin_atoms
    
    @property
    def basin_atom_dists(self) -> NDArray[np.float64]:
        """
        The distance from each basin to the nearest atom
        """
        if self._basin_atom_dists is None:
            self.run_atom_assignment()
        return self._basin_atom_dists
    
    @property
    def atom_labels(self) -> NDArray[np.int64]:
        """
        A 3D array of the same shape as the reference grid with entries
        representing the atoms the voxel belongs to. 
        
        Note that for some methods (e.g. weight) the voxels have weights 
        for each basin and this will not represent exactly how charges
        were assigned.
        These weights be stored in the basin_weights property.
        """
        if self._atom_labels is None:
            self.run_atom_assignment()
        return self._atom_labels
    
    @property
    def atom_charges(self) -> NDArray[np.float64]:
        """
        The charge assigned to each atom
        """
        if self._atom_charges is None:
            self.run_atom_assignment()
        return self._atom_charges
    
    @property
    def atom_volumes(self) -> NDArray[np.float64]:
        """
        The volume assigned to each atom
        """
        if self._atom_volumes is None:
            self.run_atom_assignment()
        return self._atom_volumes
    
    @property
    def atom_surface_distances(self) -> NDArray[np.float64]:
        """
        The distance from each basin maxima to the nearest point on
        the basins surface
        """
        if self._atom_surface_distances is None:
            self._get_atom_surface_distances()
        return self._atom_surface_distances
    
    @property
    def structure(self) -> Structure:
        """
        The structure basins are assigned to
        """
        if self._structure is None:
            self._structure = self.reference_grid.structure.copy()
        return self._structure
            
        
    @property
    def basin_edges(self) -> NDArray[np.bool_]:
        return self.get_basin_edges(self.basin_labels)
    
    
    @staticmethod
    def get_basin_edges(basin_labels: NDArray, neighbor_transforms: NDArray = None):
        """
        Gets a mask representing the edges of a bader calculation
        """
               
        # If no specific neighbors are provided, we default to all 26 neighbors
        if neighbor_transforms is None:
            neighbor_transforms = list(product([-1, 0, 1], repeat=3))
            neighbor_transforms.remove((0, 0, 0))  # Remove the (0, 0, 0) self-shift
            neighbor_transforms = np.array(neighbor_transforms)
        return get_edges(basin_labels, neighbor_transforms=neighbor_transforms)
    
    def run_bader(self):
        """
        Runs the entire bader process and saves results to class variables.
        """
        if self.method == "ongrid":
            self._run_bader_on_grid()
        
        # elif self.method == "neargrid":
        #     self._run_bader_near_grid()
    
        elif self.method == "weight":
            self._run_bader_weight()
        
        elif self.method == "hybrid-weight":
            self._run_bader_hybrid_weight()
        
        else:
            raise ValueError(
                f"{self.method} is not a valid algorithm."
                "Acceptable values are 'ongrid' and 'weight'"
                )

    def _run_bader_on_grid(self):
        """
        Assigns voxels to basins and calculates charge using the on-grid
        method:
            W. Tang, E. Sanville, and G. Henkelman 
            A grid-based Bader analysis algorithm without lattice bias, 
            J. Phys.: Condens. Matter 21, 084204 (2009).
        """
        grid = self.reference_grid
        data = grid.total
        shape = data.shape

        # get an array where each entry is that voxels unique label
        initial_labels = np.arange(np.prod(shape)).reshape(shape)

        # get shifts to move from a voxel to the 26 surrounding voxels
        neighbor_transforms = np.array([s for s in product([-1,0,1], repeat=3) if s != (0,0,0)])

        # get distance from each voxel to its neighbor in cartesian coordinates. This
        # allows us to normalize the gradients
        cartesian_shifts = grid.get_cart_coords_from_vox(neighbor_transforms)
        cartesian_dists = np.linalg.norm(cartesian_shifts, axis=1)

        # For each voxel, get the label of the surrounding voxel that has the highest
        # elf
        logging.info("Calculating steepest neighbors")
        best_label = get_steepest_pointers(
            data=data, 
            initial_labels=initial_labels, 
            neighbor_transforms=neighbor_transforms,
            neighbor_dists=cartesian_dists,
            )

        # ravel the best labels to get a 1D array pointing from each voxel to its steepest
        # neighbor
        pointers = best_label.ravel()
        # Our pointers object is a 1D array pointing each voxel to its parent voxel. We
        # essentially have a classic forrest of trees problem where each maxima is
        # a root and we want to point all of our voxels to their respective root.
        # We being a while loop. In each loop, we remap our pointers to point at
        # the index that its parent was pointing at.
        logging.info("Finding roots")
        while True:
            # reassign each index to the value at the index it is pointing to
            new_parents = pointers[pointers] 
            # check if we have the same value as before
            if np.all(new_parents == pointers):
                break
            # if not, relabel our pointers
            pointers = new_parents
        # We now have our roots. Relabel so that they go from 0 to the length of our
        # roots
        unique_roots, labels_flat = np.unique(pointers, return_inverse=True)
        # reconstruct a 3D array with our labels
        labels = labels_flat.reshape(shape)
        # store our labels
        self._basin_labels = labels
        
        # get maxima voxels
        maxima_mask = best_label == initial_labels
        maxima_vox = np.argwhere(maxima_mask)
        # get corresponding basin labels
        maxima_labels = labels[maxima_vox[:,0],maxima_vox[:,1],maxima_vox[:,2]]
        if not np.all(np.equal(maxima_labels,np.sort(maxima_labels))):
            breakpoint()
        
        # get maxima coords
        maxima_frac = grid.get_frac_coords_from_vox(maxima_vox)
        self._basin_maxima_frac = maxima_frac
        
        # get charge and volume for each label
        logging.info("Calculating basin charges and volumes")
        charge_data = self.charge_grid.total
        voxel_volume = self.charge_grid.voxel_volume
        basin_charges, basin_volumes = get_basin_charge_volume_from_label(
            basin_labels=labels, 
            charge_data=charge_data, 
            voxel_volume=voxel_volume, 
            maxima_num=len(maxima_frac))
        basin_charges /= self.charge_grid.shape.prod()
        self._basin_charges, self._basin_volumes = basin_charges, basin_volumes
    
    # def _run_bader_near_grid(self):
    #     """
    #     Assigns voxels to basins and calculates charge using the near-grid
    #     method:
    #         G. Henkelman, A. Arnaldsson, and H Jonsson.
    #         A fast and robust algorithm for Bader decomposition of charge density, 
    #         J. Phys.: Condens. Matter 21, 084204 (2009).
    #     """
    #     grid = self.reference_grid.copy()
    #     data = grid.total
        
    #     logging.info("Calculating gradient")
    #     # Calculate the gradient in fractional coords
    #     du,dv,dw = 1/grid.shape
    #     frac_gradients = []
    #     for axis, step in zip((0,1,2), (du,dv,dw)):
    #     # for axis in (0,1,2):
    #         # TODO: Also compare with center point in case it is above both of these
    #         shifted_up_data = np.roll(data, -1, axis)
    #         shifted_down_data = np.roll(data, 1, axis)
    #         shift_diff = shifted_up_data-shifted_down_data
    #         # zero out where both are lower than the central data
    #         shift_diff[(shifted_up_data<data)&(shifted_down_data<data)] = 0
    #         frac_gradients.append(shift_diff/(2*step))
    #         # frac_gradients.append(shift_diff/2)
    #     frac_gradients_stack = np.stack(frac_gradients)
    #     # convert to cartesian to remove bias of non-orthogonal lattices
    #     lattice_matrix = grid.structure.lattice.matrix
    #     M = np.linalg.inv(lattice_matrix)
    #     cart_gradients = np.einsum('ia,axyz->ixyz', M, frac_gradients_stack)
    #     # convert back to fractional coordinates.
    #     dir_gradients = np.einsum('ai,ixyz->axyz', lattice_matrix, cart_gradients)
    #     dir_grad_flat = dir_gradients.reshape(3,grid.voxel_num).T
    #     # Normalize each row so that the highest value is 1

    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         max_vals = np.max(np.abs(dir_grad_flat), axis=1)
    #         rgrad = dir_grad_flat/max_vals[:, np.newaxis]
    #     # replace nan values resulting from divide by 0
    #     rgrad[np.isnan(rgrad)] = 0
    #     # Now we calculate the rgrid value.
    #     rgrid = np.round(rgrad).astype(np.int64)
    #     delta_rs = rgrad-rgrid
    #     # Calculate expected neighbor to avoid calc in loop. Note we don't wrap the coord
    #     # here because we will need to during the loop anyways
    #     flat_voxel_coords =  np.indices(data.shape).reshape(3, -1).T
    #     pointer_voxel_coords = flat_voxel_coords + rgrid
    #     # Calculate the location of 0 shifts
    #     zero_grads = np.sum(rgrid, axis=1) == 0
    #     # Now we have two steps left. We need to start at a voxel and hill climb using
    #     # rgrid, keeping track of the accumulated delta_rs. If the delta r every goes
    #     # above 0.5 on any axis we correct the gradient and subtract the correction from
    #     # our delta r. We stop when we reach a maximum or an already labeled voxel. This
    #     # allows us to skip if we've hit an existing voxel. Then we need to perform a
    #     # single refinement.
    #     logging.info("Calculating initial assignments")
    #     # get the labels of each voxel. This allows us to point a new voxel to its corresponding
    #     # rgrid and delta r
    #     flat_voxel_indices = np.arange(np.prod(data.shape))
    #     voxel_indices = flat_voxel_indices.reshape(data.shape)
    #     neighbors = list(product([-1,0,1], repeat=3))
    #     neighbors = np.array([i for i in neighbors if i != (0,0,0)], dtype=np.int64)
    #     cart_neighbors = grid.get_cart_coords_from_vox(neighbors)
    #     neigh_dists = np.linalg.norm(cart_neighbors, axis=1)
    #     assignments, updated_pointer_voxel_coords, maxima_mask = get_near_grid_assignments(
    #         data=data,
    #         flat_voxel_coords=flat_voxel_coords,
    #         pointer_voxel_coords=pointer_voxel_coords,
    #         voxel_indices=voxel_indices,
    #         zero_grads=zero_grads,
    #         rgrid=rgrid,
    #         delta_rs=delta_rs,
    #         neighbors=neighbors,
    #         neighbor_dists=neigh_dists,
    #         )
    #     # assign maxima fractional coords
    #     flat_maxima_mask = maxima_mask.ravel()
    #     maxima_vox_coords = flat_voxel_coords[flat_maxima_mask]
    #     maxima_frac_coords = grid.get_frac_coords_from_vox(maxima_vox_coords)
    #     self._basin_maxima_frac = maxima_frac_coords
    #     # Now we need to refine the edges. First we find them
    #     edge_mask = get_edges(labeled_array=assignments, neighbor_transforms=neighbors)
    #     # remove maxima from the mask in case we have any particularly small basins
    #     edge_mask = edge_mask & ~maxima_mask
    #     flat_edge_mask = edge_mask.ravel()
    #     edge_voxel_coords = flat_voxel_coords[flat_edge_mask]
    #     # We also need to unlabel any edges so that we don't accidentally assign to
    #     # an incorrect edge
    #     refined_assignments = assignments.copy()
    #     refined_assignments[edge_mask] = 0
    #     # Now we loop over them in parallel and perform the same operation as before
    #     # but only assigning the first voxel
    #     logging.info("Refining edges")
    #     refined_assignments = refine_near_grid_edges(
    #         assignments=refined_assignments, 
    #         edge_voxel_coords=edge_voxel_coords, 
    #         pointer_voxel_coords=updated_pointer_voxel_coords, 
    #         voxel_indices=voxel_indices,
    #         delta_rs=delta_rs,
    #         )
    #     # readjust refined assignments to correct indices
    #     refined_assignments -= 1
    #     self._basin_labels = refined_assignments.copy()
    #     # self._basin_labels = assignments.copy()
    #     # get charge and volume for each label
    #     logging.info("Calculating basin charges and volumes")
    #     charge_data = self.charge_grid.total
    #     voxel_volume = self.charge_grid.voxel_volume
    #     basin_charges, basin_volumes = get_basin_charge_volume_from_label(
    #         basin_labels=refined_assignments, 
    #         # basin_labels=assignments,
    #         charge_data=charge_data, 
    #         voxel_volume=voxel_volume, 
    #         maxima_num=len(maxima_frac_coords))
    #     basin_charges /= self.charge_grid.shape.prod()
    #     self._basin_charges, self._basin_volumes = basin_charges, basin_volumes
    
    def _run_bader_weight(self):
        """
        Assigns basin weights to each voxel and assigns charge using
        the weight method:
            M. Yu and D. R. Trinkle, 
            Accurate and efficient algorithm for Bader charge integration, 
            J. Chem. Phys. 134, 064111 (2011).
        """
        reference_grid = self.reference_grid.copy()
        
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = reference_grid.voxel_voronoi_facets
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = data.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = data.ravel()
        flat_voxel_indices = np.arange(np.prod(shape))
        flat_voxel_coords =  np.indices(shape).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        # create an array that maps original voxel indices to their range in terms
        # of data
        flat_sorted_voxel_indices = np.empty_like(flat_voxel_indices)
        flat_sorted_voxel_indices[sorted_data_indices] = flat_voxel_indices
        # Get a 3D grid representing this data and the corresponding 3D indices
        sorted_voxel_indices = flat_sorted_voxel_indices.reshape(shape)
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # Get the flux of volume from each voxel to its neighbor
        logging.info("Calculating voxel flux contributions")
        flux_array, neigh_index_array, maxima_mask = get_neighbor_flux(
            data=data, 
            sorted_coords=sorted_voxel_coords.copy(), 
            voxel_indices=sorted_voxel_indices, 
            neighbor_transforms=neighbor_transforms, 
            neighbor_dists=neighbor_dists, 
            facet_areas=facet_areas)
        # get the frac coords of the maxima
        maxima_vox_coords = sorted_voxel_coords[maxima_mask]
        maxima_frac_coords = reference_grid.get_frac_coords_from_vox(maxima_vox_coords)
        maxima_num = len(maxima_frac_coords)
        self._basin_maxima_frac = maxima_frac_coords
        # Calculate the weights for each voxel to each basin
        logging.info("Calculating weights")
        weights_array = get_basin_weights(
            flux_array=flux_array,
            neigh_indices_array=neigh_index_array,
            maxima_num = maxima_num
            )
        self._basin_weights = weights_array.copy()
        self._basin_weight_voxels = sorted_voxel_coords.copy()
        self._basin_weight_voxels_1d = flat_sorted_voxel_indices.copy()
        logging.info("Calculating basin charges and volumes")
        charge_data = self.charge_grid.total
        flat_charge_data = charge_data.ravel()
        sorted_flat_charge_data = flat_charge_data[sorted_data_indices]
        voxel_volume = reference_grid.voxel_volume
        # get the charges and volumes
        charges, volumes = get_basin_charge_volume_from_weights(
            weight_array=weights_array.copy(),
            flat_charge_array=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            )

        # calculate approximate voxel assignments
        assignments = get_weighted_voxel_assignments(
            weight_array=weights_array.copy(), 
            sorted_coords=sorted_voxel_coords.copy(), 
            data=data,
            )
        self._basin_labels = assignments
        charges /= reference_grid.shape.prod()
        self._basin_charges = charges
        self._basin_volumes = volumes
    
    def _run_bader_hybrid_weight(self):
        """
        Similar to the weight method, assigns basin weights to each voxel 
        and assigns charge using the weight method. However, unlike the
        original method, maxima are first reduced by interpolating the
        values between maxima sharing a 26-neighbor border and checking
        if there is a true minima between them.
        """
        reference_grid = self.reference_grid.copy()
        # get the voronoi neighbors, their distances, and the area of the corresponding
        # facets. This is used to calculate the volume flux from each voxel
        neighbor_transforms, neighbor_dists, facet_areas, _ = reference_grid.voxel_voronoi_facets
        logging.info("Sorting reference data")
        data = reference_grid.total
        shape = data.shape
        # flatten data and get initial 1D and 3D voxel indices
        flat_data = data.ravel()
        flat_voxel_indices = np.arange(np.prod(shape))
        voxel_indices = flat_voxel_indices.reshape(shape)
        flat_voxel_coords =  np.indices(shape).reshape(3, -1).T
        # sort data from high to low
        sorted_data_indices = np.flip(np.argsort(flat_data, kind="stable"))
        # create an array that maps original voxel indices to their range in terms
        # of data
        flat_sorted_voxel_indices = np.empty_like(flat_voxel_indices)
        flat_sorted_voxel_indices[sorted_data_indices] = flat_voxel_indices
        # Get a 3D grid representing this data and the corresponding 3D indices
        sorted_voxel_indices = flat_sorted_voxel_indices.reshape(shape)
        sorted_voxel_coords = flat_voxel_coords[sorted_data_indices]
        # Get the flux of volume from each voxel to its neighbor
        logging.info("Calculating voxel flux contributions")
        flux_array, neigh_index_array, maxima_mask = get_neighbor_flux(
            data=data, 
            sorted_coords=sorted_voxel_coords.copy(), 
            voxel_indices=sorted_voxel_indices, 
            neighbor_transforms=neighbor_transforms, 
            neighbor_dists=neighbor_dists, 
            facet_areas=facet_areas)
        # get the voxel coords of the maxima
        maxima_vox_coords = sorted_voxel_coords[maxima_mask]

        logging.info("Reducing connected maxima")
        connected_maxima, maxima_frac_coords = get_connected_maxima(
            maxima_vox_coords=maxima_vox_coords, 
            flat_voxel_coords=flat_voxel_coords, 
            voxel_indices=voxel_indices, 
            grid=reference_grid,
            )
        # store the new basin maxima
        self._basin_maxima_frac = maxima_frac_coords
        # connected maxima are in original unsorted indices. We convert these back
        connected_maxima = [flat_sorted_voxel_indices[orig_indices] for orig_indices in connected_maxima]

        # create a weight array to store values
        weight_array = np.zeros((len(flux_array), len(connected_maxima)), dtype=np.float64)
        for i, maxima_group in enumerate(connected_maxima):
            weight_array[maxima_group, i] = 1.0

        # Calculate the rest of the weights
        logging.info("Calculating weights")
        weights_array = get_hybrid_basin_weights(
            flux_array=flux_array,
            neigh_indices_array=neigh_index_array,
            weight_array=weight_array,
            )
        self._basin_weights = weights_array.copy()
        self._basin_weight_voxels = sorted_voxel_coords.copy()
        self._basin_weight_voxels_1d = flat_sorted_voxel_indices.copy()
        logging.info("Calculating basin charges and volumes")
        charge_data = self.charge_grid.total
        flat_charge_data = charge_data.ravel()
        sorted_flat_charge_data = flat_charge_data[sorted_data_indices]
        voxel_volume = reference_grid.voxel_volume
        # get the charges and volumes
        charges, volumes = get_basin_charge_volume_from_weights(
            weight_array=weights_array.copy(),
            flat_charge_array=sorted_flat_charge_data,
            voxel_volume=voxel_volume,
            )

        # calculate approximate voxel assignments
        assignments = get_weighted_voxel_assignments(
            weight_array=weights_array.copy(), 
            sorted_coords=sorted_voxel_coords.copy(), 
            data=data,
            )
        self._basin_labels = assignments
        charges /= reference_grid.shape.prod()
        self._basin_charges = charges
        self._basin_volumes = volumes
    
    def run_atom_assignment(self, structure: Structure = None):
        """
        Assigns bader basins to the atoms in the provided structure. If
        no structure is provided, defaults to the reference grid structure.
        This is useful for reassigning basins to different structures.
        """
        if structure is None:
            structure = self.structure
        self._structure = structure
        # Get the frac coords for each basin and atom. These must be in the
        # same order as the corresponding basin labels
        basin_frac_coords = self.basin_maxima_frac
        atom_frac_coords = structure.frac_coords
        logging.info("Assigning atom properties")
        # create arrays for atom properties
        basin_atoms = np.empty(len(basin_frac_coords), dtype=int)
        basin_atom_dists = np.empty(len(basin_frac_coords))
        atom_labels = np.zeros(self.basin_labels.shape, dtype=np.int64)
        atom_charges = np.zeros(len(atom_frac_coords))
        atom_volumes = np.zeros(len(atom_frac_coords))
        
        for i, frac_coord in enumerate(basin_frac_coords):
            # get the difference between this basin and all of the atoms
            diffs = atom_frac_coords - frac_coord
            # wrap anything below -0.5 or above 0.5
            diffs[diffs<-0.5] += 1
            diffs[diffs>0.5] -= 1
            # convert to cartesian coords and calculate distance
            cart_diffs = diffs @ structure.lattice.matrix
            dists = np.linalg.norm(cart_diffs,axis=1)
            # get the lowest distance and corresponding atom
            min_dist = dists.min()
            assignment = np.argwhere(dists==min_dist)[0]
            # assign this atom label to this basin and update properties
            basin_atoms[i] = assignment
            basin_atom_dists[i] = min_dist
            atom_labels[self.basin_labels==i] = assignment
            atom_charges[assignment] += self.basin_charges[i]
            atom_volumes[assignment] += self.basin_volumes[i]
            
        # update class variables
        self._basin_atoms = basin_atoms
        self._basin_atom_dists = basin_atom_dists
        self._atom_labels = atom_labels
        self._atom_charges = atom_charges
        self._atom_volumes = atom_volumes

    def _get_atom_surface_distances(self):
        """
        Calculates the distance from each atom to the nearest surface
        """
        atom_labeled_voxels = self.atom_labels
        atom_radii = []
        edge_mask = self.get_basin_edges(atom_labeled_voxels)
        for atom_index in tqdm(range(len(self.structure)), desc="Calculating feature radii"):
            # get the voxels corresponding to the interior edge of this basin
            atom_edge_mask = (atom_labeled_voxels == atom_index) & edge_mask
            edge_vox_coords = np.argwhere(atom_edge_mask)
            # convert to frac coords
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(edge_vox_coords)
            atom_frac_coord = self.structure.frac_coords[atom_index]
            # Get the difference in coords between atom and edges
            coord_diff = atom_frac_coord - edge_frac_coords
            # Wrap any coords that are more than 0.5 or less than -0.5
            coord_diff -= np.round(coord_diff)
            # Convert to cartesian coordinates
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            # Calculate distance of each
            norm = np.linalg.norm(cart_coords, axis=1)
            if len(norm) == 0:
                logging.warning(f"No volume assigned to atom at site {atom_index}.")
                atom_radii.append(0)
            else:
                atom_radii.append(norm.min())
        atom_radii = np.array(atom_radii)
        self._atom_surface_distances = atom_radii
    
    def _get_basin_surface_distances(self):
        """
        Calculates the distance from each basin maxima to the nearest surface
        """
        basin_labeled_voxels = self.basin_labels
        basin_radii = []
        edge_mask = self.basin_edges
        for basin in tqdm(range(len(self.basin_maxima_frac)), desc="Calculating feature radii"):
            basin_edge_mask = (basin_labeled_voxels == basin) & edge_mask
            edge_vox_coords = np.argwhere(basin_edge_mask)
            edge_frac_coords = self.reference_grid.get_frac_coords_from_vox(edge_vox_coords)
            basin_frac_coord = self.basin_maxima_frac[basin]

            coord_diff = basin_frac_coord - edge_frac_coords
            coord_diff -= np.round(coord_diff)
            cart_coords = self.reference_grid.get_cart_coords_from_frac(coord_diff)
            norm = np.linalg.norm(cart_coords, axis=1)
            basin_radii.append(norm.min())
        basin_radii = np.array(basin_radii)
        self._basin_surface_distances = basin_radii

    @classmethod
    def from_vasp(
            cls, 
            charge_filename: str = "CHGCAR",
            reference_filename: str ="ELFCAR",
            directory: str | Path = Path("."),
            **kwargs,
            ):
        """
        Creates a Bader class object from VASP files
        """
        charge_grid = Grid.from_vasp(directory / charge_filename)
        reference_grid = Grid.from_vasp(directory / reference_filename)
        return Bader(charge_grid=charge_grid, reference_grid=reference_grid, directory=directory, **kwargs)
    
    
    def copy(self):
        """
        Returns a deep copy of this Bader object.
        """
        return copy.deepcopy(self)
    
    @property
    def results_summary(self):
        """
        A dictionary summary of all results
        """
        results_dict = {
            "method": self.method,
            "basin_maxima_frac": self.basin_maxima_frac,
            "basin_charges": self.basin_charges,
            "basin_volumes": self.basin_volumes,
            "basin_surface_distances": self.basin_surface_distances,
            "basin_atoms": self.basin_atoms,
            "basin_atom_dists": self.basin_atom_dists,
            "atom_charges": self.atom_charges,
            "atom_volumes": self.atom_volumes,
            "atom_surface_distances": self.atom_surface_distances,
            "structure": self.structure,
            }
        return results_dict
    
    def write_basin_volumes(
            self, 
            basin_indices: NDArray, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes each basin volume from a list of indices to individual 
        vasp-like files
        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()
        
        data_array = grid.total
        directory = self.directory
        for basin in basin_indices:
            mask = self.basin_labels == basin
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(self.structure,data)
            grid.write_file(directory / f"{file_prefix}_b{basin}")
    
    def write_all_basin_volumes(
            self, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes all basins to vasp-like files
        """
        basin_indices = np.array(range(len(self.basin_atoms)))
        self.write_basin_volumes(
            basin_indices=basin_indices,
            file_prefix=file_prefix,
            data_type=data_type,
            )
    
    def write_basin_volumes_sum(
            self, 
            basin_indices: NDArray, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes the volume made up of all basin indices provided to a
        vasp-like file
        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()
        
        data_array = grid.total
        directory = self.directory
        mask = np.isin(self.basin_labels, basin_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(self.structure,data)
        grid.write_file(directory / f"{file_prefix}_bsum")
        
    def write_atom_volumes(
            self, 
            atom_indices: NDArray, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes each atom volume from a list of indices to individual 
        vasp-like files
        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()
        
        data_array = grid.total
        directory = self.directory
        for atom_index in atom_indices:
            mask = self.atom_labels == atom_index
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0
            data = {"total": data_array_copy}
            grid = Grid(self.structure,data)
            grid.write_file(directory / f"{file_prefix}_a{atom_index}")
    
    def write_all_atom_volumes(
            self, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes all atoms to vasp-like files
        """
        atom_indices = np.array(range(len(self.structure)))
        self.write_atom_volumes(
            atom_indices=atom_indices,
            file_prefix=file_prefix,
            data_type=data_type,
            )
    
    def write_atom_volumes_sum(
            self, 
            atom_indices: NDArray, 
            file_prefix: str = "CHGCAR",
            data_type: Literal["charge", "reference"] = "charge",
            ):
        """
        Writes the volume made up of all atom indices provided to a
        vasp-like file
        """
        if data_type == "charge":
            grid = self.charge_grid.copy()
        elif data_type == "reference":
            grid = self.reference_grid.copy()
        
        data_array = grid.total
        directory = self.directory
        mask = np.isin(self.atom_labels, atom_indices)
        data_array_copy = data_array.copy()
        data_array_copy[~mask] = 0
        data = {"total": data_array_copy}
        grid = Grid(self.structure,data)
        grid.write_file(directory / f"{file_prefix}_asum")
