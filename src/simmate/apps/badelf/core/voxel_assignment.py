# -*- coding: utf-8 -*-

import itertools
import logging
import math
from functools import cached_property
from pathlib import Path

import dask.array as da
import numpy as np
import psutil
from numpy.typing import ArrayLike
from tqdm import tqdm

from simmate.apps.badelf.core.grid import Grid
from simmate.toolkit import Structure


class VoxelAssignmentToolkit:
    """
    A set of tools for assigning charge to atoms in a unit cell.

    Args:
        charge_grid (Grid):
            A BadELF app Grid type object usually with CHGCAR type data.
        partitioning_grid (Grid):
            A BadELF app Grid type object usually with ELFCAR type data.
        algorithm (str):
            The algorithm to use for partitioning. Defaults to BadELF
        partitioning (dict):
            A partitioning dictionary generated from the BadELF app
            PartitioningToolkit. Will be generated from the grid if None.
        electride_structure (Structure):
            The structure with electride sites. Will be generated if not given.

    """

    def __init__(
        self,
        charge_grid: Grid,
        electride_structure: Structure,
        algorithm: str,
        partitioning: dict,
        directory: Path,
    ):
        self.charge_grid = charge_grid.copy()
        self.algorithm = algorithm
        # partitioning will contain electride sites for voronelf
        self.partitioning = partitioning
        self.electride_structure = electride_structure

    @property
    def unit_cell_permutations_vox(self):
        """
        The permutations required to transform a unit cell to each of its neighbors.
        Uses voxel coordinates.
        """
        return self.charge_grid.permutations

    @property
    def unit_cell_permutations_frac(self):
        """
        The permutations required to transform a unit cell to each of its neighbors.
        Uses fractional coordinates.
        """
        unit_cell_permutations_frac = [
            (t, u, v)
            for t, u, v in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        ]
        # move 0,0,0 transformation to front as it is the most likely to be important
        unit_cell_permutations_frac.insert(0, unit_cell_permutations_frac.pop(13))
        return unit_cell_permutations_frac

    @property
    def unit_cell_permutations_cart(self):
        """
        The permutations required to transform a unit cell to each of its neighbors.
        Uses cartesian coordinates.
        """
        grid = self.charge_grid
        return grid.get_cart_coords_from_frac_full_array(
            self.unit_cell_permutations_frac
        )

    @property
    def vertices_transforms_frac(self):
        """
        The transformations required to transform the center of a voxel to its
        corners. Uses fractional coordinates.
        """
        a, b, c = self.charge_grid.grid_shape
        a1, b1, c1 = 1 / (2 * a), 1 / (2 * b), 1 / (2 * c)
        x, y, z = np.meshgrid([-a1, a1], [-b1, b1], [-c1, c1])
        vertices_transforms_frac = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        return vertices_transforms_frac

    @property
    def vertices_transforms_cart(self):
        """
        The transformations required to transform the center of a voxel to its
        corners. Uses cartesian coordinates.
        """
        return self.charge_grid.get_cart_coords_from_frac_full_array(
            self.vertices_transforms_frac
        )

    @cached_property
    def all_voxel_frac_coords(self):
        """
        The fractional coordinates for all of the voxels in the charge grid
        """
        return self.charge_grid.all_voxel_frac_coords

    @cached_property
    def all_partitioning_plane_points_and_vectors(self):
        """
        The points and vectors for all of the partitioning planes stored as
        two sets of N,3 shaped arrays.
        """
        partitioning = self.partitioning
        plane_points = []
        plane_vectors = []
        for atom_index, partitioning_df in partitioning.items():
            atom_plane_points = partitioning_df["plane_points"].to_list()
            atom_plane_vectors = partitioning_df["plane_vectors"].to_list()
            plane_points.extend(atom_plane_points)
            plane_vectors.extend(atom_plane_vectors)

        # convert plane points and vectors to arrays and then convert to the plane
        # equation
        plane_points = np.array(plane_points)
        plane_vectors = np.array(plane_vectors)
        return plane_points, plane_vectors

    @cached_property
    def all_plane_equations(self):
        """
        A (N,4) array containing every partitioning plane equation
        """
        plane_points, plane_vectors = self.all_partitioning_plane_points_and_vectors
        D = -np.sum(plane_points * plane_vectors, axis=1)
        # convert to all coefficients
        return np.column_stack((plane_vectors, D))

    @cached_property
    def number_of_planes_per_atom(self):
        """
        A list for splitting an array containing all of the partitioning planes
        back into atom based sections.
        """
        partitioning = self.partitioning
        number_of_planes_per_atom = [len(planes) for planes in partitioning.values()]
        number_of_planes_per_atom.pop(-1)
        number_of_planes_per_atom = np.array(number_of_planes_per_atom)
        sum_number_of_planes_per_atom = []
        for i, j in enumerate(number_of_planes_per_atom):
            sum_number_of_planes_per_atom.append(
                j + np.sum(number_of_planes_per_atom[:i])
            )
        return sum_number_of_planes_per_atom

    @cached_property
    def voxel_edge_vectors(self):
        """
        A (12,3) array consisting of the vectors that make up the edges of
        a voxel in the grid
        """
        # Here we define the edges of each voxel in terms of vertex indices. these are
        # not unique and other choices could be made. One potentially faster option is
        # to arrange the edges so that the first index is as small of a set of indices
        # as possible. Later when we calculate t in our line segments this would reduce
        # the number of calculations needed for the numerator which only depends on
        # the plane equation and the first vertex position
        # Get the edge vectors in cartesian coordinates
        vertices_transforms_cart = self.vertices_transforms_cart
        edges = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [3, 1],
                [3, 2],
                [3, 7],
                [5, 1],
                [5, 4],
                [5, 7],
                [6, 2],
                [6, 4],
                [6, 7],
            ]
        )
        edge_vectors = []
        for edge in edges:
            edge_vector = (
                vertices_transforms_cart[edge[1]] - vertices_transforms_cart[edge[0]]
            )
            edge_vectors.append(edge_vector)
        edge_vectors = np.array(edge_vectors)
        return edge_vectors

    def get_site_assignments_from_frac_coords(
        self,
        voxel_frac_coords: ArrayLike,
        min_dist_from_plane: float,
    ):
        """
        Gets the site assignments for an arbitrary number of voxels described
        by their fractional coordinates.

        Args:
            voxel_frac_coords (ArrayLike):
                An N,3 array of fractional coordinates corresponding to the voxels
                to assign sites to
            min_dist_from_plane (float):
                The minimum distance a point should be from the plane before
                a site can be assigned. This is value usually corresponds to
                the maximum distance the voxel center can be from the plane
                while the voxel is still being intersected by the plane.

        Returns:
            A 1D array of atomic site assignments. Assignments start at 1 with
            0 indicating no site was found for this coordinate.
        """
        grid = self.charge_grid
        plane_equations = self.all_plane_equations
        number_of_planes_per_atom = self.number_of_planes_per_atom
        unit_cell_permutations_frac = self.unit_cell_permutations_frac

        # Create an array of zeros to map back to
        zeros_array = np.zeros(len(voxel_frac_coords))
        # Create an array that the results will be added to
        results_array = zeros_array.copy()

        # create zeros array for any problems
        # global_indices_to_zero = np.array([])
        # check every possible permutation
        for transformation in tqdm(
            unit_cell_permutations_frac,
            total=len(unit_cell_permutations_frac),
            ascii="░▒▓",
        ):
            # Get the indices where voxels haven't been assigned. Get only these
            # frac coords
            indices_where_zero = np.where(results_array == 0)[0]
            new_frac_coords = voxel_frac_coords.copy()[indices_where_zero]
            # transform the fractional coords to the next transformation
            x1, y1, z1 = transformation
            new_frac_coords[:, 0] += x1
            new_frac_coords[:, 1] += y1
            new_frac_coords[:, 2] += z1
            # Convert the frac coords into cartesian coords
            cart_coords = grid.get_cart_coords_from_frac_full_array(
                new_frac_coords
            ).astype(float)
            points = np.array(cart_coords).astype(float)
            planes = np.array(plane_equations).astype(float)
            # There is a difference in the speed of dask vs numpy. Dask has a
            # lot of overhead, but at a certain point it is faster than numpy.
            # We check which one we should use here.
            plane_distances_to_calc = len(points) * len(planes)
            if plane_distances_to_calc > 7.8e8:
                dask = True
            else:
                dask = False

            if dask:
                # DASK ARRAY VERSION
                # points = da.from_array(points)
                # planes = da.from_array(planes)
                distances = da.dot(points, planes[:, :3].T) + planes[:, 3]
                # Round the distances to within 5 decimals. Everything to this point has
                # been based on lattice position from vasp which typically have 5-6
                # decimal places (7 sig figs)
                distances = np.round(distances, 12)
                # We write over the distances with a more simplified boolean to save
                # space. This is also where we filter if we're near a plane if desired
                distances = da.where(distances < -min_dist_from_plane, True, False)
                distances = distances.compute()

            else:
                # BASE NUMPY VERSION
                distances = np.dot(points, planes[:, :3].T) + planes[:, 3]
                # Round the distances to within 5 decimals. Everything to this point has
                # been based on lattice position from vasp which typically have 5-6
                # decimal places (7 sig figs)
                distances = np.round(distances, 12)
                # We write over the distances with a more simplified boolean to save
                # space. This is also where we filter if we're near a plane if desired
                distances = np.where(distances < -min_dist_from_plane, True, False)

            # split the array into the planes belonging to each atom. Again we write
            # over to save space
            distances = np.array_split(distances, number_of_planes_per_atom, axis=1)
            # get a 1D array representing the voxel indices with the atom index where the
            # voxel is assigned to a site and 0s where they are not
            new_results_arrays = []
            # for atom_index, atom_array in enumerate(distances_split_by_atom):
            for atom_index, atom_array in enumerate(distances):
                voxel_result = np.all(atom_array, axis=1)
                voxel_result = np.where(
                    voxel_result == True, atom_index + 1, voxel_result
                )
                new_results_arrays.append(voxel_result)

            indices_to_zero = []
            new_results_array = np.zeros(len(new_results_arrays[0]))
            for i, sub_results_array in enumerate(new_results_arrays):
                new_results_array = np.sum(
                    [new_results_array, sub_results_array], axis=0
                )
                indices_to_zero.extend(np.where(new_results_array > i + 1)[0])
            indices_to_zero = np.unique(indices_to_zero).astype(int)
            # print(len(indices_to_zero))
            new_results_array[indices_to_zero] = 0
            # Sum the results
            # new_results_array = np.sum(new_results_arrays, axis=0)
            # add results to the results_array
            results_array[indices_where_zero] = new_results_array
        return results_array

    def get_site_assignments_from_frac_coords_with_memory_handling(
        self,
        voxel_frac_coords: ArrayLike,
        min_dist_from_plane: float,
    ):
        """
        Gets the site assignments for an arbitrary number of voxels described
        by their fractional coordinates. Takes available memory into account
        and divides the voxels into chunks to perform operations.

        Args:
            voxel_frac_coords (ArrayLike):
                An N,3 array of fractional coordinates corresponding to the voxels
                to assign sites to
            min_dist_from_plane (float):
                The minimum distance a point should be from the plane before
                a site can be assigned. This is value usually corresponds to
                the maximum distance the voxel center can be from the plane
                while the voxel is still being intersected by the plane.

        Returns:
            A 1D array of atomic site assignments. Assignments start at 1 with
            0 indicating no site was found for this coordinate.
        """
        partitioning = self.partitioning
        # determine how much memory is available. Then calculate how many distance
        # calculations would be possible to do at once with this much memory.
        available_memory = psutil.virtual_memory().available / (1024**2)

        handleable_plane_distance_calcs_numpy = available_memory / 0.00007
        handleable_plane_distance_calcs_dask = available_memory / 0.000025
        plane_distances_to_calc = len(voxel_frac_coords) * sum(
            [len(i) for i in partitioning.values()]
        )

        # I found there is a cutoff where Dask becomes faster than numpy. This
        # may vary with the number of cores available. It is largely due to Dask
        # having a large overhead.
        if plane_distances_to_calc > 7.8e8:
            # calculate the number of chunks the voxel array should be split into to not
            # overload the memory. Then split the array by this number
            split_num = math.ceil(
                plane_distances_to_calc / handleable_plane_distance_calcs_dask
            )
        else:
            split_num = math.ceil(
                plane_distances_to_calc / handleable_plane_distance_calcs_numpy
            )
        split_voxel_frac_coords = np.array_split(voxel_frac_coords, split_num, axis=0)
        # create an array to store results
        voxel_results_array = np.array([])
        # for each split, calculate the results and add to the end of our results
        for chunk, split_voxel_array in enumerate(split_voxel_frac_coords):
            logging.info(
                f"Calculating site assignments for voxel chunk {chunk+1}/{split_num}"
            )
            split_result = self.get_site_assignments_from_frac_coords(
                voxel_frac_coords=split_voxel_array,
                min_dist_from_plane=min_dist_from_plane,
            )
            voxel_results_array = np.concatenate([voxel_results_array, split_result])
        return voxel_results_array

    def get_single_site_voxel_assignments(self, all_site_voxel_assignments: ArrayLike):
        """
        Gets the voxel assignments for voxels that are not split by a plane.

        Args:
            all_site_voxel_assignments (ArrayLike):
                A 1D array of integers representing the site assignments for
                each voxel in the grid.

        Returns:
            A 1D array of the same length as the input with additional site
            assignments.
        """
        all_voxel_assignments = all_site_voxel_assignments.copy()
        # charge_grid = self.charge_grid
        # In the BadELF algorithm the electride sites will have already been
        # assigned. In VoronELF they won't be. Here we search for unassigned
        # voxels and then run the alg on the remaining ones
        unassigned_indices = np.where(all_voxel_assignments == 0)[0]
        all_voxel_frac_coords = self.all_voxel_frac_coords
        frac_coords_to_find = all_voxel_frac_coords[unassigned_indices]
        # min_dist_from_plane = charge_grid.max_voxel_dist
        single_site_voxel_assignments = (
            self.get_site_assignments_from_frac_coords_with_memory_handling(
                frac_coords_to_find, min_dist_from_plane=0
            )
        )
        all_voxel_assignments[unassigned_indices] = single_site_voxel_assignments
        return all_voxel_assignments

    def get_multi_site_voxel_assignments_from_frac_coords(
        self, voxel_frac_coords: ArrayLike
    ):
        """
        Gets the voxel assignments for voxels that are not within the partitioning
        surface.

        Args:
            all_site_voxel_assignments (ArrayLike):
                A 1D array of integers representing the site assignments for
                each voxel in the grid.

        Returns:
            A 1D array of the same length as the input with additional site
            assignments.
        """
        # get the partitioning planes
        planes = self.all_plane_equations.astype(float)
        number_of_planes_per_atom = self.number_of_planes_per_atom

        if self.algorithm == "badelf":
            structure = self.charge_grid.structure
        elif self.algorithm == "voronelf":
            structure = self.electride_structure

        # get all possible permutations of fractional coords
        voxel_frac_coords_perm = []
        permutations = self.unit_cell_permutations_frac
        for x, y, z in permutations:
            new_frac_coords = voxel_frac_coords.copy()
            new_frac_coords[:, 0] += x
            new_frac_coords[:, 1] += y
            new_frac_coords[:, 2] += z
            voxel_frac_coords_perm.append(new_frac_coords)
        voxel_frac_coords_perm = np.concatenate(voxel_frac_coords_perm)
        cart_coords_to_find_perm = (
            self.charge_grid.get_cart_coords_from_frac_full_array(
                voxel_frac_coords_perm
            ).astype(float)
        )
        # calculate distances from each voxel to each partitioning plane
        voxel_plane_distances = (
            da.dot(cart_coords_to_find_perm, planes[:, :3].T) + planes[:, 3]
        )
        # We are looking for voxels that are outside of the partitioning planes or
        # on a partitioning plane so we replace any distances that are below/inside
        # of a plane with a large number
        voxel_plane_distances = voxel_plane_distances.round(12)
        voxel_plane_distances = da.where(
            voxel_plane_distances >= 0, voxel_plane_distances, 50
        )
        voxel_plane_distances = voxel_plane_distances.compute()
        # For each atom we only need the lowest distance from the voxel to one of
        # its planes. This can greatly decrease the size of the array. We do this
        # here
        voxel_plane_distances = np.array_split(
            voxel_plane_distances, number_of_planes_per_atom, axis=1
        )
        new_distances = []
        for distance in voxel_plane_distances:
            one_distance = np.min(distance, axis=1)
            new_distances.append(one_distance)
        voxel_plane_distances = np.column_stack(new_distances)
        # At each transformation, we only want to check if the voxel is close to
        # planes belonging to atoms that are relatively close to the voxel. To do
        # this we need to calculate the distance from the voxels to each of the atoms
        # at different translations. Our goal is a bool array with the same shape as
        # our distances array that we can use as a mask.

        # First, we want to define what the maximum reasonable distance a voxel can
        # be from an atom and still have useful planes. This is chosen as the maximum
        # distance from an atom to one of its neighbors.
        # !!! This may be too large if I'm not reducing possible planes at all.

        max_atom_dists = []
        # breakpoint()
        for site_df in self.partitioning.values():
            dists = site_df["dist"]
            neigh_site = site_df.loc[len(site_df)-1,"neigh_index"]
            neigh_radius = self.partitioning[neigh_site].loc[0,"radius"]
            max_atom_dists.append(max(dists)-neigh_radius)

        max_atom_dists = np.array(max_atom_dists)
        max_atom_dists = np.tile(max_atom_dists, (len(voxel_frac_coords), 1))
        all_valid_atoms = []
        for x, y, z in permutations:
            # get the new coords for this translation
            new_frac_coords = voxel_frac_coords.copy()
            new_frac_coords[:, 0] += x
            new_frac_coords[:, 1] += y
            new_frac_coords[:, 2] += z
            # calculate the distances to each atom from each voxel
            cart_coords_to_find = self.charge_grid.get_cart_coords_from_frac_full_array(
                new_frac_coords
            )
            site_carts = structure.cart_coords
            voxel_atom_distances = np.sqrt(
                np.sum((cart_coords_to_find[:, None, :] - site_carts) ** 2, axis=2)
            )

            # get which atoms should be considered for each translated voxel
            potential_atoms = np.where(
                voxel_atom_distances < max_atom_dists, True, False
            )
            all_valid_atoms.append(potential_atoms)
        all_valid_atoms = np.concatenate(all_valid_atoms)

        # Replace any distance values where the atom should not be considered with
        # a very high value
        voxel_plane_distances = np.where(all_valid_atoms, voxel_plane_distances, 50)
        # Currently, we have the shortest distance from a voxel to an atoms planes.
        # We have this value for each translation of the voxel or 50, if the atom
        # should not be considered at this translation. We want to consider any of
        # the distances that aren't 50 for each voxel so we can take the shortest
        # value to each atom plane regardless of the translation. This compresses
        # the array into a shape with the number of unassigned voxels by the
        # number of atoms.
        voxel_plane_distances_split = np.array_split(voxel_plane_distances, 27)
        voxel_plane_distances_compressed = np.min(
            np.stack(voxel_plane_distances_split), axis=0
        )
        # Now we want to find the minimum value in each row. The atoms that have
        # this distance will be assigned part of or all of the voxel
        voxel_plane_min_distances = np.min(voxel_plane_distances_compressed, axis=1)
        # We tile to get the same shape as our voxel_plane_distances_compressed array
        voxel_plane_min_distances = np.tile(
            voxel_plane_min_distances, (len(structure), 1)
        ).T
        # Now we assign each voxel by checking which distances match the relavent
        # values
        multi_site_assignments = np.where(
            voxel_plane_distances_compressed == voxel_plane_min_distances, 1, 0
        )
        return multi_site_assignments

    def get_multi_site_assignments_from_frac_coords_with_memory_handling(
        self,
        voxel_frac_coords: ArrayLike,
    ):
        """
        Gets the multi-site assignments for an arbitrary number of voxels described
        by their fractional coordinates. Takes available memory into account
        and divides the voxels into chunks to perform operations.

        Args:
            voxel_frac_coords (ArrayLike):
                An N,3 array of fractional coordinates corresponding to the voxels
                to assign sites to

        Returns:
            A 2D array of site assignments with indices (i,j) where i is the
            voxel index and j is the site index.
        """
        logging.info("Calculating voxel assignments outside partitioning")
        partitioning = self.partitioning
        # determine how much memory is available. Then calculate how many distance
        # calculations would be possible to do at once with this much memory.
        available_memory = psutil.virtual_memory().available / (1024**2)

        # 0.00084 is selected as being 10% larger than the maximum memory needed
        handleable_plane_distance_calcs_dask = available_memory / 0.00084
        plane_distances_to_calc = len(voxel_frac_coords) * sum(
            [len(i) for i in partitioning.values()]
        )

        # calculate the number of chunks the voxel array should be split into to not
        # overload the memory. Then split the array by this number
        split_num = math.ceil(
            plane_distances_to_calc / handleable_plane_distance_calcs_dask
        )

        split_voxel_frac_coords = np.array_split(voxel_frac_coords, split_num, axis=0)
        # create an array to store results
        voxel_results_array = []
        # for each split, calculate the results and add to the end of our results
        for chunk, split_voxel_array in enumerate(split_voxel_frac_coords):
            logging.info(
                f"Calculating multi-site assignments for voxel chunk {chunk+1}/{split_num}"
            )
            split_result = self.get_multi_site_voxel_assignments_from_frac_coords(
                voxel_frac_coords=split_voxel_array,
            )
            voxel_results_array.append(split_result)
        voxel_results_array = np.concatenate(voxel_results_array)
        return voxel_results_array

    def get_multi_site_voxel_assignments(self, all_site_voxel_assignments: ArrayLike):
        """
        Gets the voxel assignments for voxels that are not split by a plane.

        Args:
            all_site_voxel_assignments (ArrayLike):
                A 1D array of integers representing the site assignments for
                each voxel in the grid.

        Returns:
            A 2D array of site assignments with indices (i,j) where i is the
            voxel index and j is the site index.
        """
        all_voxel_assignments = all_site_voxel_assignments.copy()
        # Search for unassigned voxels
        unassigned_indices = np.where(all_voxel_assignments == 0)[0]
        if len(unassigned_indices) > 0:
            all_voxel_frac_coords = self.all_voxel_frac_coords
            frac_coords_to_find = all_voxel_frac_coords[unassigned_indices]
            multi_site_voxel_assignments = (
                self.get_multi_site_assignments_from_frac_coords_with_memory_handling(
                    frac_coords_to_find
                )
            )
            return multi_site_voxel_assignments
        else:
            logging.info("No sites found outside partitioning.")
            return np.array([])
