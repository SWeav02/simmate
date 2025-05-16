# -*- coding: utf-8 -*-

import itertools
import logging
from functools import cached_property
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice
from pymatgen.io.vasp import VolumetricData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label, zoom, binary_dilation
from scipy.spatial import Voronoi

from simmate.toolkit import Structure

class Grid(VolumetricData):
    """
    This class is a wraparound for Pymatgen's VolumetricData class with additional
    properties and methods useful to the badelf algorithm
    """

    @property
    def total(self):
        return self.data["total"]

    @total.setter
    def total(self, new_total):
        self.data["total"] = new_total

    @property
    def diff(self):
        return self.data.get("diff")

    @diff.setter
    def diff(self, new_diff):
        self.data["diff"] = new_diff

    @property
    def shape(self):
        return np.array(self.total.shape)

    @property
    def matrix(self):
        """
        A 3x3 matrix defining the a, b, and c sides of the unit cell
        """
        return self.structure.lattice.matrix

    @property
    def a(self):
        """
        The cartesian coordinates for the lattice vector "a"
        """
        return self.matrix[0]

    @property
    def b(self):
        """
        The cartesian coordinates for the lattice vector "b"
        """
        return self.matrix[1]

    @property
    def c(self):
        """
        The cartesian coordinates for the lattice vector "c"
        """
        return self.matrix[2]

    @property
    def frac_coords(self):
        """
        Array of fractional coordinates for each atom.
        """
        return self.structure.frac_coords

    @property
    def all_voxel_frac_coords(self):
        """
        The fractional coordinates for all of the voxels in the grid
        """
        a, b, c = self.shape
        voxel_indices = np.indices(self.shape).reshape(3, -1).T
        frac_coords = voxel_indices.copy().astype(float)
        frac_coords[:, 0] /= a
        frac_coords[:, 1] /= b
        frac_coords[:, 2] /= c
        return frac_coords

    @cached_property
    def voxel_dist_to_origin(self):
        frac_coords = self.all_voxel_frac_coords
        cart_coords = self.get_cart_coords_from_frac(frac_coords)
        corners = [
            np.array([0, 0, 0]),
            self.a,
            self.b,
            self.c,
            self.a + self.b,
            self.a + self.c,
            self.b + self.c,
            self.a + self.b + self.c,
        ]
        distances = []
        for corner in corners:
            voxel_distances = np.linalg.norm(cart_coords - corner, axis=1).round(6)
            distances.append(voxel_distances)
        min_distances = np.min(np.column_stack(distances), axis=1)
        min_distances = min_distances.reshape(self.shape)
        return min_distances

    @property
    def voxel_volume(self):
        """
        The volume of each voxel in the grid
        """
        volume = self.structure.volume
        voxel_num = np.prod(self.shape)
        return volume / voxel_num

    @property
    def voxel_num(self):
        """
        The number of voxels in the grid
        """
        return self.shape.prod()

    @property
    def max_voxel_dist(self):
        """
        Finds the maximum distance a voxel can be from a dividing plane that still
        allows for the possibility that the voxel is intercepted by the plane.

        Returns:
            The maximum distance a voxel can be from a dividing plane and still
            be intercepted by the plane.
        """
        # We need to find the coordinates that make up a single voxel. This
        # is just the cartesian coordinates of the unit cell divided by
        # its grid size
        end = [0, 0, 0]
        vox_a = [x / self.shape[0] for x in self.a]
        vox_b = [x / self.shape[1] for x in self.b]
        vox_c = [x / self.shape[2] for x in self.c]
        # We want the three other vertices on the other side of the voxel. These
        # can be found by adding the vectors in a cycle (e.g. a+b, b+c, c+a)
        vox_a1 = [x + x1 for x, x1 in zip(vox_a, vox_b)]
        vox_b1 = [x + x1 for x, x1 in zip(vox_b, vox_c)]
        vox_c1 = [x + x1 for x, x1 in zip(vox_c, vox_a)]
        # The final vertex can be found by adding the last unsummed vector to any
        # of these
        end1 = [x + x1 for x, x1 in zip(vox_a1, vox_c)]
        # The center of the voxel sits exactly between the two ends
        center = [(x + x1) / 2 for x, x1 in zip(end, end1)]
        # Shift each point here so that the origin is the center of the
        # voxel.
        voxel_vertices = []
        for vector in [
            center,
            end,
            vox_a,
            vox_b,
            vox_c,
            vox_a1,
            vox_b1,
            vox_c1,
            end,
        ]:
            new_vector = [(x - x1) for x, x1 in zip(vector, center)]
            voxel_vertices.append(new_vector)

        # Now we need to find the maximum distance from the center of the voxel
        # to one of its edges. This should be at one of the vertices.
        # We can't say for sure which one is the largest distance so we find all
        # of their distances and return the maximum
        max_distance = max([np.linalg.norm(vector) for vector in voxel_vertices])
        return max_distance
    
    @property
    def voxel_voronoi_facets(self):
        """
        The transformations, areas, and vertices of the voronoi surface
        between any points and its neighbors in the grid
        """
        voxel_positions = np.array(list(itertools.product([-1,0,1], repeat=3)))
        cart_positions = self.get_cart_coords_from_vox(voxel_positions)
        voronoi = Voronoi(cart_positions)
        site_neighbors = []
        facet_vertices = []
        facet_areas = []
        
        def facet_area(vertices):
            # You can use a 2D or 3D area formula for a polygon
            # Here we assume the vertices are in a 2D plane for simplicity
            # For 3D, a more complicated approach (e.g., convex hull or triangulation) is needed
            p0 = np.array(vertices[0])
            area = 0
            for i in range(1, len(vertices)-1):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[i+1])
                area += np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0
            return area
        
        for i, neighbor_pair in enumerate(voronoi.ridge_points):
            if 13 in neighbor_pair:
                neighbor = [i for i in neighbor_pair if i != 13][0]
                vertex_indices = voronoi.ridge_vertices[i]
                vertices = voronoi.vertices[vertex_indices]
                area = facet_area(vertices)
                site_neighbors.append(neighbor)
                facet_vertices.append(vertices)
                facet_areas.append(area)
        transforms = voxel_positions[np.array(site_neighbors)]
        cart_transforms = cart_positions[np.array(site_neighbors)]
        transform_dists = np.linalg.norm(cart_transforms,axis=1)
        return transforms, transform_dists, np.array(facet_areas), facet_vertices

    @property
    def permutations(self):
        """
        The permutations for translating a voxel coordinate to nearby unit
        cells. This is necessary for the many voxels that will not be directly
        within an atoms partitioning.

        Returns:
            A list of voxel permutations unique to the grid dimensions.
        """
        a, b, c = self.shape
        permutations = [
            (t, u, v)
            for t, u, v in itertools.product([-a, 0, a], [-b, 0, b], [-c, 0, c])
        ]
        # sort permutations. There may be a better way of sorting them. I
        # noticed that generally the correct site was found most commonly
        # for the original site and generally was found at permutations that
        # were either all negative/0 or positive/0
        permutations_sorted = []
        for item in permutations:
            if all(val <= 0 for val in item):
                permutations_sorted.append(item)
            elif all(val >= 0 for val in item):
                permutations_sorted.append(item)
        for item in permutations:
            if item not in permutations_sorted:
                permutations_sorted.append(item)
        permutations_sorted.insert(0, permutations_sorted.pop(7))
        return permutations_sorted

    @property
    def voxel_resolution(self):
        volume = self.structure.volume
        number_of_voxels = self.shape.prod()
        return number_of_voxels / volume

    @cached_property
    def symmetry_data(self):
        return SpacegroupAnalyzer(self.structure).get_symmetry_dataset()

    @property
    def equivalent_atoms(self):
        return self.symmetry_data.equivalent_atoms

    def interpolate_value_at_frac_coords(
        self, frac_coords, method: str = "linear"
    ) -> list[float]:
        coords = self.get_voxel_coords_from_frac(np.array(frac_coords))
        padded_data = np.pad(self.total, 10, mode="wrap")

        # interpolate grid to find values that lie between voxels. This is done
        # with a cruder interpolation here and then the area close to the minimum
        # is examened more closely with a more rigorous interpolation in
        # get_line_frac_min
        a, b, c = self.get_padded_grid_axes(10)
        fn = RegularGridInterpolator((a, b, c), padded_data, method=method)
        values = []
        for pos in coords:
            adjusted_pos = [x + 10 for x in pos]
            value = float(fn(adjusted_pos))
            values.append(value)
        return values

    def get_slice_around_voxel_coord(
        self, voxel_coords: NDArray, neighbor_size: int = 1
    ):
        slices = []
        for dim, c in zip(self.shape, voxel_coords):
            idx = np.arange(c - neighbor_size, c + 2) % dim
            idx = idx.astype(int)
            slices.append(idx)
        return self.total[np.ix_(slices[0], slices[1], slices[2])]

    def get_maxima_near_frac_coord(self, frac_coords: NDArray, neighbor_size: int = 2):
        coords = self.get_voxel_coords_from_frac(frac_coords).astype(int)
        init_coords = coords + 1
        new_coords = coords.copy()
        # First hill climb until the voxel max is reached
        cycles = 0
        while not np.allclose(init_coords, new_coords, rtol=0, atol=0.001) and cycles < 100:
            init_coords = new_coords.copy()
            subset = self.get_slice_around_voxel_coord(init_coords, neighbor_size)
            max_val = subset.max()
            max_loc = np.array(np.where(subset == max_val))
            res = max_loc.mean(axis=1).round()
            local_offset = res - neighbor_size  # shift from subset center
            voxel_coords = new_coords + local_offset
            new_coords = voxel_coords % np.array(self.shape)
            cycles += 1
        # Now get the average in the area
        # Use np.ix_ to get the full 3D cube using broadcasting
        subset = self.get_slice_around_voxel_coord(new_coords, neighbor_size)
        max_val = subset.max()
        max_loc = np.array(np.where(subset == max_val))
        res = max_loc.mean(axis=1)
        local_offset = res - neighbor_size  # shift from subset center
        voxel_coords = new_coords + local_offset
        new_coords = voxel_coords % np.array(self.shape)
        # print(self.get_frac_coords_from_vox(new_coords))

        new_frac_coords = self.get_frac_coords_from_vox(new_coords)

        return new_frac_coords

    def get_2x_supercell(self, data: NDArray):
        """
        Duplicates data with the same dimensions as the grid to make a 2x2x2
        supercell

        Args:
            data (NDArray):
                The data to duplicate. Must have the same dimensions as the
                grid.

        Returns:
            The duplicated data
        """
        new_data = np.tile(data, (2,2,2))
        return new_data

    def get_voxels_in_radius(self, radius: float, voxel: NDArray):
        """
        Gets the indices of the voxels in a radius around a voxel

        Args:
            radius (float):
                The radius in Angstroms around the voxel

            voxel (NDArray):
                The voxel coordinates of the voxel to find the sphere around

        Returns:
            The voxel indices of the voxels within the provided radius
        """
        voxel = np.array(voxel)
        # Get the distance from each voxel to the origin
        voxel_distances = self.voxel_dist_to_origin

        # Get the indices that are within the radius
        sphere_indices = np.where(voxel_distances <= radius)
        sphere_indices = np.column_stack(sphere_indices)

        # Get indices relative to the voxel
        sphere_indices = sphere_indices + voxel
        # adjust voxels to wrap around grid
        # line = [[round(float(a % b), 12) for a, b in zip(position, grid_data.shape)]]
        new_x = (sphere_indices[:, 0] % self.shape[0]).astype(int)
        new_y = (sphere_indices[:, 1] % self.shape[1]).astype(int)
        new_z = (sphere_indices[:, 2] % self.shape[2]).astype(int)
        sphere_indices = np.column_stack([new_x, new_y, new_z])
        # return new_x, new_y, new_z
        return sphere_indices

    def get_voxels_transformations_to_radius(self, radius: float):
        """
        Gets the transformations required to move from a voxel to the voxels
        surrounding it within the provided radius

        Args:
            radius (float):
                The radius in Angstroms around the voxel

        Returns:
            An array of transformations to add to a voxel to get to each of the
            voxels within the radius surrounding it
        """
        # Get voxels around origin
        voxel_distances = self.voxel_dist_to_origin
        # sphere_grid = np.where(voxel_distances <= radius, True, False)
        # eroded_grid = binary_erosion(sphere_grid)
        # shell_indices = np.where(sphere_grid!=eroded_grid)
        shell_indices = np.where(voxel_distances <= radius)
        # Now we want to translate these indices to next to the corner so that
        # we can use them as transformations to move a voxel to the edge
        final_shell_indices = []
        for a, x in zip(self.shape, shell_indices):
            new_x = x - a
            abs_new_x = np.abs(new_x)
            new_x_filter = abs_new_x < x
            final_x = np.where(new_x_filter, new_x, x)
            final_shell_indices.append(final_x)

        return np.column_stack(final_shell_indices)

    def get_padded_grid_axes(self, padding: int = 0):
        """
        Gets the the possible indices for each dimension of a padded grid.
        e.g. if the original charge density grid is 20x20x20, and is padded
        with one extra layer on each side, this function will return three
        arrays with integers from 0 to 21.

        Args:
            padding (int):
                The amount the grid has been padded

        Returns:
            three arrays with lengths the same as the grids shape
        """
        grid = self.total
        a = np.linspace(
            0,
            grid.shape[0] + (padding - 1) * 2 + 1,
            grid.shape[0] + padding * 2,
        )
        b = np.linspace(
            0,
            grid.shape[1] + (padding - 1) * 2 + 1,
            grid.shape[1] + padding * 2,
        )
        c = np.linspace(
            0,
            grid.shape[2] + (padding - 1) * 2 + 1,
            grid.shape[2] + padding * 2,
        )
        return a, b, c

    def copy(self):
        """
        Convenience method to get a copy of the grid.

        Returns:
            A copy of the Grid.
        """
        return self.__class__(
            self.structure.copy(),
            self.data.copy(),
        )

    @classmethod
    def from_file(cls, grid_file: str | Path):
        """Create a grid instance using a CHGCAR or ELFCAR file. This uses
        pymatgens implementation of loading a VASP grid and is usually
        slower than the 'from_vasp' method.

        Args:
            grid_file (string):
                The file the instance should be made from. Should be a VASP
                CHGCAR or ELFCAR type file.

        Returns:
            Grid from the specified file.
        """
        logging.info(f"Loading {grid_file} from file")
        # Create string to add structure to.
        poscar, data, _ = cls.parse_file(grid_file)

        return Grid(poscar.structure, data)
    
    @classmethod
    def from_vasp(cls, filename: str | Path):
        """
        Create a grid instance using a CHGCAR or ELFCAR file
        
        Args:
            filename (string):
                The file the instance should be made from. Should be a VASP
                CHGCAR or ELFCAR type file.

        Returns:
            Grid from the specified file.
        """
        # ensure we have a path object
        filename = Path(filename)
        with open(filename, 'r') as f:
            # Read header lines first
            next(f)  # line 0
            scale = float(next(f).strip())  # line 1
        
            lattice_matrix = np.array([[float(x) for x in next(f).split()] for _ in range(3)]) * scale
        
            atom_types = next(f).split()
            atom_counts = list(map(int, next(f).split()))
            total_atoms = sum(atom_counts)
        
            # Skip the 'Direct' or 'Cartesian' line
            next(f)
        
            coords = np.array([
                list(map(float, next(f).split()))
                for _ in range(total_atoms)
            ])
        
            lattice = Lattice(lattice_matrix)
            atom_list = [elem for typ, count in zip(atom_types, atom_counts) for elem in [typ]*count]
            structure = Structure(lattice=lattice, species=atom_list, coords=coords)
        
            # Read the FFT grid line
            # skip empty line
            next(f)
            nx, ny, nz = map(int, next(f).split())
            ngrid = nx * ny * nz
        
            # Read the rest of the file as a single string to avoid Python loop overhead
            # Read the remainder of the file as a single string
            rest = f.read()
            
            # Truncate everything after the word "augmentation"
            cutoff_index = rest.lower().find("augmentation")
            if cutoff_index != -1:
                rest = rest[:cutoff_index]
            
            # Split into values and convert to float
            data_array = np.fromiter((float(x) for x in rest.split()), dtype=float)
            
            if len(data_array) > ngrid:
                delete_indices = np.arange(ngrid, ngrid+3)
                data_array = np.delete(data_array, delete_indices)
            
            # Fast check for spin-polarized case
            if len(data_array) == ngrid:
                total = data_array.reshape((nx, ny, nz), order='F')
                data = {"total": total}
            elif len(data_array) == 2 * ngrid:
                total = data_array[:ngrid].reshape((nx, ny, nz), order='F')
                diff = data_array[ngrid:].reshape((nx, ny, nz), order='F')
                data = {"total": total, "diff": diff}
            else:
                raise ValueError("Unexpected number of data points: does not match grid size.")
        
        return Grid(structure, data)

    def get_atoms_in_volume(self, volume_mask):
        """
        Checks if an atom is within this volume. This only checks the
        area immediately around the atom, so outer core basins may not
        be caught by this.
        """
        site_voxel_coords = self.get_voxel_coords_from_frac(
            self.structure.frac_coords
            ).astype(int)
        atom_values = []
        for atom_idx, atom_coords in enumerate(site_voxel_coords):
            site_value = volume_mask[atom_coords[0], atom_coords[1], atom_coords[2]]
            if site_value:
                atom_values.append(atom_idx)
        return atom_values

    def get_atoms_surrounded_by_volume(self, mask, return_type: bool = False):
        """
        Checks if a list of basins completely surround any of the atoms
        in the structure. This method uses scipy's ndimage package to
        label features in the grid combined with a supercell to check
        if atoms identical through translation are connected.
        """
        # first we get any atoms that are within the mask itself. These won't be
        # found otherwise because they will always sit in unlabeled regions.
        structure = np.ones([3, 3, 3])
        dilated_mask = binary_dilation(mask, structure)
        init_atoms = self.get_atoms_in_volume(dilated_mask)
        # Now we create a supercell of the mask so we can check connections to
        # neighboring cells. This will be used to check if the feature connects
        # to itself in each direction
        dilated_supercell_mask = self.get_2x_supercell(dilated_mask)
        # We also get an inversion of this mask. This will be used to check if
        # the mask surrounds each atom. To do this, we use the dilated supercell
        # We do this to avoid thin walls being considered connections
        # in the inverted mask
        inverted_mask = dilated_supercell_mask == False
        # Now we use use scipy to label unique features in our masks

        inverted_feature_supercell = self.label(inverted_mask, structure)

        # if an atom was fully surrounded, it should sit inside one of our labels.
        # The same atom in an adjacent unit cell should have a different label.
        # To check this, we need to look at the atom in each section of the supercell
        # and see if it has a different label in each.
        # Similarly, if the feature is disconnected from itself in each unit cell
        # any voxel in the feature should have different labels in each section.
        # If not, the feature is connected to itself in multiple directions and
        # must surround many atoms.
        transformations = np.array(list(itertools.product([0,1], repeat=3)))
        transformations = self.get_voxel_coords_from_frac(transformations)
        # Check each atom to determine how many atoms it surrounds
        surrounded_sites = []
        for i, site in enumerate(self.structure):
            # Get the voxel coords of each atom in their equivalent spots in each
            # quadrant of the supercell
            frac_coords = site.frac_coords
            voxel_coords = self.get_voxel_coords_from_frac(frac_coords)
            transformed_coords = (transformations + voxel_coords).astype(int)
            # Get the feature label at each transformation. If the atom is not surrounded
            # by this basin, at least some of these feature labels will be the same
            features = inverted_feature_supercell[
                transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2]
            ]
            if len(np.unique(features)) == 8:
                # The atom is completely surrounded by this basin and the basin belongs
                # to this atom
                surrounded_sites.append(i)
        surrounded_sites.extend(init_atoms)
        surrounded_sites = np.unique(surrounded_sites)
        types = []
        for site in surrounded_sites:
            if site in init_atoms:
                types.append(0)
            else:
                types.append(1)
        if return_type:
            return surrounded_sites, types
        return surrounded_sites
    
    def check_if_infinite_feature(self, mask: NDArray) -> bool:
        """
        Checks if a feature extends infinitely in at least one direction
        """
        # First we check that there is at least one feature in the mask. If not
        # we return False as there is no feature.
        if (~mask).all():
            return False
        
        structure = np.ones([3, 3, 3])
        # Now we create a supercell of the mask so we can check connections to
        # neighboring cells. This will be used to check if the feature connects
        # to itself in each direction
        supercell_mask = self.get_2x_supercell(mask)
        # Now we use use scipy to label unique features in our masks
        feature_supercell = self.label(supercell_mask, structure)
        # Now we check if we have the same label in any of the adjacent unit
        # cells. If yes we have an infinite feature.
        transformations = np.array(list(itertools.product([0,1], repeat=3)))
        transformations = self.get_voxel_coords_from_frac(transformations)
        initial_coord = np.argwhere(mask)[0]
        transformed_coords = (transformations + initial_coord).astype(int)

        # Get the feature label at each transformation. If the atom is not surrounded
        # by this basin, at least some of these feature labels will be the same
        features = feature_supercell[
            transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2]
        ]
        
        inf_feature = False
        # If any of the transformed coords have the same feature value, this
        # feature extends between unit cells in at least 1 direction and is
        # infinite. This corresponds to the list of unique features being below
        # 8
        if len(np.unique(features)) < 8:
            inf_feature = True

        return inf_feature
    
    def regrid(
        self,
        desired_resolution: int = 1200,
        new_shape: np.array = None,
        order: int = 3,
    ):
        """
        Returns a new grid resized using scipy's ndimage.zoom method

        Args:
            desired_resolution (int):
                The desired resolution in voxels/A^3.
            new_shape (NDArray):
                The new array shape. Takes precedence over desired_resolution.
            order (int):
                The order of spline interpolation to use.

        Returns:
            Changes the grid data in place.
        """
        # Get data
        total = self.total
        diff = self.diff

        # # Get the lattice unit vectors as a 3x3 array
        # lattice_array = self.matrix

        # get the original grid size and lattice volume.
        shape = self.shape
        volume = self.structure.volume

        if new_shape is None:
            # calculate how much the number of voxels along each unit cell must be
            # multiplied to reach the desired resolution.
            scale_factor = ((desired_resolution * volume) / shape.prod()) ** (1 / 3)

            # calculate the new grid shape. round up to the nearest integer for each
            # side
            new_shape = np.around(shape * scale_factor).astype(np.int32)

        # get the factor to zoom by
        zoom_factor = new_shape / shape
        # get the new total data
        new_total = zoom(
            total, zoom_factor, order=order, mode="grid-wrap", grid_mode=True
        )  # , prefilter=False,)
        # if the diff exists, get the new diff data
        if diff is not None:
            new_diff = zoom(
                diff, zoom_factor, order=order, mode="grid-wrap", grid_mode=True
            )  # , prefilter=False,)
            data = {"total": new_total, "diff": new_diff}
        else:
            # get the new data dict and return a new grid
            data = {"total": new_total}

        return Grid(self.structure, data)

    def split_to_spin(self, data_type: Literal["elf", "charge"] = "elf"):
        """
        Splits the grid to spin up and spin down contributions
        """
        # first check if the grid has spin parts
        if not self.is_spin_polarized:
            raise Exception(
                "Only one set of data detected. The grid cannot be split into spin up and spin down"
            )
        # Now we get the separate data parts. If the data is ELF, the parts are
        # stored as total=spin up and diff = spin down
        if data_type == "elf":
            spin_up_data = self.total.copy()
            spin_down_data = self.diff.copy()
        elif data_type == "charge":
            spin_data = self.spin_data
            # pymatgen uses some custom class as keys here
            for key in spin_data.keys():
                if key.value == 1:
                    spin_up_data = spin_data[key].copy()
                elif key.value == -1:
                    spin_down_data = spin_data[key].copy()

        # convert to dicts
        spin_up_data = {"total": spin_up_data}
        spin_down_data = {"total": spin_down_data}

        spin_up_grid = self.__class__(
            self.structure.copy(),
            spin_up_data,
        )
        spin_down_grid = self.__class__(
            self.structure.copy(),
            spin_down_data,
        )

        return spin_up_grid, spin_down_grid

    @classmethod
    def sum_grids(cls, grid1, grid2):
        """
        Takes in two grids and returns a single grid summing their values.

        Args:
            grid1 (Grid):
                The first grid to sum

            grid2 (Grid):
                The second grid to sum

        Returns:
            A Grid object with both the total and diff parts summed

        """
        if not np.all(grid1.shape == grid2.shape):
            logging.exception("Grids must have the same size.")
        total1 = grid1.total
        diff1 = grid1.diff

        total2 = grid2.total
        diff2 = grid2.diff

        total = total1 + total2
        if diff1 is not None and diff2 is not None:
            diff = diff1 + diff2
            data = {"total": total, "diff": diff}
        else:
            data = {"total": total, "diff": None}

        # Note that we copy the first grid here rather than making a new grid
        # instance because we want to keep any useful information such as whether
        # the grid is spin polarized or not.
        new_grid = grid1.copy()
        new_grid.data = data
        return new_grid

    @staticmethod
    def label(input: NDArray, structure: NDArray = np.ones([3, 3, 3])):
        """
        Uses scipy's ndimage package to label an array, and corrects for
        periodic boundaries
        """
        if structure is not None:
            labeled_array, _ = label(input, structure)
            if len(np.unique(labeled_array)) == 1:
                # there is one feature or no features
                return labeled_array
            # Features connected through opposite sides of the unit cell should
            # have the same label, but they don't currently. To handle this, we
            # pad our featured grid, re-label it, and check if the new labels
            # contain multiple of our previous labels.
            padded_featured_grid = np.pad(labeled_array, 1, "wrap")
            relabeled_array, label_num = label(padded_featured_grid, structure)
        else:
            labeled_array, _ = label(input)
            padded_featured_grid = np.pad(labeled_array, 1, "wrap")
            relabeled_array, label_num = label(padded_featured_grid)

        # We want to keep track of which features are connected to each other
        unique_connections = [[] for i in range(len(np.unique(labeled_array)))]

        for i in np.unique(relabeled_array):
            # for i in range(label_num):
            # Get the list of features that are in this super feature
            mask = relabeled_array == i
            connected_features = list(np.unique(padded_featured_grid[mask]))
            # Iterate over these features. If they exist in a connection that we
            # already have, we want to extend the connection to include any other
            # features in this super feature
            for j in connected_features:

                unique_connections[j].extend([k for k in connected_features if k != j])

                unique_connections[j] = list(np.unique(unique_connections[j]))

        # create set/list to keep track of which features have already been connected
        # to others and the full list of connections
        already_connected = set()
        reduced_connections = []

        # loop over each shared connection
        for i in range(len(unique_connections)):
            if i in already_connected:
                # we've already done these connections, so we skip
                continue
            # create sets of connections to compare with as we add more
            connections = set()
            new_connections = set(unique_connections[i])
            while connections != new_connections:
                # loop over the connections we've found so far. As we go, add
                # any features we encounter to our set.
                connections = new_connections.copy()
                for j in connections:
                    already_connected.add(j)
                    new_connections.update(unique_connections[j])

            # If we found any connections, append them to our list of reduced connections
            if connections:
                reduced_connections.append(sorted(new_connections))

        # For each set of connections in our reduced set, relabel all values to
        # the lowest one.
        for connections in reduced_connections:
            connected_features = np.unique(connections)
            lowest_idx = connected_features[0]
            for higher_idx in connected_features[1:]:
                labeled_array = np.where(
                    labeled_array == higher_idx, lowest_idx, labeled_array
                )

        # Now we reduce the feature labels so that they start at 0
        for i, j in enumerate(np.unique(labeled_array)):
            labeled_array = np.where(labeled_array == j, i, labeled_array)

        return labeled_array

    @staticmethod
    def periodic_center_of_mass(labels, label_vals=None) -> NDArray:
        """
        Computes center of mass for each label in a 3D periodic array.

        Args:
            labels: 3D array of integer labels
            label_vals: list/array of unique labels to compute (default: all nonzero)

        Returns:
            A 3xN array of centers of mass
        """
        shape = labels.shape
        if label_vals is None:
            label_vals = np.unique(labels)
            label_vals = label_vals[label_vals != 0]

        centers = []
        for val in label_vals:
            # get the voxel coords for each voxel in this label
            coords = np.array(np.where(labels == val)).T  # shape (N, 3)
            # If we have no coords for this label, we skip
            if coords.shape[0] == 0:
                continue

            # From chap-gpt: Get center of mass using spherical distance
            center = []
            for i, size in enumerate(shape):
                angles = coords[:, i] * 2 * np.pi / size
                x = np.cos(angles).mean()
                y = np.sin(angles).mean()
                mean_angle = np.arctan2(y, x)
                mean_pos = (mean_angle % (2 * np.pi)) * size / (2 * np.pi)
                center.append(mean_pos)
            centers.append(center)
        centers = np.array(centers)
        centers = centers.round(6)

        return centers

    def get_critical_points(
        self, array: NDArray, threshold: float = 5e-03, return_hessian_s: bool = True
    ):
        """
        Finds the critical points in the grid. If return_hessians is true,
        the hessian matrices for each critical point will be returned along
        with their type index.
        """
        # !!! Check if padding and threshold effect final result
        # get gradient using a padded grid to handle periodicity
        padding = 2
        a, b, c = self.get_padded_grid_axes(padding)
        padded_array = np.pad(array, padding, mode="wrap")
        dx, dy, dz = np.gradient(padded_array)

        # get magnitude of the gradient
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

        # unpad the magnitude
        slicer = tuple(slice(padding, -padding) for _ in range(3))
        magnitude = magnitude[slicer]

        # now we want to get where the magnitude is close to 0. To do this, we
        # will create a mask where the magnitude is below a threshold. We will
        # then label the regions where this is true using scipy, then combine
        # the regions into one
        magnitude_mask = magnitude < threshold
        # critical_points = np.where(magnitude<threshold)
        # padded_critical_points = np.array(critical_points).T + padding

        label_structure = np.ones((3, 3, 3), dtype=int)
        labeled_magnitude_mask = self.label(magnitude_mask, label_structure)
        min_indices = []
        for idx in np.unique(labeled_magnitude_mask):
            label_mask = labeled_magnitude_mask == idx
            label_indices = np.where(label_mask)
            min_mag = magnitude[label_indices].min()
            min_indices.append(np.argwhere((magnitude == min_mag) & label_mask)[0])
        min_indices = np.array(min_indices)

        critical_points = min_indices[:, 0], min_indices[:, 1], min_indices[:, 2]

        # critical_points = self.periodic_center_of_mass(labeled_magnitude_mask)
        padded_critical_points = tuple([i + padding for i in critical_points])
        values = array[critical_points]
        # # get the value at each of these critical points
        # fn_values = RegularGridInterpolator((a, b, c), padded_array , method="linear")
        # values = fn_values(padded_critical_points)

        if not return_hessian_s:
            return critical_points, values

        # now we want to get the hessian eigenvalues around each of these points
        # using interpolation. First, we get the second derivatives
        d2f_dx2 = np.gradient(dx, axis=0)
        d2f_dy2 = np.gradient(dy, axis=1)
        d2f_dz2 = np.gradient(dz, axis=2)
        # # now create interpolation functions for each
        # fn_dx2 = RegularGridInterpolator((a, b, c), d2f_dx2, method="linear")
        # fn_dy2 = RegularGridInterpolator((a, b, c), d2f_dy2, method="linear")
        # fn_dz2 = RegularGridInterpolator((a, b, c), d2f_dz2, method="linear")
        # and calculate the hessian eigenvalues for each point
        # H00 = fn_dx2(padded_critical_points)
        # H11 = fn_dy2(padded_critical_points)
        # H22 = fn_dz2(padded_critical_points)
        H00 = d2f_dx2[padded_critical_points]
        H11 = d2f_dy2[padded_critical_points]
        H22 = d2f_dz2[padded_critical_points]
        # summarize the hessian eigenvalues by getting the sum of their signs
        hessian_eigs = np.array([H00, H11, H22])
        hessian_eigs = np.moveaxis(hessian_eigs, 1, 0)
        hessian_eigs_signs = np.where(hessian_eigs > 0, 1, hessian_eigs)
        hessian_eigs_signs = np.where(hessian_eigs < 0, -1, hessian_eigs_signs)
        # Now we get the sum of signs for each set of hessian eigenvalues
        s = np.sum(hessian_eigs_signs, axis=1)

        return critical_points, values, s

    ###########################################################################
    # The following is a series of methods that are useful for converting between
    # voxel coordinates, fractional coordinates, and cartesian coordinates.
    # Voxel coordinates go from 0 to grid_size-1. Fractional coordinates go
    # from 0 to 1. Cartesian coordinates convert to real space based on the
    # crystal lattice.
    ###########################################################################
    def get_voxel_coords_from_index(self, site):
        """
        Takes in a site index and returns the equivalent voxel grid index.

        Args:
            site (int):
                the index of the site to find the grid index for

        Returns:
            A voxel grid index as an array.

        """

        voxel_coords = [a * b for a, b in zip(self.shape, self.frac_coords[site])]
        # voxel positions go from 1 to (grid_size + 0.9999)
        return np.array(voxel_coords)

    def get_voxel_coords_from_neigh_CrystalNN(self, neigh):
        """
        Gets the voxel grid index from a neighbor atom object from CrystalNN or
        VoronoiNN

        Args:
            neigh (Neigh):
                a neighbor type object from pymatgen

        Returns:
            A voxel grid index as an array.
        """
        grid_size = self.shape
        frac = neigh["site"].frac_coords
        voxel_coords = [a * b for a, b in zip(grid_size, frac)]
        # voxel positions go from 1 to (grid_size + 0.9999)
        return np.array(voxel_coords)

    def get_voxel_coords_from_neigh(self, neigh):
        """
        Gets the voxel grid index from a neighbor atom object from the pymatgen
        structure.get_neighbors class.

        Args:
            neigh (dict):
                a neighbor dictionary from pymatgens structure.get_neighbors
                method.

        Returns:
            A voxel grid index as an array.
        """
        grid_size = self.shape
        frac_coords = neigh.frac_coords
        voxel_coords = [a * b for a, b in zip(grid_size, frac_coords)]
        # voxel positions go from 1 to (grid_size + 0.9999)
        return np.array(voxel_coords)

    def get_frac_coords_from_cart(self, cart_coords: NDArray | list):
        """
        Takes in a cartesian coordinate and returns the fractional coordinates.

        Args:
            cart_coords (NDArray):
                A cartesian coordinate.

        Returns:
            fractional coordinates as an Array
        """
        inverse_matrix = np.linalg.inv(self.matrix)

        return cart_coords @ inverse_matrix

    def get_voxel_coords_from_cart(self, cart_coords: NDArray | list):
        """
        Takes in a cartesian coordinate and returns the voxel coordinates.

        Args:
            cart_coords (NDArray): A cartesian coordinate.

        Returns:
            Voxel coordinates as an Array
        """
        frac_coords = self.get_frac_coords_from_cart(cart_coords)
        voxel_coords = self.get_voxel_coords_from_frac(frac_coords)
        return voxel_coords

    def get_cart_coords_from_frac(self, frac_coords: NDArray):
        """
        Takes in a 2D array of shape (N,3) representing fractional coordinates
        at N points and calculates the equivalent cartesian coordinates.

        Args:
            frac_coords (NDArray):
                An (N,3) shaped array of fractional coordinates

        Returns:
            An (N,3) shaped array of cartesian coordinates
        """
        
        return frac_coords @ self.matrix

    def get_frac_coords_from_vox(self, vox_coords: NDArray):
        """
        Takes in a 2D array of shape (N,3) representing voxel coordinates
        at N points and calculates the equivalent fractional coordinates.

        Args:
            vox_coords (NDArray):
                An (N,3) shaped array of voxel coordinates

        Returns:
            An (N,3) shaped array of fractional coordinates
        """
        
        return vox_coords/self.shape

    def get_voxel_coords_from_frac(self, frac_coords: NDArray):
        """
        Takes in a 2D array of shape (N,3) representing fractional coordinates
        at N points and calculates the equivalent voxel coordinates.

        Args:
            frac_coords (NDArray):
                An (N,3) shaped array of fractional coordinates

        Returns:
            An (N,3) shaped array of voxel coordinates
        """
        return frac_coords * self.shape

    def get_cart_coords_from_vox(self, vox_coords: NDArray):
        """
        Takes in a 2D array of shape (N,3) representing voxel coordinates
        at N points and calculates the equivalent cartesian coordinates.

        Args:
            frac_coords (NDArray):
                An (N,3) shaped array of voxel coordinates

        Returns:
            An (N,3) shaped array of cartesian coordinates
        """
        frac_coords = self.get_frac_coords_from_vox(vox_coords)
        return self.get_cart_coords_from_frac(frac_coords)

    def _plot_points(self, points, ax, fig, color, size: int = 20):
        """
        Plots points of form [x,y,z] using matplotlib

        Args:
            points (list): A list of points to plot
            fig: A matplotlib.pyplot.figure() instance
            ax: A matplotlib Subplot instance
            color (str): The color to plot the points
            size (int): The pt size to plot
        """
        x = []
        y = []
        z = []
        for point in points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        ax.scatter(x, y, z, c=color, s=size)

    def _plot_unit_cell(self, ax, fig):
        """
        Plots the unit cell of a structure using matplotlib

        Args:
            fig: A matplotlib.pyplot.figure() instance
            ax: A matplotlib Subplot instance
        """
        if ax is None or fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        # Get the points at the lattice vertices to plot and plot them
        a = self.a
        b = self.b
        c = self.c
        points = [np.array([0, 0, 0]), a, b, c, a + b, a + c, b + c, a + b + c]
        self._plot_points(points, ax, fig, "purple")

        # get the structure to plot.
        structure = self.structure
        species = structure.symbol_set

        # get a color map to distinguish between sites
        color_map = matplotlib.colormaps.get_cmap("tab10")
        # Go through each atom type and plot all instances with the same color
        for i, specie in enumerate(species):
            color = color_map(i)
            sites_indices = structure.indices_from_symbol(specie)
            for site in sites_indices:
                coords = structure[site].coords
                self._plot_points([coords], ax, fig, color, size=40)
