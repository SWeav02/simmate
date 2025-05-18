# -*- coding: utf-8 -*-

# from simmate.apps.bader.toolkit import Grid
# import numpy as np
# from numpy.typing import NDArray
# from itertools import product
# from simmate.apps.badelf.utilities.helper_classes import UnionFind

# from scipy.interpolate import RegularGridInterpolator

# def get_connected_maxima(
#         maxima_vox_coords: NDArray, 
#         flat_voxel_coords: NDArray,
#         voxel_indices: NDArray,
#         grid: Grid,
#             ):
#     """
#     Finds maxima that neighbor each other within the 26 neighbors, then
#     uses cubic interpolation to determine if these should be combined to
#     one maximum. Finally, returns a list of groups of maxima that should
#     be combined to one.
#     """
#     data = grid.total
#     # find maxima that match
#     maxima_indices = voxel_indices[maxima_vox_coords[:,0],maxima_vox_coords[:,1],maxima_vox_coords[:,2]]
#     adjacent_pairs = []
#     for transform in product([-1,0,1], repeat=3):
#         if transform == (0,0,0):
#             continue
#         trans_coords = maxima_vox_coords + transform
#         trans_coords %= grid.shape
#         neigh_labels = voxel_indices[trans_coords[:,0],trans_coords[:,1],trans_coords[:,2]]
#         # find where labels match others
#         mask = np.isin(neigh_labels, maxima_indices)
#         voxels = maxima_indices[mask]
#         neighbors = neigh_labels[mask]
#         pairs = np.column_stack((voxels, neighbors))
#         adjacent_pairs.extend(pairs)
#     adjacent_pairs = np.array(adjacent_pairs)
#     # remove repeats
#     adjacent_pairs = np.sort(adjacent_pairs, axis=1)
#     adjacent_pairs = np.unique(adjacent_pairs, axis=0)
    
#     # Now for each pair we get 3 points between them
#     maxima1 = flat_voxel_coords[adjacent_pairs[:,0]]
#     maxima2 = flat_voxel_coords[adjacent_pairs[:,1]]
#     maxima_diff = maxima2-maxima1
#     interp_num = 3
#     interp_frac = 1/(interp_num+1)
#     interp_coords = [maxima1]
#     for i in range(interp_num+1):
#         new_coords = maxima1 + (i+1)*interp_frac*maxima_diff
#         new_coords %= grid.shape
#         interp_coords.append(new_coords)
    
#     # We combine all of these points so we only need to use the regular grid
#     # interpolator once
#     interp_coords_combined = np.concatenate(interp_coords)
#     padded_interp_coords_combined = interp_coords_combined + 1
#     a,b,c = grid.get_padded_grid_axes(1)
#     padded_data = np.pad(data,pad_width=1,mode="wrap")
#     fn = RegularGridInterpolator((a,b,c),padded_data,method="cubic")
#     values = fn(padded_interp_coords_combined)
    
#     # split back to 5 arrays, then combine to 2D array with each row being a line
#     interp_values = np.split(values, 5)
#     interp_values = np.column_stack(interp_values)
#     # get the number of maxima along each line
#     left = interp_values[:, :-2]
#     center = interp_values[:, 1:-1]
#     right = interp_values[:, 2:]
#     interior_maxima = (center > left) & (center > right)
#     interior_counts = np.sum(interior_maxima, axis=1)
#     left_edge_max = interp_values[:, 0] > interp_values[:, 1]
#     right_edge_max = interp_values[:, -1] > interp_values[:, -2]
#     total_maxima = interior_counts + left_edge_max.astype(int) + right_edge_max.astype(int)
#     # Get the union of the maxima that are connected
#     unions = adjacent_pairs[total_maxima==1]
#     uf = UnionFind()
#     for x, y in unions:
#         uf.union(x, y)
#     for x in maxima_indices:
#         uf.union(x,x)
#     maxima_groups = uf.groups()
#     maxima_groups = [np.sort(list(group)) for group in maxima_groups]
#     # We also want the fractional coords of each group of maxima. 
#     new_maxima_coords = []
#     for group in maxima_groups:
#         group_vox_coords = flat_voxel_coords[group]
#         group_data = data[group_vox_coords[:,0],group_vox_coords[:,1],group_vox_coords[:,2]]
#         max_vox_coords = group_vox_coords[group_data==group_data.max()]
#         max_frac_coords = grid.get_frac_coords_from_vox(max_vox_coords)
#         # This may result in multiple frac coords. If this is the case, we combine
#         # them
#         if max_frac_coords.ndim == 1:
#             new_maxima_coords.append(max_frac_coords)
#             continue
#         frac_diffs = max_frac_coords - max_frac_coords[0]
#         # wrap values above 0.5 and below -0.5
#         frac_diffs -= frac_diffs.round()
#         avg_frac_diffs = np.mean(frac_diffs, axis=0)
#         new_max_frac_coord = avg_frac_diffs + max_frac_coords[0]
#         new_maxima_coords.append(new_max_frac_coord)
    
#     return maxima_groups, np.array(new_maxima_coords)

# def get_basin_charge_volume_from_weights(
#     weight_array: np.ndarray,      # shape (n_voxels, n_basins)
#     flat_charge_array: np.ndarray, # shape (n_voxels,)
#     voxel_volume: float,
# ):
#     # We transform the array so that each row is one basin. We then compute the
#     # dot product with the charge array for each of these rows.
#     # charge_array[j] = sum_i(weight[i,j] * charge[i])
#     charge_array = np.dot(weight_array.T, flat_charge_array)
#     # charge_array = weight_array.T.dot(flat_charge_array)

#     # Similar for volume
#     # volume_array[j] = sum_i(weight[i,j] * voxel_volume)
#     volume_array = np.sum(weight_array, axis=0) * voxel_volume
#     # volume_array = weight_array.sum(axis=0) * voxel_volume

#     return charge_array, volume_array