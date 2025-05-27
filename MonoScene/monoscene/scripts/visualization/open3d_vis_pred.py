import open3d as o3d
import numpy as np
import pickle
import hydra

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]

    # Create the camera mesh
    camera_mesh = o3d.geometry.TriangleMesh()
    camera_mesh.vertices = o3d.utility.Vector3dVector(np.vstack([x, y, z]).T)
    camera_mesh.triangles = o3d.utility.Vector3iVector(
        [(0, 1, 2), (0, 1, 4), (0, 3, 4), (0, 2, 3)]
    )
    camera_mesh.paint_uniform_color([0, 0, 0])  # black color for camera

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    # Create point clouds for inside and outside FOV voxels
    fov_points = o3d.geometry.PointCloud()
    fov_points.points = o3d.utility.Vector3dVector(fov_voxels[:, :3])
    fov_colors = np.zeros((fov_voxels.shape[0], 3))
    fov_colors[:, 1] = 1  # Green for inside FOV
    fov_points.colors = o3d.utility.Vector3dVector(fov_colors)

    outfov_points = o3d.geometry.PointCloud()
    outfov_points.points = o3d.utility.Vector3dVector(outfov_voxels[:, :3])
    outfov_colors = np.zeros((outfov_voxels.shape[0], 3))
    outfov_colors[:, 0] = 1  # Red for outside FOV
    outfov_points.colors = o3d.utility.Vector3dVector(outfov_colors)

    # Visualize everything using Open3D
    o3d.visualization.draw_geometries([fov_points, outfov_points, camera_mesh])

@hydra.main(config_path=None)
def main(config):
    scan = config.file
    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    fov_mask_1 = b["fov_mask_1"]
    T_velo_2_cam = b["T_velo_2_cam"]
    vox_origin = np.array([0, -25.6, -2])

    y_pred = b["y_pred"]

    if config.dataset == "kitti_360":
        # Visualize KITTI-360
        draw(
            y_pred,
            T_velo_2_cam,
            vox_origin,
            fov_mask_1,
            voxel_size=0.2,
            f=552.55426,
            img_size=(1408, 376),
            d=7,
        )
    else:
        # Visualize Semantic KITTI
        draw(
            y_pred,
            T_velo_2_cam,
            vox_origin,
            fov_mask_1,
            img_size=(1220, 370),
            f=707.0912,
            voxel_size=0.2,
            d=7,
        )


if __name__ == "__main__":
    main()

