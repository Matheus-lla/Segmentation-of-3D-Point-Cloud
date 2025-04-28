# %% [markdown]
# # Ground Plane Fitting and Scan Line Run for 3D LiDAR
# 
# Ground Plane Fitting (GPF) and Naive Baseline for 3D LiDAR Segmentation
# 
# This notebook implements ground segmentation using the Ground Plane Fitting (GPF) algorithm 
# proposed in:
# 
# "Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications"
# by D. Zermas, I. Izzat, and N. Papanikolopoulos, 2017.
# 
# The implementation also includes a naive baseline method for comparison, as well as 
# basic clustering and visualization tools.

# %% [markdown]
# # Imports

# %%
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree  # Used for spatial clustering
import open3d as o3d

# %% [markdown]
# # Hyperparameters

# %%
# --- Hyperparameters for Ground Plane Fitting (GPF) ---
NUM_LOWEST_POINTS = 2000           # Number of lowest elevation points used to estimate initial ground seed (LPR)
NUM_ITERATIONS = 5                 # Number of iterations for plane refinement in GPF
SEED_HEIGHT_THRESHOLD = 0.4        # Max height above LPR to consider a point as a ground seed
PLANE_DISTANCE_THRESHOLD = 0.2     # Max distance from plane to classify a point as ground

# --- Parameters for Scan Line Run (SLR) clustering ---
SLR_RUN_DISTANCE_THRESHOLD = 0.2   # Max distance between consecutive points in a scanline to form a run
SLR_MERGE_THRESHOLD = 1.0          # Max distance between runs in adjacent scanlines to be considered the same cluster

# %% [markdown]
# # Naive Baseline Method

# %%
def naive_ground_extractor(point_cloud: np.ndarray, num_lowest_points: int) -> np.ndarray:
    """
    Naive ground extraction method (baseline).
    
    This simple method selects the points with the lowest Z values 
    and assumes they belong to the ground surface. It does not model 
    the ground plane and is used as a baseline for comparison against 
    more robust algorithms like Ground Plane Fitting (GPF).
    
    Args:
        point_cloud (np.ndarray): N x D array of point cloud data.
        num_lowest_points (int): Number of points with lowest Z values to classify as ground.
    
    Returns:
        ground_indices (np.ndarray): Indices of the selected ground points.
    """

    # Select indices of points with the lowest Z values
    return np.argsort(point_cloud[:, 2])[:num_lowest_points]

# %% [markdown]
# # Ground Plane Fitting (GPF)

# %%
def extract_initial_seed_indices(
    point_cloud: np.ndarray, 
    num_points: int = 1000, 
    height_threshold: float = 0.4
) -> np.ndarray:
    """
    Extract initial seed points for ground plane estimation (GPF).
    
    Args:
        point_cloud (np.ndarray): N x 3 array of points (x, y, z).
        num_points (int): number of lowest Z points to average as LPR.
        height_threshold (float): threshold to select seeds close to LPR height.
    
    Returns:
        seeds_ids (np.ndarray): indices of points selected as initial seeds.
    """

    # Step 1: Sort the point cloud by Z axis (height)
    sorted_indices = np.argsort(point_cloud[:, 2])  # Get indices sorted by height
    sorted_points = point_cloud[sorted_indices]     # Apply sorting

    # Step 2: Compute LPR (Lowest Point Representative)
    lpr_height = np.mean(sorted_points[:num_points, 2])

    # Step 3: Select point ids that are within threshold distance from LPR
    mask = sorted_points[:, 2] < (lpr_height + height_threshold)
    return sorted_indices[mask]

# %%
def estimate_ground_plane(points: np.ndarray) -> "tuple[np.ndarray, float]":
    """
    Estimate the ground plane parameters using Singular Value Decomposition (SVD).
    
    Args:
        points (np.ndarray): N x 3 array (x, y, z) of seed points assumed to be on or near the ground.

    Returns:
        tuple: 
            - normal (np.ndarray): Normal vector (a, b, c) of the estimated ground plane.
            - d (float): Offset term of the estimated plane equation (ax + by + cz + d = 0).
    """
    
    # Step 1: Compute centroid of the seed points
    centroid  = np.mean(points, axis=0)
    centered_points  = points - centroid 

    # Step 2: Compute the covariance matrix of centered points
    covariance_matrix = np.cov(centered_points.T)

    # Step 3: Perform SVD on the covariance matrix to extract principal directions
    _, _, vh = np.linalg.svd(covariance_matrix)

    # Step 4: Normal vector is the direction with smallest variance (last column of V^T)
    normal = vh[-1]

    # Step 5: Compute plane bias using point-normal form: ax + by + cz + d = 0
    d = -np.dot(normal, centroid)

    return (normal, d)

# %%
def refine_ground_plane(
    point_cloud: np.ndarray,
    num_points: int = 1000,
    height_threshold: float = 0.4,
    distance_threshold: float = 0.2,
    num_iterations: int = 5
) -> "tuple[np.ndarray, np.ndarray, float]":
    """
    Iteratively refine the ground plane estimation using seed points and distance threshold.
    
    Args:
        point_cloud (np.ndarray): Nx6 array [x, y, z, true_label, pred_label, scanline_id].
        num_points (int): Number of lowest Z points used to compute the initial ground seed height (LPR).
        height_threshold (float): Vertical distance threshold from the LPR used to select initial seed points.
        distance_threshold (float): Max allowed point-to-plane distance for a point to be considered ground.
        num_iterations (int): Number of iterations to refine the plane and ground classification.
    
    Returns:
        tuple: 
            - point_cloud (np.ndarray): Nx6 array [x, y, z, true_label, pred_label, scanline_id], input array with ground points labeled.
            - normal (np.ndarray): Normal vector (a, b, c) of the estimated ground plane.
            - d (float): Offset term of the estimated plane equation (ax + by + cz + d = 0).
    """

    # Step 0: Use only XYZ for plane estimation
    xyz = point_cloud[:, :3]

    # Step 1: Get initial seed points based on lowest Z values
    seed_indices = extract_initial_seed_indices(xyz, num_points, height_threshold)

    for _ in range(num_iterations):
        # Step 2: Estimate ground plane using current seeds
        normal, d = estimate_ground_plane(xyz[seed_indices])

        # Step 3: Compute distances from all points to the estimated plane
        distances = np.abs(np.dot(xyz, normal) + d) / np.linalg.norm(normal)

        # Step 4: Classify as ground if within distance threshold
        is_ground = distances < distance_threshold

        # Step 5: Update seeds with newly classified ground points
        seed_indices = np.where(is_ground)[0]

    # Final ground classification using last iteration's result
    point_cloud[seed_indices, 4] = 9 # Set label = 9 for ground

    return (point_cloud, normal, d)

# %% [markdown]
# # Scan Line Run (SLR)

# %%
def group_by_scanline(point_cloud: np.ndarray) -> "list[np.ndarray]":
    """
    Group points by their scanline index in a vectorized way.

    Args:
        point_cloud (np.ndarray): N x 6 array [x, y, z, true_label, pred_label, scanline_id].

    Returns:
        list[np.ndarray]: List of arrays. Each array contains the points (N_i x 6)
                          from one scanline, sorted by scanline_id.
    """
    scan_ids = point_cloud[:, 5].astype(int)
    unique_ids = np.unique(scan_ids)

    return [point_cloud[scan_ids == s_id] for s_id in unique_ids]

# %%
def find_runs(scanline_points: np.ndarray, distance_threshold: float = 0.5) -> "list[np.ndarray]":
    """
    Identify runs within a single scanline based on distance between consecutive points.

    Args:
        scanline_points (np.ndarray): N x 6 array [x, y, z, true_label, pred_label, scanline_id].
        distance_threshold (float): Distance threshold to consider two points part of the same run.

    Returns:
        list[np.ndarray]: List of arrays where each array contains the points of a run.
    """
    num_points = len(scanline_points)
    runs = []
    current_run_indices = [0]  # start with the index of the first point

    for i in range(1, num_points):
        dist = np.linalg.norm(scanline_points[i, :3] - scanline_points[i - 1, :3])
        if dist < distance_threshold:
            current_run_indices.append(i)
        else:
            runs.append(scanline_points[current_run_indices])
            current_run_indices = [i]

    # append the last run
    runs.append(scanline_points[current_run_indices])

    # Check if first and last points are close (circular case)
    circular_dist = np.linalg.norm(scanline_points[0, :3] - scanline_points[-1, :3])
    # Only merge runs if:
    # - the scanline appears to be circular (first and last points are close), and
    # - there is more than one run (otherwise merging doesn't make sense)
    if circular_dist < distance_threshold and len(runs) > 1:
        # Merge last run with the first
        runs[0] = np.vstack((runs[-1], runs[0]))
        runs.pop()

    return runs

# %%
def update_labels(
    runs_current: "list[np.ndarray]",
    runs_above: "list[np.ndarray]",
    label_equivalences: dict,
    merge_threshold: float = 1.0
):
    """
    Update labels of current scanline runs based on proximity to runs from previous scanline using KDTree.

    Args:
        runs_current (list[np.ndarray]): List of N x 6 arrays for current scanline runs.
        runs_above (list[np.ndarray]): List of N x 6 arrays for previous scanline runs.
        label_equivalences (dict): Dictionary of label equivalences.
        merge_threshold (float): Maximum distance to consider connection between runs.
    """
    def resolve_label(label: int) -> int:
        """Find the final label by following the equivalence chain."""
        while label != label_equivalences[label]:
            label = label_equivalences[label]
        return label

    global_label_counter = max(label_equivalences.values()) + 1

    points_above = np.vstack(runs_above)
    tree_above = KDTree(points_above[:, :3])  # use only x, y, z

    for run in runs_current:
        neighbor_labels = set()

        # Check nearest neighbor of each point in current run
        dists, indices = tree_above.query(run[:, :3], k=1)
        for dist, idx in zip(dists[:, 0], indices[:, 0]):
            if dist < merge_threshold:
                neighbor_label = points_above[idx, 4]
                resolved_label = resolve_label(neighbor_label)
                neighbor_labels.add(resolved_label)

        if not neighbor_labels:
            # No close neighbors → assign new label
            while global_label_counter == 9 or global_label_counter in label_equivalences:
                global_label_counter += 1
            run[:, 4] = global_label_counter
            label_equivalences[global_label_counter] = global_label_counter
        else:
            # Inherit the smallest label and unify equivalences
            min_label = min(neighbor_labels)
            run[:, 4] = min_label
            for lbl in neighbor_labels:
                label_equivalences[lbl] = min_label

# %%
def extract_clusters(scanlines: "list[np.ndarray]", label_equivalences: dict) -> np.ndarray:
    """
    Apply resolved labels to all points and return a unified point cloud.

    Args:
        scanlines (list[np.ndarray]): List of N x 6 arrays for each scanline.
        label_equivalences (dict): Dictionary of final label equivalences.

    Returns:
        np.ndarray: N x 6 array with updated labels in column 4.
    """
    non_ground_points = np.vstack(scanlines)

    for idx in range(0, len(non_ground_points)):
        non_ground_points[idx][4] = label_equivalences[non_ground_points[idx][4]]

    return non_ground_points

# %%
def scan_line_run_clustering(
    point_cloud: np.ndarray, 
    distance_threshold: float = 0.5, 
    merge_threshold: float = 1.0
) -> np.ndarray:
    """
    Perform scan line run clustering on non-ground points (predicted_label == 0).

    This function detects connected components (runs) within scanlines, propagates
    and merges labels across scanlines, and assigns final labels to each point.

    Args:
        point_cloud (np.ndarray): N x 6 array [x, y, z, true_label, predicted_label, scanline_index].
        distance_threshold (float): Distance threshold to consider two points part of the same run.
        merge_threshold (float): Maximum distance to consider connection between runs.

    Returns:
        np.ndarray: Point cloud with updated predicted labels (column 4).
    """
    label_counter = 0
    label_equivalences = {}

    # Filter non-ground points (predicted_label == 0)
    non_ground_mask = point_cloud[:, 4] == 0
    non_ground_points = point_cloud[non_ground_mask]
    ground_points = point_cloud[~non_ground_mask]

    # Group points into scanlines
    scanlines = group_by_scanline(non_ground_points)

    # Initialize clustering with the first scanline
    runs_above = find_runs(scanlines[0], distance_threshold)
    for runs in runs_above:
        label_counter += 1
        if label_counter == 9:  # reserve label 9 for ground
            label_counter += 1
        runs[:, 4] = label_counter
        label_equivalences[label_counter] = label_counter

    scanlines[0] = np.vstack(runs_above)
        
    # Propagate labels through remaining scanlines
    for i in range(1, len(scanlines)):
        runs_current = find_runs(scanlines[i], distance_threshold)
        update_labels(runs_current, runs_above, label_equivalences, merge_threshold)

        scanlines[i] = np.vstack(runs_current)
        runs_above = runs_current

    clustered_points = extract_clusters(scanlines, label_equivalences)
    return np.vstack((clustered_points, ground_points))

# %% [markdown]
# # Dataset

# %%
class Dataset:
    def __init__(self, data_path: str, split: str = 'train') -> None:
        """
        Initialize dataset loader.

        Args:
            data_path (str or Path): Base path to the SemanticKITTI dataset.
            split (str): Dataset split to use ('train', 'valid', or 'test').
        """
        self.data_path: Path = Path(data_path)
        self.split: str = split
        self.is_test: bool = split == 'test'

        # Paths to YAML config and data folders
        self.yaml_path: Path = self.data_path / 'semantic-kitti.yaml'
        self.velodynes_path: Path = self.data_path / 'data_odometry_velodyne/dataset/sequences'
        self.labels_path: Path = self.data_path / 'data_odometry_labels/dataset/sequences'

        # Load dataset metadata and label mappings
        with open(self.yaml_path, 'r') as file:
            metadata: dict = yaml.safe_load(file)

        self.sequences: list[int] = metadata['split'][split]
        self.learning_map: dict[int, int] = metadata['learning_map']

        # Convert label map to numpy for fast lookup
        max_label: int = max(self.learning_map.keys())
        self.learning_map_np: np.ndarray = np.zeros((max_label + 1,), dtype=np.uint32)
        for raw_label, mapped_label in self.learning_map.items():
            self.learning_map_np[raw_label] = mapped_label

        # Collect all frame paths for selected sequences
        self.frame_paths: list[tuple[str, str]] = self._collect_frame_paths()

    def _collect_frame_paths(self) -> "list[tuple[str, str]]":
        """Collect all (sequence, frame_id) pairs from the dataset split."""
        frame_list = []
        for seq in self.sequences:
            seq_str = f"{int(seq):02d}"
            seq_velo_path = self.velodynes_path/seq_str/'velodyne'
            velo_files = sorted(seq_velo_path.glob('*.bin'))
            for file in velo_files:
                frame_list.append((seq_str, file.stem))
        return frame_list

    def __len__(self) -> int:
        """Return number of samples in the dataset split."""
        return len(self.frame_paths)

    def _compute_scanline_ids(self, point_cloud: np.ndarray, n_scans: int = 64) -> np.ndarray:
        """
        Approximate scanline indices based on point order.

        Args:
            point_cloud (np.ndarray): Nx3 array of 3D points.
            n_scans (int): Number of LiDAR scanlines (e.g., 64 for HDL-64E).

        Returns:
            np.ndarray: Nx1 array with estimated scanline indices (0 to n_scans - 1).
        """
        total_points = point_cloud.shape[0]
        scanline_ids = np.floor(np.linspace(0, n_scans, total_points, endpoint=False)).astype(int)
        return scanline_ids.reshape(-1, 1)

    def __getitem__(self, idx: int) -> "tuple[np.ndarray, dict[str, np.ndarray]]":
        """
        Load a sample from the dataset.

        Args:
            idx (int): Index of the frame to load.

        Returns:
            tuple:
                - point_cloud_with_label (np.ndarray): Nx6 array [x, y, z, true_label, pred_label, scanline_id].
                - item_dict (dict): Contains 'point_cloud', 'label', and 'mask'.
        """

        seq, frame_id = self.frame_paths[idx]

        # Load point cloud (Nx4), drop reflectance
        velodyne_file_path = self.velodynes_path/seq/'velodyne'/f"{frame_id}.bin"
        with open(velodyne_file_path, 'rb') as file:
            point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]

        # Load and map semantic labels
        if not self.is_test:
            label_file_path = self.labels_path/seq/'labels'/f"{frame_id}.label"
            if label_file_path.exists():
                with open(label_file_path, 'rb') as file:
                    raw_labels = np.fromfile(file, dtype=np.uint32) & 0xFFFF
                labels = self.learning_map_np[raw_labels]
                mask = labels != 0
            else:
                labels = np.zeros(point_cloud.shape[0], dtype=np.uint32)
                mask = np.ones(point_cloud.shape[0], dtype=bool)
        else:
            labels = np.zeros(point_cloud.shape[0], dtype=np.uint32)
            mask = np.ones(point_cloud.shape[0], dtype=bool)

        # Estimate scanline indices
        scanline_ids = self._compute_scanline_ids(point_cloud)

        # Final format: [x, y, z, true_label, predicted_label, scanline_id]
        point_cloud_with_label = np.hstack((
            point_cloud,
            labels.reshape(-1, 1),
            np.zeros((point_cloud.shape[0], 1), dtype=np.float32),
            scanline_ids
        ))

        item_dict = {
            'point_cloud': point_cloud,
            'label': labels,
            'mask': mask
        }

        return point_cloud_with_label, item_dict

# %% [markdown]
# # Generate Plane

# %%
def generate_plane_points(
    point_cloud: np.ndarray, 
    normal: np.ndarray, 
    d: float, 
    size: float = 30, 
    resolution: float = 0.5
) -> np.ndarray:
    """
    Generate a grid of 3D points lying on a specified plane, and return them with label placeholders.
    The plane is defined by the equation: ax + by + cz + d = 0

    Args:
        point_cloud (np.ndarray): Nx6 array [x, y, z, true_label, pred_label, scanline_id].
        normal (np.ndarray): Plane normal vector [a, b, c].
        d (float): Plane offset in the equation ax + by + cz + d = 0.
        size (float): Half-length of the plane square grid to generate (in meters).
        resolution (float): Spacing between points in the grid.

    Returns:
        np.ndarray: Mx6 array of points with [x, y, z, label1, label2, label3],
                    where the last 3 columns are filled with -1 as placeholders.
    """

    # Compute center of the plane as the centroid of the point cloud
    center = point_cloud[:, :3].mean(axis=0)
    a, b, c = normal

    # Create a mesh grid around the center point in the XY plane
    x_vals = np.arange(center[0] - size, center[0] + size, resolution)
    y_vals = np.arange(center[1] - size, center[1] + size, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    
    # Solve for z using the plane equation: ax + by + cz + d = 0 => z = (-d - ax - by)/c
    zz = (-d - a * xx - b * yy) / c

    # Stack into N x 3 array of [x, y, z]
    xyz = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

    # Create label columns filled with -1 as placeholders (e.g., for plane visualization)
    labels = np.full((xyz.shape[0], 3), -1, dtype=np.float32)

    # Concatenate coordinates and labels into a final N x 6 array
    return np.hstack((xyz, labels))

# %% [markdown]
# # Visualizer for Point Clouds

# %%
class PointCloudVisualizer:
    def __init__(self, point_size: float = 1.0):
        """
        Visualizer class for rendering point clouds using Open3D with color-coded semantic labels.

        Args:
            point_size (float): Default size of points in the Open3D viewer.
        """
        self.point_size = point_size
        self.fixed_colors_rgb = self._get_fixed_colors_rgb()

    def _get_fixed_colors_rgb(self) -> "dict[int, list[float]]":
        fixed_colors = {
            -1: [255, 255, 255],  # plane
             0: [0, 0, 0],        # unlabeled
             1: [245, 150, 100],  # car
             2: [245, 230, 100],  # bicycle
             3: [150, 60, 30],    # motorcycle
             4: [180, 30, 80],    # truck
             5: [250, 80, 100],   # other-vehicle
             6: [30, 30, 255],    # person
             7: [200, 40, 255],   # bicyclist
             8: [90, 30, 150],    # motorcyclist
             9: [255, 0, 255],    # road
            10: [255, 150, 255],  # parking
            11: [75, 0, 75],      # sidewalk
            12: [75, 0, 175],     # other-ground
            13: [0, 200, 255],    # building
            14: [50, 120, 255],   # fence
            15: [0, 175, 0],      # vegetation
            16: [0, 60, 135],     # trunk
            17: [80, 240, 150],   # terrain
            18: [150, 240, 255],  # pole
            19: [0, 0, 255],      # traffic-sign
        }
        return {label: [c / 255.0 for c in reversed(rgb)] for label, rgb in fixed_colors.items()}

    def _get_color_map(self, labels: np.ndarray) -> np.ndarray:
        """
        Assigns RGB colors to labels.

        Args:
            labels (np.ndarray): Array of label ids.

        Returns:
            np.ndarray: Nx3 array of RGB colors.
        """
        color_map = np.zeros((labels.shape[0], 3))
        for i, label in enumerate(labels):
            if label in self.fixed_colors_rgb:
                color_map[i] = self.fixed_colors_rgb[label]
            else:
                np.random.seed(label)
                color_map[i] = np.random.rand(3)
        return color_map
    
    def _create_grid(self, size=50, spacing=1.0):
        points = []
        lines = []
        colors = []
    
        for i in range(-size, size + 1):
            # Linhas paralelas ao eixo X
            points.append([i * spacing, -size * spacing, 0])
            points.append([i * spacing, size * spacing, 0])
            lines.append([len(points) - 2, len(points) - 1])
    
            # Linhas paralelas ao eixo Y
            points.append([-size * spacing, i * spacing, 0])
            points.append([size * spacing, i * spacing, 0])
            lines.append([len(points) - 2, len(points) - 1])
    
            # Cores: mais clara a cada 10 unidades
            color = [0.3, 0.3, 0.3] if i % 10 else [0.6, 0.6, 0.6]
            colors.extend([color, color])
    
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector(colors)
        return grid
    
    def _create_plane(self, normal_d_tuple, size=100.0):
        """
        Cria um plano baseado no vetor normal e no valor d.

        Args:
            normal_d_tuple (tuple): Tuple contendo o vetor normal (x, y, z) e o valor d.
            size (float): Tamanho do plano a ser desenhado.
        
        Returns:
            o3d.geometry.TriangleMesh: Mesh do plano.
        """
        normal, d = normal_d_tuple

        # Normalizar o vetor normal
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        
        # Calcular 4 pontos do plano
        p1 = np.array([-size, -size, -(normal[0] * (-size) + normal[1] * (-size) + d) / normal[2]])
        p2 = np.array([size, -size, -(normal[0] * size + normal[1] * (-size) + d) / normal[2]])
        p3 = np.array([size, size, -(normal[0] * size + normal[1] * size + d) / normal[2]])
        p4 = np.array([-size, size, -(normal[0] * (-size) + normal[1] * size + d) / normal[2]])

        # Criar os pontos para o mesh
        points = np.vstack((p1, p2, p3, p4))

        # Criar os triângulos que formam o plano
        triangles = [
            [0, 1, 2],  # Triângulo 1
            [0, 2, 3]   # Triângulo 2
        ]
        
        # Criar a malha triangular
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(points)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Colorir a malha do plano de azul
        plane_mesh.paint_uniform_color([0.45, 0.45, 0.45])
        
        return plane_mesh

    def _create_axis_arrow(self, length=1.0, color=[1, 0, 0], rotation=None):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.03,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2
        )
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(color)
        if rotation is not None:
            arrow.rotate(rotation, center=(0, 0, 0))
        return arrow



    def show(
        self,
        point_cloud: np.ndarray,
        normal_d_tuple: tuple,
        show_true_label: bool = False,
        show_ground: bool = True,
        show_clusters: bool = True,
        show_unlabeled: bool = True,
        show_plane: bool = False,
        point_size = None,
        show_grid: bool = False
    ) -> None:
        """
        Visualize the filtered point cloud using Open3D.

        Args:
            point_cloud (np.ndarray): N x 6 array [x, y, z, true_label, pred_label, scanline_id].
        """

        label_col = 3 if show_true_label else 4
        labels = point_cloud[:, label_col]

        # Apply filter mask
        mask = (
            (show_plane & (labels == -1)) |
            (show_unlabeled & (labels == 0)) |
            (show_ground & (labels == 9)) |
            (show_clusters & (labels >= 1) & (labels != 9))
        )

        xyz = point_cloud[mask, :3]
        visible_labels = labels[mask].astype(int)
        colors = self._get_color_map(visible_labels)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize with point size
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Plane Visualization", width=800, height=600)
        vis.add_geometry(pcd)

        # ===== CRIAÇÃO DOS EIXOS =====
        # X (vermelho): rotaciona -90° ao redor Z
        arrow_x = self._create_axis_arrow(length=1.0, color=[1, 0, 0], rotation=o3d.geometry.get_rotation_matrix_from_xyz([0, -np.pi / 2, 0]))
        # Y (verde): rotaciona +90° ao redor X
        arrow_y = self._create_axis_arrow(length=1.0, color=[0, 1, 0], rotation=o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]))
        # Z (azul): já está na direção Z por padrão
        arrow_z = self._create_axis_arrow(length=1.0, color=[0, 0, 1], rotation=None)

        vis.add_geometry(arrow_x)
        vis.add_geometry(arrow_y)
        vis.add_geometry(arrow_z)

        if show_grid:
            vis.add_geometry(self._create_grid(size=50, spacing=1.0))  # grid grande com espaçamento 1m
        if show_plane:
            vis.add_geometry(self._create_plane(normal_d_tuple))

        opt = vis.get_render_option()
        opt.point_size = point_size or self.point_size
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # estilo AutoCAD / pptk

        # Ajustar a transparência do plano
        opt.mesh_show_back_face = True  # Exibir o lado de trás do plano
        opt.line_width = 10  # Tamanho das linhas de malha

        o3d.visualization.draw([pcd])

        vis.run()
        vis.destroy_window()

# %% [markdown]
# # Run Example

# %% [markdown]
# ## Bin file

# %%
dataset = Dataset('../datasets/semantic-kitti-data')
point_cloud, item = dataset[4]

point_cloud, normal, d = refine_ground_plane(point_cloud, 
                                             num_points=NUM_LOWEST_POINTS, 
                                             height_threshold=SEED_HEIGHT_THRESHOLD, 
                                             distance_threshold=PLANE_DISTANCE_THRESHOLD, 
                                             num_iterations=NUM_ITERATIONS)

point_cloud = scan_line_run_clustering(point_cloud, SLR_RUN_DISTANCE_THRESHOLD, SLR_MERGE_THRESHOLD)

# plane = generate_plane_points(point_cloud, normal, d)
# point_cloud = np.vstack((point_cloud, plane))

# %%
visualizer = PointCloudVisualizer()
visualizer.show(point_cloud, normal_d_tuple=(normal, d))
