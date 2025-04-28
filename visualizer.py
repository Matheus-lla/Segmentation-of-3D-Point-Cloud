import os
import sys
import json
import shutil
import numpy as np
import open3d as o3d

class PointCloudVisualizer:
    def __init__(self, point_size: float = 1.0, grid_size=50, grid_spacing=1.0, grid_line_width=10):
        """
        Visualizer class for rendering point clouds using Open3D with color-coded semantic labels.

        Args:
            point_size (float): Default size of points in the Open3D viewer.
        """
        self.point_size = point_size
        self.fixed_colors_rgb = self._get_fixed_colors_rgb()
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.grid_line_width = grid_line_width

    def set_point_sieze(self, point_size):
        self.point_size = point_size
        
    def set_grid_size(self, grid_size):
        self.grid_size = grid_size

    def set_grid_spacing(self, grid_spacing):
        self.grid_spacing = grid_spacing

    def set_grid_line_width(self, grid_line_width):
        self.grid_line_width = grid_line_width

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
    
    def _create_grid(self, grid_size=50, grid_spacing=1.0):
        points = []
        lines = []
        colors = []
    
        for i in range(-grid_size, grid_size+ 1):
            # Linhas paralelas ao eixo X
            points.append([i * grid_spacing, -grid_size* grid_spacing, 0])
            points.append([i * grid_spacing, grid_size* grid_spacing, 0])
            lines.append([len(points) - 2, len(points) - 1])
    
            # Linhas paralelas ao eixo Y
            points.append([-grid_size* grid_spacing, i * grid_spacing, 0])
            points.append([grid_size* grid_spacing, i * grid_spacing, 0])
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
        normal_d_tuple: tuple | None = None,
        show_true_label: bool = False,
        show_ground: bool = True,
        show_clusters: bool = True,
        show_unlabeled: bool = True,
        show_plane: bool = False,
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

        # Visualize with point size
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Plane Visualization", width=800, height=600)

        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # estilo AutoCAD / pptk
        opt.show_coordinate_frame = True
        opt.mesh_show_back_face = True  # Exibir o lado de trás do plano

        # extrair para uma funcao
        # X (vermelho): rotaciona -90° ao redor Z
        arrow_x = self._create_axis_arrow(length=1.0, color=[1, 0, 0], rotation=o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]))
        # Y (verde): rotaciona +90° ao redor X
        arrow_y = self._create_axis_arrow(length=1.0, color=[0, 1, 0], rotation=o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]))
        # Z (azul): já está na direção Z por padrão
        arrow_z = self._create_axis_arrow(length=1.0, color=[0, 0, 1], rotation=None)
        vis.add_geometry(arrow_x)
        vis.add_geometry(arrow_y)
        vis.add_geometry(arrow_z)


        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)

        if show_grid:
            vis.add_geometry(self._create_grid(self.grid_size, self.grid_spacing))  # grid grande com espaçamento 1m
            opt.line_width = self.grid_line_width
        if show_plane and normal_d_tuple != None:
            vis.add_geometry(self._create_plane(normal_d_tuple))


        vis.run()
        vis.destroy_window()


import sys
import os
import numpy as np
import json
import shutil

def main():
    if len(sys.argv) < 2:
        print("Uso: python visualizer.py <caminho_para_temp_dir>")
        sys.exit(1)

    temp_dir = sys.argv[1]

    # Carrega arquivos
    point_cloud = np.load(os.path.join(temp_dir, 'point_cloud.npy'))

    normal_d_tuple = None
    normal_path = os.path.join(temp_dir, 'normal_d.npy')
    if os.path.exists(normal_path):
        normal_d = np.load(normal_path)
        normal_d_tuple = (normal_d[:3], normal_d[3])

    with open(os.path.join(temp_dir, 'visualizer_config.json'), 'r') as f:
        config = json.load(f)

    visualizer = PointCloudVisualizer(
        point_size=config.get('point_size', 1.0),
        grid_size=config.get('grid_size', 50),
        grid_spacing=config.get('grid_spacing', 1.0),
        grid_line_width=config.get('grid_line_width', 10),
    )

    visualizer.show(
        point_cloud,
        normal_d_tuple=normal_d_tuple,
        show_plane=config.get('show_plane', False),
        show_grid=config.get('show_grid', False),
        show_ground=config.get('show_ground', True),
        show_clusters=config.get('show_clusters', True),
        show_unlabeled=config.get('show_unlabeled', True),
        show_true_label=config.get('show_true_label', False)
    )

    # Limpar diretório temporário
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
