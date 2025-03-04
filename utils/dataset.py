import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from tqdm import tqdm

class PointCloudDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.file_list = [os.path.join(self.folder, f) for f in sorted(os.listdir(self.folder)) if f.endswith('.pcd')]
        self.flat_kdtrees = []

        for filename in tqdm(self.file_list):
            point_cloud = self.load_pcd(filename)
            kd_tree = self.build_kdtree(point_cloud)
            self.flat_kdtrees.append(self.flatten_kdtree(kd_tree))

        self.flat_kdtrees = np.array(self.flat_kdtrees)
        self.pca = PCA(n_components=100)
        self.pca.fit(self.flat_kdtrees)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        transformed = self.pca.transform(self.flat_kdtrees[None, idx])
        return torch.from_numpy(transformed).float()

    def load_pcd(self, file_path):
        points = np.loadtxt(file_path, skiprows=10)
        return points

    def build_kdtree(self, points, depth=0):
        if len(points) <= 0:
            return None
        axis = depth % points.shape[1]
        sorted_points = points[points[:, axis].argsort()]
        median_index = len(points) // 2

        return {
            'point': sorted_points[median_index],
            'left': self.build_kdtree(sorted_points[:median_index], depth + 1),
            'right': self.build_kdtree(sorted_points[median_index + 1:], depth + 1)
        }

    def flatten_kdtree(self, kd_tree):
        flat_tree = []
        self._inorder_traversal(kd_tree, flat_tree)
        return np.array(flat_tree).flatten()

    def _inorder_traversal(self, node, flat_tree):
        if node is None:
            return
        self._inorder_traversal(node['left'], flat_tree)
        flat_tree.append(node['point'])
        self._inorder_traversal(node['right'], flat_tree)
