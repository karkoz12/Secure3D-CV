import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# --- 1. Helper Functions for 3D Data Simulation ---

def generate_benign_scene(n_points: int = 1000) -> np.ndarray:
    """
    Simulates a benign (safe) 3D scene as a point cloud.
    
    For simplicity, simulates a slightly noisy plane (x, y) with z=0.
    In a real scenario, this would be a complex 3D model loading function.
    
    Args:
        n_points: Number of points in the cloud.
        
    Returns:
        A NumPy array of shape (n_points, 3) representing the point cloud.
    """
    # Generate points on a 10x10 plane
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.zeros(n_points)
    
    # Add small Gaussian noise to simulate sensor imperfections
    noise = np.random.normal(0, 0.05, (n_points, 3))
    
    point_cloud = np.vstack([x, y, z]).T + noise
    return point_cloud

def simulate_attack(point_cloud: np.ndarray, attack_type: str = 'noise', severity: float = 0.1) -> np.ndarray:
    """
    Simulates a cyberattack on the point cloud (Section 2.2).
    
    Args:
        point_cloud: The original point cloud.
        attack_type: 'noise' (perturbation) or 'tampering' (injection).
        severity: Controls the intensity of the attack.
        
    Returns:
        The attacked point cloud.
    """
    if attack_type == 'noise':
        # Attack Type 1: Perturbation (Depth map perturbation [4])
        # Adds significant noise to a small percentage of points
        n_points = point_cloud.shape[0]
        n_attacked = int(n_points * severity)
        
        # Select random points to perturb
        indices = np.random.choice(n_points, n_attacked, replace=False)
        
        # Add large, non-Gaussian noise (simulating a malicious manipulation)
        perturbation = np.random.uniform(-1, 1, (n_attacked, 3)) * 0.5
        point_cloud[indices] += perturbation
        
    elif attack_type == 'tampering':
        # Attack Type 2: Injection (Point cloud tampering [5])
        # Injects a small, dense cluster of points far from the scene
        n_injected = int(point_cloud.shape[0] * severity)
        
        # Create a small cluster of points far away (e.g., at [100, 100, 100])
        tamper_cluster = np.random.normal(loc=[100, 100, 100], scale=0.1, size=(n_injected, 3))
        
        point_cloud = np.vstack([point_cloud, tamper_cluster])
        
    return point_cloud

def extract_features(point_cloud: np.ndarray) -> np.ndarray:
    """
    Implements the feature extraction function Phi(S) (Section 2.3).
    
    For simplicity, calculates basic statistical features.
    In a real scenario, this would include geometric moments, curvature, etc.
    
    Args:
        point_cloud: The input point cloud.
        
    Returns:
        A feature vector x (1D array).
    """
    # Feature 1: Number of points (useful for 'tampering' detection)
    n_points = point_cloud.shape[0]
    
    # Feature 2: Standard deviation of Z-coordinates (simulates flatness/noise)
    std_z = np.std(point_cloud[:, 2])
    
    # Feature 3: Bounding box volume (useful for 'injection' detection)
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    volume = np.prod(max_coords - min_coords)
    
    # Feature 4: Mean distance from origin (simple measure of scene location)
    mean_dist = np.mean(np.linalg.norm(point_cloud, axis=1))
    
    # Feature 5: Mean curvature proxy (e.g., mean of the standard deviation of all coordinates)
    mean_std = np.mean(np.std(point_cloud, axis=0))
    
    # Return the feature vector (x)
    return np.array([n_points, std_z, volume, mean_dist, mean_std])

# --- 2. Main Dataset Generation Function ---

def generate_dataset(n_samples: int = 2000, attack_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generates the full synthetic dataset for Secure3D-CV.
    
    Args:
        n_samples: Total number of scenes to generate.
        attack_ratio: Proportion of samples that will be attacked.
        
    Returns:
        Tuple: (X: Feature Matrix, y: Ground Truth Labels, Scenes: List of point clouds)
    """
    n_attacked = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attacked
    
    X = []  # Feature matrix
    y = []  # Labels (0=Benign, 1=Attacked)
    Scenes = [] # List to store the actual point clouds
    
    # 1. Generate Benign Samples
    for _ in range(n_benign):
        scene = generate_benign_scene()
        features = extract_features(scene)
        
        X.append(features)
        y.append(0)
        Scenes.append(scene)
        
    # 2. Generate Attacked Samples
    attack_types = ['noise', 'tampering']
    
    for i in range(n_attacked):
        # Start with a benign scene
        scene = generate_benign_scene()
        
        # Apply a random attack type
        attack_type = attack_types[i % len(attack_types)]
        attacked_scene = simulate_attack(scene, attack_type=attack_type, severity=0.05)
        
        features = extract_features(attacked_scene)
        
        X.append(features)
        y.append(1)
        Scenes.append(attacked_scene)
        
    # Convert to NumPy arrays and shuffle
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the dataset
    p = np.random.permutation(n_samples)
    X = X[p]
    y = y[p]
    Scenes = [Scenes[i] for i in p]
    
    print(f"Dataset generated: {n_samples} samples ({n_benign} benign, {n_attacked} attacked).")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, Scenes

if __name__ == '__main__':
    # Example usage: Generate a dataset of 1000 samples with 20% attacks
    X_features, y_labels, scenes = generate_dataset(n_samples=1000, attack_ratio=0.2)
    
    # Display the first few feature vectors and labels
    print("\nFirst 5 feature vectors (X):")
    print(X_features[:5])
    print("\nFirst 5 labels (y):")
    print(y_labels[:5])
    
    # Example: Check the features of a benign vs. an attacked scene
    # Note: The features will show clear differences due to the simple simulation
    
    # Find a benign sample (label 0)
    benign_index = np.where(y_labels == 0)[0][0]
    print(f"\nFeatures of a Benign Scene (Index {benign_index}):")
    print(X_features[benign_index])
    
    # Find an attacked sample (label 1)
    attacked_index = np.where(y_labels == 1)[0][0]
    print(f"\nFeatures of an Attacked Scene (Index {attacked_index}):")
    print(X_features[attacked_index])
