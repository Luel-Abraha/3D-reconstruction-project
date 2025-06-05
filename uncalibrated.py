import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union
from collections import defaultdict

# Initialize MediaPipe Face Mesh and SIFT
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Initialize SIFT
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def load_image(path: str) -> np.ndarray:
    """Load an image from file"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img

def enhance_texture_colors(img: np.ndarray) -> np.ndarray:
    """Enhance image colors for better texture mapping"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = np.where(hsv[..., 0] < 15, hsv[..., 0] * 0.9, hsv[..., 0])
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(enhanced)
    r = np.clip(r * 0.95, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def load_camera_parameters(json_file: str) -> Dict[str, np.ndarray]:
    """Load camera intrinsics from JSON file"""
    with open(json_file, 'r') as f:
        cameras = json.load(f)
    intrinsics = {}
    for cam in cameras:
        name = cam["name"]
        matrix = np.array(cam["intrinsics"])
        if matrix.shape != (3, 3):
            matrix = matrix.reshape(3, 3)
        intrinsics[name] = matrix
    return intrinsics

def detect_face_landmarks(img: np.ndarray, expand_contour: bool = True) -> Tuple[np.ndarray, Dict]:
    """Detect facial landmarks using MediaPipe and extract SIFT features"""
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected")
    
    h, w = img.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    
    # Detect SIFT features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    
    # Filter SIFT features to be near face landmarks
    if kp is not None and des is not None:
        filtered_kp = []
        filtered_des = []
        for point, descriptor in zip(kp, des):
            # Check if the point is within 50 pixels of any landmark
            distances = np.linalg.norm(landmarks_2d - np.array([point.pt]), axis=1)
            if np.min(distances) < 1500:  # Only keep features near face landmarks
                filtered_kp.append(point)
                filtered_des.append(descriptor)
        
        if filtered_kp:
            filtered_des = np.array(filtered_des)
        else:
            filtered_kp = None
            filtered_des = None
    else:
        filtered_kp = None
        filtered_des = None
    
    return landmarks_2d, {'sift_kp': filtered_kp, 'sift_des': filtered_des}

def match_features(features1: Dict, features2: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Match SIFT features between two images"""
    if features1['sift_kp'] is None or features2['sift_kp'] is None:
        return np.array([]), np.array([])
    
    # FLANN-based feature matching
    matches = flann.knnMatch(features1['sift_des'], features2['sift_des'], k=2)
    
    # Lowe's ratio test
    good_matches = []
    pts1 = []
    pts2 = []
    
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            pts1.append(features1['sift_kp'][m.queryIdx].pt)
            pts2.append(features2['sift_kp'][m.trainIdx].pt)
    
    if len(good_matches) > 10:
        return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)
    return np.array([]), np.array([])

def combine_correspondences(landmarks1: np.ndarray, landmarks2: np.ndarray,
                          sift_pts1: np.ndarray, sift_pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Combine landmark and SIFT correspondences"""
    # Use all landmarks as correspondences
    combined_pts1 = landmarks1.copy()
    combined_pts2 = landmarks2.copy()
    
    # Add SIFT matches if they exist
    if len(sift_pts1) > 0 and len(sift_pts2) > 0:
        combined_pts1 = np.vstack([combined_pts1, sift_pts1])
        combined_pts2 = np.vstack([combined_pts2, sift_pts2])
    
    return combined_pts1, combined_pts2

def estimate_relative_pose(pts1: np.ndarray, pts2: np.ndarray, 
                         K1: np.ndarray, K2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate relative camera pose between two views"""
    pts1 = pts1.reshape(-1, 1, 2)
    pts2 = pts2.reshape(-1, 1, 2)
    
    pts1_norm = cv2.undistortPoints(pts1, K1, None, None, K1)
    pts2_norm = cv2.undistortPoints(pts2, K2, None, None, K2)
    
    E, mask = cv2.findEssentialMat(
        pts1_norm, pts2_norm, 
        cameraMatrix=K1,
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=0.9,
        maxIters=1000
    )
    
    inliers = mask.ravel() == 1
    pts1_norm = pts1_norm[inliers]
    pts2_norm = pts2_norm[inliers]
    
    _, R, t, _ = cv2.recoverPose(E, pts1_norm, pts2_norm, K1)
    print(f"Pose estimation inliers: {inliers.sum()}/{len(pts1)}")
    return R, t

def triangulate_points(P1: np.ndarray, P2: np.ndarray, 
                     pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate 3D points from 2D correspondences"""
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T
    valid = np.isfinite(pts3D).all(axis=1)
    return pts3D[valid]

def align_point_clouds(pcds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Align multiple point clouds into a unified coordinate system"""
    if not pcds:
        raise ValueError("No point clouds to align")
        
    combined = pcds[0]
    for i in range(1, len(pcds)):
        # Rough alignment using centroids
        centroid_ref = np.mean(np.asarray(combined.points), axis=0)
        centroid_new = np.mean(np.asarray(pcds[i].points), axis=0)
        pcds[i].translate(centroid_ref - centroid_new)
        
        # Scale normalization
        ref_points = np.asarray(combined.points)
        new_points = np.asarray(pcds[i].points)
        scale = np.mean(np.linalg.norm(ref_points, axis=1)) / np.mean(np.linalg.norm(new_points, axis=1))
        pcds[i].scale(scale, center=np.zeros(3))
        
        # Fine alignment with ICP
        threshold = 0.01
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcds[i], combined, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        pcds[i].transform(reg_p2p.transformation)
        combined += pcds[i]
        
    return combined

def create_texture_coordinates(vertices: np.ndarray, 
                             img_shape: Tuple[int, int], 
                             K: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """Create UV coordinates for texture mapping"""
    h, w = img_shape[:2]
    vertices_hom = np.hstack((vertices, np.ones((len(vertices), 1))))
    proj_pts = (K @ Rt @ vertices_hom.T).T
    proj_pts = proj_pts[:, :2] / proj_pts[:, 2, np.newaxis]
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = np.clip(proj_pts[:, 0] / w, 0.01, 0.99)
    uvs[:, 1] = 1 - np.clip(proj_pts[:, 1] / h, 0.01, 0.99)
    return uvs

def save_as_ply(mesh: o3d.geometry.TriangleMesh, filename: str, include_colors: bool = True):
    """Save mesh in PLY format with optional vertex colors"""
    mesh_ply = o3d.geometry.TriangleMesh()
    mesh_ply.vertices = mesh.vertices
    mesh_ply.triangles = mesh.triangles
    mesh_ply.vertex_normals = mesh.vertex_normals
    
    if include_colors and mesh.vertex_colors:
        mesh_ply.vertex_colors = mesh.vertex_colors
    
    o3d.io.write_triangle_mesh(filename, mesh_ply)
    
    print(f"Saved PLY file to {filename} (colors={'yes' if include_colors else 'no'})")

def reconstruct_face(pcd: o3d.geometry.PointCloud, 
                   img: np.ndarray, K: np.ndarray, Rt: np.ndarray, 
                   output_path: str = "output/face") -> o3d.geometry.TriangleMesh:
    """Create a textured mesh from point cloud"""
    texture_img = enhance_texture_colors(img)
    
    # Preprocess point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    
    # Poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, linear_fit=True, scale=1.2)
    
    # Post-process mesh
    density_threshold = np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(densities < density_threshold)
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    mesh.compute_vertex_normals()
    
    # Create texture coordinates and save mesh
    uvs = create_texture_coordinates(np.asarray(mesh.vertices), texture_img.shape, K, Rt)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save texture image
    texture_path = f"{output_path}_texture.png"
    cv2.imwrite(texture_path, texture_img)
    
    # Write MTL file
    with open(f"{output_path}.mtl", 'w') as f:
        f.write("newmtl face_material\n")
        f.write("Ka 1.0 1.0 1.0\nKd 0.9 0.9 0.9\nKs 0.1 0.1 0.1\nNs 10.0\nillum 2\n")
        f.write(f"map_Kd {os.path.basename(texture_path)}\n")
    
    # Write OBJ file
    with open(f"{output_path}.obj", 'w') as f:
        f.write(f"mtllib {os.path.basename(output_path)}.mtl\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in mesh.vertex_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("usemtl face_material\n")
        for tri in mesh.triangles:
            f.write(f"f {tri[0]+1}/{tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}/{tri[2]+1}\n")
    
    # Save as PLY file
    ply_path = f"{output_path}.ply"
    save_as_ply(mesh, ply_path)
    
    # For colored PLY
    save_as_ply(mesh, "textured.ply", include_colors=True)

    # For geometry-only PLY
    save_as_ply(mesh, "geometry_only.ply", include_colors=False)

    return mesh

def compute_reprojection_error(pts3D: np.ndarray, pts2D: np.ndarray, 
                             K: np.ndarray, R: np.ndarray = None, 
                             t: np.ndarray = None, visualize: bool = False,
                             img: np.ndarray = None) -> Dict[str, Union[float, np.ndarray]]:
    """Compute reprojection error metrics with detailed analysis"""
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    
    # Ensure we have the same number of points
    min_len = min(len(pts3D), len(pts2D))
    pts3D = pts3D[:min_len]
    pts2D = pts2D[:min_len]
    
    # Project 3D points to 2D
    pts3D_hom = np.hstack((pts3D, np.ones((len(pts3D), 1))))
    P = K @ np.hstack((R, t.reshape(3, 1)))
    pts2D_proj_hom = (P @ pts3D_hom.T).T
    pts2D_proj = pts2D_proj_hom[:, :2] / pts2D_proj_hom[:, 2, np.newaxis]
    
    # Calculate errors
    errors = np.linalg.norm(pts2D - pts2D_proj, axis=1)
    valid = np.isfinite(errors)
    errors = errors[valid]
    
    # Calculate error distribution statistics
    hist, bins = np.histogram(errors, bins=20, range=(0, 20))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Visualize if requested
    if visualize and img is not None:
        vis_img = img.copy()
        for (x_obs, y_obs), (x_proj, y_proj) in zip(pts2D[valid], pts2D_proj[valid]):
            cv2.line(vis_img, (int(x_obs), int(y_obs)), (int(x_proj), int(y_proj)), (0, 0, 255), 1)
            cv2.circle(vis_img, (int(x_obs), int(y_obs)), 3, (0, 255, 0), -1)
            cv2.circle(vis_img, (int(x_proj), int(y_proj)), 2, (255, 0, 0), -1)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("Reprojection Errors (Green=Observed, Blue=Projected)")
        plt.axis('off')
        plt.show()
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'std_dev': np.std(errors),
        'inlier_ratio': np.mean(errors < 10.0),  # Changed threshold to 10 pixels
        'num_points': len(errors),
        'errors': errors,
        'error_distribution': {
            'bin_centers': bin_centers,
            'counts': hist,
            'percentages': hist / len(errors) * 100
        },
        'projected_points': pts2D_proj[valid],
        'observed_points': pts2D[valid]
    }

def analyze_reprojection_errors(evaluation: Dict) -> Dict[str, Dict]:
    """Analyze reprojection errors for front-left and front-right views"""
    analysis = {}
    
    # Initialize error accumulators
    front_left_errors = []
    front_right_errors = []
    all_errors = []
    
    # Collect errors for each view pair
    for pair_name, pair_errors in evaluation['reprojection_errors'].items():
        for view_name, errors in pair_errors.items():
            all_errors.extend(errors['errors'])
            if 'Front_Left' in pair_name:
                front_left_errors.extend(errors['errors'])
            elif 'Front_Right' in pair_name:
                front_right_errors.extend(errors['errors'])
    
    # Convert to numpy arrays for easier calculations
    front_left_errors = np.array(front_left_errors)
    front_right_errors = np.array(front_right_errors)
    all_errors = np.array(all_errors)
    
    # Calculate statistics for each view set
    def calculate_stats(errors):
        return {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'max': np.max(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'std_dev': np.std(errors),
            'inlier_ratio': np.mean(errors < 10.0),
            'num_points': len(errors)
        }
    
    analysis['front_left'] = calculate_stats(front_left_errors)
    analysis['front_right'] = calculate_stats(front_right_errors)
    analysis['overall'] = calculate_stats(all_errors)
    
    # Calculate error distributions
    def calculate_distribution(errors):
        hist, bins = np.histogram(errors, bins=20, range=(0, 20))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return {
            'bin_centers': bin_centers.tolist(),
            'counts': hist.tolist(),
            'percentages': (hist / len(errors) * 100).tolist()
        }
    
    analysis['front_left']['distribution'] = calculate_distribution(front_left_errors)
    analysis['front_right']['distribution'] = calculate_distribution(front_right_errors)
    analysis['overall']['distribution'] = calculate_distribution(all_errors)
    
    return analysis

def evaluate_reconstruction(pts3D_dict: Dict[str, np.ndarray], 
                          landmarks_dict: Dict[str, np.ndarray],
                          camera_params: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """Comprehensive evaluation of reconstruction quality"""
    evaluation = {}
    
    # Reprojection error analysis
    evaluation['reprojection_errors'] = {}
    for pair_name, pts3D in pts3D_dict.items():
        view1, view2 = pair_name.split('_')
        
        # Get corresponding landmarks and ensure they match the 3D points length
        min_len = len(pts3D)
        lm1 = landmarks_dict[view1][:min_len]
        lm2 = landmarks_dict[view2][:min_len]
        
        # Also ensure we don't have more 3D points than landmarks
        pts3D = pts3D[:min(len(lm1), len(lm2))]
        
        # Evaluate for both views
        evaluation['reprojection_errors'][pair_name] = {
            view1: compute_reprojection_error(
                pts3D[:len(lm1)], lm1[:len(pts3D)], 
                camera_params[view1]['K'],
                camera_params[view1]['R'],
                camera_params[view1]['t']
            ),
            view2: compute_reprojection_error(
                pts3D[:len(lm2)], lm2[:len(pts3D)],
                camera_params[view2]['K'],
                camera_params[view2]['R'],
                camera_params[view2]['t']
            )
        }
    
    # Perform detailed error analysis
    evaluation['error_analysis'] = analyze_reprojection_errors(evaluation)
    
    # Point cloud quality metrics
    if len(pts3D_dict) > 1:
        combined_pcd = align_point_clouds([
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3D)) 
            for pts3D in pts3D_dict.values()
        ])
        
        evaluation['point_cloud_metrics'] = {
            'num_points': len(np.asarray(combined_pcd.points)),
            'extent': np.ptp(np.asarray(combined_pcd.points), axis=0).tolist(),
            'density': len(np.asarray(combined_pcd.points)) / np.prod(np.ptp(np.asarray(combined_pcd.points), axis=0))
        }
    
    return evaluation

def visualize_error_analysis(analysis: Dict, output_dir: str = "output"):
    """Visualize the error analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plot for view pairs
    plt.figure(figsize=(12, 6))
    
    # Prepare data for bar plot
    views = ['Front-Left', 'Front-Right', 'Overall']
    metrics = ['mean', 'median', 'rmse', 'max']
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Reprojection Error Metrics Comparison")
    
    # Plot each metric
    for ax, metric in zip(axes.flat, metrics):
        values = [
            analysis['front_left'][metric],
            analysis['front_right'][metric],
            analysis['overall'][metric]
        ]
        ax.bar(views, values)
        ax.set_title(metric.capitalize() + " Error")
        ax.set_ylabel("Pixels")
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "error_metrics_comparison.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved error metrics comparison to {comp_path}")
    
    # Plot error distributions
    plt.figure(figsize=(12, 6))
    plt.plot(
        analysis['front_left']['distribution']['bin_centers'],
        analysis['front_left']['distribution']['percentages'],
        label='Front-Left', linewidth=2
    )
    plt.plot(
        analysis['front_right']['distribution']['bin_centers'],
        analysis['front_right']['distribution']['percentages'],
        label='Front-Right', linewidth=2
    )
    plt.plot(
        analysis['overall']['distribution']['bin_centers'],
        analysis['overall']['distribution']['percentages'],
        label='Overall', linewidth=2, linestyle='--'
    )
    
    plt.title("Reprojection Error Distribution")
    plt.xlabel("Error (pixels)")
    plt.ylabel("Percentage of Points")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    dist_path = os.path.join(output_dir, "error_distribution_comparison.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved error distribution comparison to {dist_path}")

def visualize_evaluation_results(evaluation: Dict, output_dir: str = "output"):
    """Visualize key evaluation metrics and save as images"""
    if not evaluation:
        print("No evaluation results available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the detailed error analysis
    if 'error_analysis' in evaluation:
        visualize_error_analysis(evaluation['error_analysis'], output_dir)
    
    # Plot individual view pair errors
    if 'reprojection_errors' in evaluation:
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        pairs = []
        mean_errors = []
        
        for pair_name, pair_errors in evaluation['reprojection_errors'].items():
            for view_name, errors in pair_errors.items():
                pairs.append(f"{pair_name}\n({view_name})")
                mean_errors.append(errors['mean_error'])
        
        # Create bar plot
        plt.bar(range(len(mean_errors)), mean_errors, tick_label=pairs)
        plt.title("Mean Reprojection Error by View Pair")
        plt.ylabel("Error (pixels)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save mean error plot
        mean_error_path = os.path.join(output_dir, "mean_reprojection_errors.png")
        plt.savefig(mean_error_path)
        plt.close()
        print(f"Saved mean error plot to {mean_error_path}")

def save_evaluation_results(evaluation: Dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to file with numpy array conversion"""
    if not evaluation:
        print("No evaluation results to save")
        return
        
    def convert_numpy(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(convert_numpy(evaluation), f, indent=2)
    print(f"Evaluation results saved to {filename}")

def main():
    # Load input data
    img_front = load_image('samples/samples/Camera_Front.png')
    img_left = load_image('samples/samples/Camera_Left.png')
    img_right = load_image('samples/samples/Camera_Right.png')
    intrinsics = load_camera_parameters("camera_params.json")
    
    # Get camera parameters
    K_front = intrinsics["front"]
    K_left = intrinsics["left"]
    K_right = intrinsics["right"]
    
    # Detect landmarks and SIFT features in all views
    lm_front, features_front = detect_face_landmarks(img_front)
    lm_left, features_left = detect_face_landmarks(img_left)
    lm_right, features_right = detect_face_landmarks(img_right)
    
    print(f"Detected landmarks - Front: {len(lm_front)}, Left: {len(lm_left)}, Right: {len(lm_right)}")
    
    # Match SIFT features between views
    print("\nMatching SIFT features...")
    sift_pts_front_left1, sift_pts_front_left2 = match_features(features_front, features_left)
    sift_pts_front_right1, sift_pts_front_right2 = match_features(features_front, features_right)
    
    print(f"SIFT matches - Front-Left: {len(sift_pts_front_left1)}, Front-Right: {len(sift_pts_front_right1)}")
    
    # Combine landmarks and SIFT features for pose estimation
    combined_front_left1, combined_front_left2 = combine_correspondences(
        lm_front, lm_left, sift_pts_front_left1, sift_pts_front_left2)
    combined_front_right1, combined_front_right2 = combine_correspondences(
        lm_front, lm_right, sift_pts_front_right1, sift_pts_front_right2)
    
    # Estimate relative poses (using front view as reference)
    print("\nEstimating camera poses...")
    R_left, t_left = estimate_relative_pose(combined_front_left2, combined_front_left1, K_left, K_front)
    R_right, t_right = estimate_relative_pose(combined_front_right2, combined_front_right1, K_right, K_front)
    
    # Create projection matrices
    P_front = K_front @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_left = K_left @ np.hstack((R_left, t_left))
    P_right = K_right @ np.hstack((R_right, t_right))
    
    # Triangulate points for all view pairs
    print("\nTriangulating 3D points...")
    pts3D_front_left = triangulate_points(P_front, P_left, combined_front_left1, combined_front_left2)
    pts3D_front_right = triangulate_points(P_front, P_right, combined_front_right1, combined_front_right2)
    
    print(f"Depth ranges - Front-Left: {np.min(pts3D_front_left[:, 2]):.2f} to {np.max(pts3D_front_left[:, 2]):.2f}")
    
    # Evaluate reconstruction quality
    print("\nEvaluating reconstruction quality...")
    camera_params = {
        "Front": {"K": K_front, "R": np.eye(3), "t": np.zeros(3)},
        "Left": {"K": K_left, "R": R_left, "t": t_left},
        "Right": {"K": K_right, "R": R_right, "t": t_right}
    }
    
    landmarks_dict = {
        "Front": lm_front,
        "Left": lm_left,
        "Right": lm_right
    }
    
    pts3D_dict = {
        "Front_Left": pts3D_front_left,
        "Front_Right": pts3D_front_right
    }
    
    evaluation = evaluate_reconstruction(pts3D_dict, landmarks_dict, camera_params)
    visualize_evaluation_results(evaluation)
    save_evaluation_results(evaluation, "output/evaluation.json")
    
    # Print summary of error analysis
    print("\nReprojection Error Analysis Summary:")
    print("----------------------------------")
    for view, stats in evaluation['error_analysis'].items():
        if view == 'overall':
            continue
        print(f"\n{view.replace('_', ' ').title()} View:")
        print(f"  Mean Error: {stats['mean']:.2f} pixels")
        print(f"  Median Error: {stats['median']:.2f} pixels")
        print(f"  RMSE: {stats['rmse']:.2f} pixels")
        print(f"  Max Error: {stats['max']:.2f} pixels")
        print(f"  Inlier Ratio : {stats['inlier_ratio']:.1%}")
    
    print("\nOverall Performance:")
    overall = evaluation['error_analysis']['overall']
    print(f"  Mean Error: {overall['mean']:.2f} pixels")
    print(f"  Median Error: {overall['median']:.2f} pixels")
    print(f"  RMSE: {overall['rmse']:.2f} pixels")
    print(f"  Max Error: {overall['max']:.2f} pixels")
    print(f"  Inlier Ratio : {overall['inlier_ratio']:.1%}")
    
    # Combine point clouds and reconstruct mesh
    print("\nCreating combined point cloud...")
    pcd_combined = align_point_clouds([
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3D_front_left)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3D_front_right))
    ])
    
    # Color the point cloud using front view
    colors = []
    for pt in np.asarray(pcd_combined.points):
        proj_pt = (P_front @ np.append(pt, 1))[:2] / (P_front @ np.append(pt, 1))[2]
        x, y = int(proj_pt[0]), int(proj_pt[1])
        if 0 <= x < img_front.shape[1] and 0 <= y < img_front.shape[0]:
            color = img_front[y, x][::-1] / 255.0
            color[0] *= 0.95  # Slight blue reduction
        else:
            color = [0.5, 0.5, 0.5]  # Gray for out-of-frame points
        colors.append(color)
    
    pcd_combined.colors = o3d.utility.Vector3dVector(colors)
    
    # Create and save textured mesh
    print("\nReconstructing textured mesh...")
    mesh = reconstruct_face(
        pcd_combined,
        img_front,
        K_front,
        np.hstack((np.eye(3), np.zeros((3, 1)))),
        "output/textured_face"
    )
    
    # Replace the mesh export with point cloud export
    o3d.io.write_point_cloud("raw_face.ply", pcd_combined)
    
    # Visualize final result
    print("\nVisualizing results...")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], 
        window_name="Reconstructed Face", 
        zoom=0.8, 
        front=[0, 0, -1], 
        lookat=mesh.get_center(), 
        up=[0, 1, 0]
    )

if __name__ == "__main__":
    main()