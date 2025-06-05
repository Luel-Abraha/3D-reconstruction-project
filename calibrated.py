import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
from collections import defaultdict

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1
)

# Initialize SIFT
sift = cv2.SIFT_create()

def load_image(path):
    """Load image while preserving color channels"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img

def enhance_texture_colors(img):
    """Enhanced color processing with better red balance"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Increase saturation with red channel correction
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = np.where(hsv[..., 0] < 15, hsv[..., 0] * 0.9, hsv[..., 0])  # Reduce red hue
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)  # Boost saturation
    
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Additional color balance in BGR space
    b, g, r = cv2.split(enhanced)
    r = np.clip(r * 0.95, 0, 255).astype(np.uint8)  # Reduce red channel
    enhanced = cv2.merge((b, g, r))
    
    return enhanced

def load_camera_parameters(json_file):
    """Load camera calibration data"""
    with open(json_file, 'r') as f:
        cameras = json.load(f)

    intrinsics = {}
    extrinsics = {}
    projections = {}

    for cam in cameras:
        name = cam["name"]
        K = np.array(cam["intrinsics"])
        R = np.array(cam["rotation"])
        t = np.array(cam["translation"]).reshape((3, 1))
        Rt = np.hstack((R, t))
        P = K @ Rt

        intrinsics[name] = K
        extrinsics[name] = Rt
        projections[name] = P
    
    return intrinsics, extrinsics, projections
    
def detect_face_landmarks(img):
    """Detect facial landmarks with MediaPipe with duplicate removal"""
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face landmarks detected")
    
    h, w = img.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Convert to array and remove duplicates
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    
    # Remove duplicate points (eyes often have overlapping landmarks)
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    return points[sorted(unique_indices)]

def detect_sift_features(img, mask=None):
    """Detect and describe SIFT features in an image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, mask)
    return kp, desc

def match_features(desc1, desc2, ratio_thresh=0.5):
    """Match SIFT features using FLANN matcher with ratio test"""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return good_matches

def combine_landmarks_and_sift(landmarks1, landmarks2, kp1, kp2, matches, img_shape):
    """Combine face landmarks with SIFT matches for robust correspondences"""
    # Convert keypoints to coordinates
    sift_pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    sift_pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    # Combine with landmarks
    combined_pts1 = np.vstack((landmarks1, sift_pts1))
    combined_pts2 = np.vstack((landmarks2, sift_pts2))
    
    return combined_pts1, combined_pts2

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate 3D points from 2D correspondences"""
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T
    
    # Validate points
    valid_mask = np.isfinite(pts3D).all(axis=1)
    if not valid_mask.all():
        print(f"Warning: Filtered {np.sum(~valid_mask)} invalid 3D points")
        pts3D = pts3D[valid_mask]
        pts1 = pts1[valid_mask]
        pts2 = pts2[valid_mask]
    
    return pts3D, pts1, pts2

def compute_reprojection_error(pts3D, pts2D, K, Rt):
    """
    Compute reprojection error between 3D points and their 2D correspondences
    
    Args:
        pts3D: Array of 3D points (Nx3)
        pts2D: Array of corresponding 2D points (Nx2)
        K: Camera intrinsic matrix (3x3)
        Rt: Camera extrinsic matrix (3x4)
        
    Returns:
        Dictionary containing reprojection error metrics
    """
    if len(pts3D) != len(pts2D):
        raise ValueError("Number of 3D points and 2D points must match")
    
    if len(pts3D) == 0:
        return {
            'mean_error': float('inf'),
            'median_error': float('inf'),
            'rmse': float('inf'),
            'inlier_ratio': 0.0,
            'errors': [],
            'projected_points': None,
            'original_points': None
        }
    
    # Project 3D points to 2D
    pts3D_hom = np.hstack((pts3D, np.ones((len(pts3D), 1))))
    proj_pts = (K @ Rt @ pts3D_hom.T).T
    proj_pts = (proj_pts[:, :2] / proj_pts[:, 2, np.newaxis])
    
    # Compute errors
    errors = np.linalg.norm(proj_pts - pts2D, axis=1)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(mean_squared_error(proj_pts, pts2D))
    
    # Adaptive inlier threshold (2% of image diagonal)
    img_diag = np.linalg.norm([K[0,2]*2, K[1,2]*2])  # Approximate image size
    inlier_thresh = img_diag * 0.02
    inlier_ratio = np.mean(errors < inlier_thresh)
    
    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'rmse': rmse,
        'inlier_ratio': inlier_ratio,
        'inlier_threshold': inlier_thresh,
        'errors': errors,
        'projected_points': proj_pts,
        'original_points': pts2D
    }

def visualize_reprojection_errors(img, original_points, projected_points, errors, inlier_thresh, title="Reprojection Errors"):
    """Create a detailed visualization of reprojection errors"""
    plt.figure(figsize=(15, 10))
    
    # Convert image to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a normalized color map for errors
    max_error = np.percentile(errors, 95)  # Use 95th percentile to avoid outlier skew
    norm = plt.Normalize(vmin=0, vmax=max_error)
    cmap = plt.cm.get_cmap('viridis')
    
    # Plot the image
    plt.imshow(img_rgb)
    
    # Plot the correspondences with color-coded errors
    for orig, proj, err in zip(original_points, projected_points, errors):
        color = cmap(norm(err))
        plt.plot([orig[0], proj[0]], [orig[1], proj[1]], '-', 
                 color=color, linewidth=0.5, alpha=0.7)
        plt.plot(orig[0], orig[1], 'o', markersize=3, 
                 color=color, alpha=0.7)
        plt.plot(proj[0], proj[1], 'x', markersize=3, 
                 color=color, alpha=0.7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04)
    cbar.set_label('Reprojection Error (pixels)')
    
    # Add threshold indicator
    plt.axhline(y=img.shape[0] - 20, xmin=0.7, xmax=0.95, color='red', linewidth=2)
    plt.text(img.shape[1] * 0.7, img.shape[0] - 15, 
             f'Inlier Threshold: {inlier_thresh:.1f} px', color='red')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_error_distribution(errors, inlier_thresh, title="Error Distribution"):
    """Create a histogram of reprojection errors"""
    plt.figure(figsize=(10, 6))
    
    # Histogram with logarithmic y-scale
    n, bins, patches = plt.hist(errors, bins=50, color='skyblue', 
                                edgecolor='navy', log=True)
    
    # Add threshold line
    plt.axvline(x=inlier_thresh, color='red', linestyle='--', 
                linewidth=2, label=f'Inlier Threshold ({inlier_thresh:.1f} px)')
    
    # Highlight inlier region
    inlier_mask = bins <= inlier_thresh
    for i in range(len(inlier_mask)-1):
        if inlier_mask[i]:
            patches[i].set_facecolor('green')
            patches[i].set_alpha(0.6)
    
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Frequency (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

def evaluate_reconstruction_with_reprojection(reconstruction_data, output_dir="output/evaluation"):
    """Comprehensive evaluation with multiple visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    results = defaultdict(dict)
    all_errors = []
    
    # Get available matplotlib styles
    available_styles = plt.style.available
    
    plot_style = 'seaborn-v0_8' if 'seaborn-v0_8' in available_styles else 'default'
    plt.style.use(plot_style)
    
    # Get all unique camera indices
    camera_indices = reconstruction_data['camera_indices']
    unique_cameras = set(camera_indices)
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cameras)))

    # ===== 1. Per-view Error Analysis =====
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ===== 2. Spatial Error Distribution ===== 
    fig2 = plt.figure(figsize=(15, 10))
    gs = fig2.add_gridspec(2, len(unique_cameras))
    spatial_axes = []
    for i in range(len(unique_cameras)):
        spatial_axes.append(fig2.add_subplot(gs[0, i]))
    cum_ax = fig2.add_subplot(gs[1, :])

    for idx, cam in enumerate(unique_cameras):
        # Get points for this camera
        mask = [i for i, c in enumerate(camera_indices) if c == cam]
        pts3D = np.vstack([reconstruction_data['points3D'][i] for i in mask])
        pts2D = np.vstack([reconstruction_data['points2D'][i] for i in mask])
        
        # Compute reprojection errors
        reproj_error = compute_reprojection_error(
            pts3D, pts2D,
            reconstruction_data['intrinsics'][cam],
            reconstruction_data['extrinsics'][cam]
        )
        errors = reproj_error['errors']
        
        # Store errors for this view
        results['per_view'][cam] = {
            **reproj_error,
            'points3D': pts3D,
            'points2D': pts2D
        }
        all_errors.extend(errors)
        
        # 1A. Error Distribution Histogram
        ax1.hist(errors, bins=50, alpha=0.7, color=colors[idx], 
                label=f'{cam} (μ={np.mean(errors):.2f}px)')
        
        # 1B. Boxplot
        ax2.boxplot(errors, positions=[idx], patch_artist=True,
                   boxprops=dict(facecolor=colors[idx], alpha=0.7),
                   medianprops=dict(color='red'))
        
        # 2A. Spatial Error Heatmap
        if 'img_shape' in reconstruction_data:
            h, w = reconstruction_data['img_shape'][:2]
            valid_mask = (pts2D[:,0] >= 0) & (pts2D[:,0] < w) & (pts2D[:,1] >= 0) & (pts2D[:,1] < h)
            if np.any(valid_mask):
                heatmap, xedges, yedges = np.histogram2d(
                    pts2D[valid_mask,0], pts2D[valid_mask,1], bins=50, 
                    range=[[0, w], [0, h]], 
                    weights=errors[valid_mask]
                )
                extent = [0, w, 0, h]
                im = spatial_axes[idx].imshow(heatmap.T, extent=extent, origin='lower', 
                                            cmap='hot', aspect='auto')
                spatial_axes[idx].set_title(f'Spatial Errors - {cam}')
                fig2.colorbar(im, ax=spatial_axes[idx], label='Error (px)')
        
        # 2B. Cumulative Error
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cum_ax.plot(sorted_errors, np.linspace(0, 1, len(sorted_errors)), 
                      label=f'{cam} (90%: {np.percentile(errors, 90):.2f}px)', 
                      color=colors[idx])

    # ===== 1. Finalize Comparative Plots =====
    ax1.set_title('Reprojection Error Distribution')
    ax1.set_xlabel('Error (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Error Distribution by Camera')
    ax2.set_xticks(range(len(unique_cameras)))
    ax2.set_xticklabels(unique_cameras)
    ax2.set_ylabel('Error (pixels)')
    ax2.grid(True)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'error_comparison.png'))
    plt.close(fig1)
    
    # ===== 2. Finalize Spatial Analysis =====
    cum_ax.set_title('Cumulative Error Distribution')
    cum_ax.set_xlabel('Error Threshold (px)')
    cum_ax.set_ylabel('Fraction of Points')
    cum_ax.legend()
    cum_ax.grid(True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'spatial_analysis.png'))
    plt.close(fig2)
    
    # ===== 3. 3D Error Visualization =====
    if len(all_errors) > 0:
        # Create colored point cloud by error magnitude
        error_pcd = o3d.geometry.PointCloud()
        all_pts3D = np.vstack([res['points3D'] for res in results['per_view'].values()])
        error_pcd.points = o3d.utility.Vector3dVector(all_pts3D)
        
        # Normalize errors for coloring
        max_error = max(all_errors) if max(all_errors) > 0 else 1
        errors_normalized = np.array(all_errors) / max_error
        
        # Color points (red=high, blue=low)
        colors = np.zeros((len(errors_normalized), 3))
        colors[:,0] = errors_normalized  # Red channel
        colors[:,2] = 1 - errors_normalized  # Blue channel
        error_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save error point cloud
        o3d.io.write_point_cloud(os.path.join(output_dir, 'error_visualization.ply'), error_pcd)
        
        # Error vs Depth analysis
        if len(all_pts3D) == len(all_errors):
            depths = all_pts3D[:,2]
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sc = ax3.scatter(depths, all_errors, alpha=0.5, c=errors_normalized, cmap='viridis')
            ax3.set_title('Error vs Depth')
            ax3.set_xlabel('Depth (Z coordinate)')
            ax3.set_ylabel('Reprojection Error (px)')
            plt.colorbar(sc, ax=ax3, label='Normalized Error')
            fig3.savefig(os.path.join(output_dir, 'error_vs_depth.png'))
            plt.close(fig3)
    
    # Save comprehensive results
    if all_errors:
        results['aggregate'] = {
            'mean_error': np.mean(all_errors),
            'median_error': np.median(all_errors),
            'rmse': np.sqrt(mean_squared_error([0]*len(all_errors), all_errors)),
            'inlier_ratio': np.mean(np.array(all_errors) < 5.0),
            'total_points': len(all_errors),
            'plots': {
                'comparison': os.path.join(output_dir, 'error_comparison.png'),
                'spatial': os.path.join(output_dir, 'spatial_analysis.png'),
                'error_3d': os.path.join(output_dir, 'error_visualization.ply'),
                'depth_analysis': os.path.join(output_dir, 'error_vs_depth.png')
            }
        }
    
    return results

def create_texture_coordinates(vertices, img_shape, K, Rt):
    """Generate proper UV coordinates for texture mapping with boundary handling"""
    h, w = img_shape[:2]
    vertices_hom = np.hstack((vertices, np.ones((len(vertices), 1))))
    proj_pts = (K @ Rt @ vertices_hom.T).T
    proj_pts = proj_pts[:, :2] / proj_pts[:, 2, np.newaxis]
    
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = np.clip(proj_pts[:, 0] / w, 0.01, 0.99)
    uvs[:, 1] = 1 - np.clip(proj_pts[:, 1] / h, 0.01, 0.99)
    
    # For points that project outside the image, use nearest valid point
    invalid_mask = (proj_pts[:, 0] < 0) | (proj_pts[:, 0] >= w) | \
                   (proj_pts[:, 1] < 0) | (proj_pts[:, 1] >= h)
    
    if np.any(invalid_mask):
        valid_uvs = uvs[~invalid_mask]
        if len(valid_uvs) > 0:
            tree = cKDTree(valid_uvs)
            _, idx = tree.query(uvs[invalid_mask], k=1)
            uvs[invalid_mask] = valid_uvs[idx]
    
    return uvs

def check_facial_symmetry(pcd):
    """Remove points that violate basic facial symmetry"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Mirror points across the X-axis (assuming face is centered)
    mirrored = points.copy()
    mirrored[:, 0] *= -1  # Flip X coordinate
    
    # Find nearest neighbors for symmetry check
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    keep_mask = np.ones(len(points), dtype=bool)
    
    for i, point in enumerate(mirrored):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            original_idx = idx[0]
            # If mirrored point is too far from nearest neighbor, discard
            if np.linalg.norm(point - points[original_idx]) > 0.1:  # 10cm threshold
                keep_mask[i] = False
    
    return pcd.select_by_index(np.where(keep_mask)[0])


def reconstruct_face(pcd, img, K, Rt, output_path="output/face",scale_factor=1.0):
    """Reconstruct 3D face mesh with texture mapping and improved cleaning"""
    # 1. Enhanced texture processing
    texture_img = enhance_texture_colors(img)
    
    # 2. Enhanced point cloud processing
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    # More aggressive outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.05)
    
    # Cluster filtering to remove isolated points
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=50))
    if len(labels) > 0:
        max_label = labels.max()
        if max_label >= 0:  # -1 is noise
            largest_cluster = np.argmax(np.bincount(labels[labels>=0]+1))
            pcd = pcd.select_by_index(np.where(labels == largest_cluster-1)[0])
    
    # 3. Normal estimation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.2, max_nn=50))
    pcd.orient_normals_towards_camera_location(np.zeros(3))

    # 4. Poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=12, linear_fit=True)
    
    # 5. Remove low-density vertices
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.25))

    # 6. Post-processing
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    mesh.compute_vertex_normals()

    if scale_factor != 1.0:
        vertices = np.asarray(mesh.vertices)
        mesh.vertices = o3d.utility.Vector3dVector(vertices * scale_factor)
    # 7. Create texture coordinates
    uvs = create_texture_coordinates(
        np.asarray(mesh.vertices), 
        texture_img.shape, 
        K, 
        Rt
    )

    # 8. Save textured mesh
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save texture image
    texture_path = f"{output_path}_texture.png"
    cv2.imwrite(texture_path, texture_img)

    # Write MTL file
    with open(f"{output_path}.mtl", 'w') as f:
        f.write("newmtl face_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 0.9 0.9 0.9\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("Ns 10.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {os.path.basename(texture_path)}\n")

    # Write OBJ file
    with open(f"{output_path}.obj", 'w') as f:
        f.write(f"mtllib {os.path.basename(output_path)}.mtl\n")
        
        # Vertices
        vertices = np.asarray(mesh.vertices)
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Texture coordinates
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Vertex normals
        normals = np.asarray(mesh.vertex_normals)
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # Faces
        triangles = np.asarray(mesh.triangles)
        f.write("usemtl face_material\n")
        for tri in triangles:
            f.write(f"f {tri[0]+1}/{tri[0]+1}/{tri[0]+1} "
                   f"{tri[1]+1}/{tri[1]+1}/{tri[1]+1} "
                   f"{tri[2]+1}/{tri[2]+1}/{tri[2]+1}\n")

    return mesh

def main():
    # Load images
    img_front = load_image('/home/gl05584z/cvp/samples/samples/Camera_Front.png')
    img_left = load_image('/home/gl05584z/cvp/samples/samples/Camera_Left.png')
    img_right = load_image('/home/gl05584z/cvp/samples/samples/Camera_Right.png')
    
    # Load camera parameters
    intrinsics, extrinsics, projections = load_camera_parameters("camera_params.json")

    # Detect facial landmarks with duplicate removal
    lm_front = detect_face_landmarks(img_front)
    lm_left = detect_face_landmarks(img_left)
    lm_right = detect_face_landmarks(img_right)

    # Detect SIFT features
    kp_front, desc_front = detect_sift_features(img_front)
    kp_left, desc_left = detect_sift_features(img_left)
    kp_right, desc_right = detect_sift_features(img_right)


    feature_match_threshold = 0.2  # Can be reduced to 0.2 for tighter matching
    reconstruction_scale = 1.0    # Compensate for shrinkage when threshold is reduced
    
    if feature_match_threshold < 0.3:
        reconstruction_scale = 1.2

    # Match features between view pairs
    matches_LF = match_features(desc_left, desc_front, ratio_thresh=feature_match_threshold)
    matches_RF = match_features(desc_right, desc_front, ratio_thresh=feature_match_threshold)

    # Combine landmarks with SIFT features for robust correspondences
    combined_pts_LF_left, combined_pts_LF_front = combine_landmarks_and_sift(
        lm_left, lm_front, kp_left, kp_front, matches_LF, img_front.shape)
    
    combined_pts_RF_right, combined_pts_RF_front = combine_landmarks_and_sift(
        lm_right, lm_front, kp_right, kp_front, matches_RF, img_front.shape)

    # Triangulate 3D points from all view pairs (with reprojection tracking)
    pts3D_LF, pts2D_LF_left, pts2D_LF_front = triangulate_points(
        projections["left"],
        projections["front"],
        combined_pts_LF_left, combined_pts_LF_front
    )
    pts3D_RF, pts2D_RF_right, pts2D_RF_front = triangulate_points(
        projections["right"], 
        projections["front"],
        combined_pts_RF_right, combined_pts_RF_front
    )

    # Prepare reconstruction data for evaluation
    reconstruction_data = {
        'points3D': [pts3D_LF, pts3D_RF],
        'points2D': [pts2D_LF_left, pts2D_RF_right],
        'camera_indices': ['left', 'right'],
        'projections': projections,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'images': {
            'left': img_left,
            'right': img_right,
            'front': img_front
        },
        'img_shape': img_front.shape
    }

    # Evaluate reconstruction with enhanced visualization
    evaluation = evaluate_reconstruction_with_reprojection(reconstruction_data)

    # Print evaluation results
    print("\n=== Landmarks + SIFT Method Evaluation ===")
    print("Per-view performance:")
    for cam, metrics in evaluation['per_view'].items():
        print(f"  {cam} camera:")
        print(f"    Mean reprojection error: {metrics['mean_error']:.2f} px")
        print(f"    Median error: {metrics['median_error']:.2f} px")
        print(f"    RMSE: {metrics['rmse']:.2f} px")
        print(f"    Inlier ratio: {metrics['inlier_ratio']:.2%}")
    
    if 'aggregate' in evaluation:
        agg = evaluation['aggregate']
        print("\nOverall performance:")
        print(f"  Mean reprojection error: {agg['mean_error']:.2f} px")
        print(f"  Median error: {agg['median_error']:.2f} px")
        print(f"  RMSE: {agg['rmse']:.2f} px")
        print(f"  Inlier ratio: {agg['inlier_ratio']:.2%}")
        print(f"  Total points evaluated: {agg['total_points']}")

    # Create colored point cloud from all 3D points
    pcd = o3d.geometry.PointCloud()
    all_pts3D = np.vstack((pts3D_LF, pts3D_RF))
    pcd.points = o3d.utility.Vector3dVector(all_pts3D)
    
    # Colorize points with red balance correction
    colors = []
    K_front = intrinsics["front"]
    Rt_front = extrinsics["front"]
    
    for pt in np.asarray(pcd.points):
        pt_hom = np.append(pt, 1)
        proj_pt = K_front @ Rt_front @ pt_hom
        proj_pt = (proj_pt[:2] / proj_pt[2]).astype(int)
        
        # Sample 3x3 region for stable colors
        x, y = proj_pt
        x_min = max(x-1, 0)
        x_max = min(x+2, img_front.shape[1])
        y_min = max(y-1, 0)
        y_max = min(y+2, img_front.shape[0])
        
        if x_min < x_max and y_min < y_max:
            patch = img_front[y_min:y_max, x_min:x_max]
            color = np.mean(patch, axis=(0,1))[::-1] / 255.0  # BGR to RGB
            color[0] = color[0] * 0.95  # Reduce red component
        else:
            color = [0.5, 0.5, 0.5]
        colors.append(color)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Apply symmetry check before reconstruction
    pcd = check_facial_symmetry(pcd)

    # Reconstruct textured mesh
    mesh = reconstruct_face(
        pcd, 
        img_front,
        intrinsics["front"],
        extrinsics["front"],
        "output/textured_face",
        scale_factor=reconstruction_scale
    )

    # Visualize
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