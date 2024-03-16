import os
import numpy as np
import cv2
import math
import json
import IPython
import open3d as o3d
import argparse
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import numpy.matlib as npm
from superpose3d import Superpose3D

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

from face_geometry import PCF, get_metric_landmarks, procrustes_landmark_basis
from face_geometry import canonical_metric_landmarks


def save_image(path, img, pts2D, thickness):
    img_ = img.copy()
    for i in range(pts2D.shape[0]):
        cv2.circle(img_, (int(pts2D[i, 0]), int(pts2D[i, 1])), 1, [0, 0, 255], thickness)
    cv2.imwrite(path, img_)


def convert_trans_matrix(back_front_trans, transform_rear):
    '''
    convert the transformation of rear-facing camera to front-facing camera
    :param transformation:
    :return:
    '''
    if transform_rear.shape[0] == 3:
        transform_rear = np.vstack((transform_rear, np.array([0., 0., 0., 1.])))

    assert transform_rear.shape[0] == 4

    transform_front = np.matmul(back_front_trans, transform_rear)
    transform_front = np.matmul(transform_front, LA.inv(back_front_trans))

    return transform_front


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def triangulate(proj_prev, proj_next, kp1, kp2):
    '''
    Function to triangulate and get the 3D points
    '''
    kp1 = kp1.T
    kp2 = kp2.T
    point_cloud = cv2.triangulatePoints(proj_prev, proj_next, kp1, kp2)
    point_cloud /= point_cloud[3]

    return kp1, kp2, point_cloud


def pnp_ransac(point_cloud, kp1, kp2, K, distortion_coeff, args):
    '''
    Function of perspective-n-point method
    '''
    method = args.pnp_method
    reprojErr = args.reprojection_error
    iterCount = args.iterations_count
    confid = args.confidence
    if method == 'iterative':
        _, rot, trans, inliers = cv2.solvePnPRansac(
            point_cloud, kp1, K, distortion_coeff, cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=iterCount, reprojectionError=reprojErr, confidence=confid,
        )
    elif method == 'epnp':
        _, rot, trans, inliers = cv2.solvePnPRansac(
            point_cloud, kp1, K, distortion_coeff, flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=iterCount, reprojectionError=reprojErr, confidence=confid,
        )
    elif method == 'p3p':
        _, rot, trans, inliers = cv2.solvePnPRansac(
            point_cloud, kp1, K, distortion_coeff, flags=cv2.SOLVEPNP_P3P,
            iterationsCount=iterCount, reprojectionError=reprojErr, confidence=confid,
        )
    elif method == 'sqpnp':
        _, rot, trans, inliers = cv2.solvePnPRansac(
            point_cloud, kp1, K, distortion_coeff, flags=cv2.SOLVEPNP_SQPNP,
            iterationsCount=iterCount, reprojectionError=reprojErr, confidence=confid,
        )
    else:
        IPython.embed(header='unknown PnP method')

    inliers = inliers[:, 0]
    rot, _ = cv2.Rodrigues(rot)

    return rot, trans, kp1[inliers], kp2[inliers], point_cloud[inliers]


def get_reproj_err(point_cloud, kp, trans_matrix, K, distCoeffs=None):
    """
    Function to calculate the reprojection error of 3d point cloud compared to image plane
    """
    rot, _ = cv2.Rodrigues(trans_matrix[:3, :3])
    kp_reprojected, _ = cv2.projectPoints(point_cloud, rot, trans_matrix[:3, 3], K, distCoeffs)
    kp_reprojected = np.float32(kp_reprojected[:, 0, :])
    errs = LA.norm(kp_reprojected - kp, axis=1)

    return errs, kp_reprojected


def trifocal_view(kp2, kp2_next, kp3):
    """
    Function to return common points in the 3 images and a set of non-common points
    Kp2 are keypoints we got from image(n-1) and image(n)
    kp2_next and kp3 are the keypoints we got from image(n) and image(n+1)
    idx1 - for matched items in kp2
    idx2 - for matched items in kp2_next & kp3
    """
    idx1 = []
    idx2 = []
    for i in range(kp2.shape[0]):
        if (kp2[i, :] == kp2_next).any():
            idx1.append(i)
        x = np.where(kp2_next == kp2[i, :])
        if x[0].size != 0:
            idx2.append(x[0][0])

    kp3_uncommon = []
    kp2_next_uncommon = []
    for k in range(kp3.shape[0]):
        if k not in idx2:
            kp3_uncommon.append(list(kp3[k, :]))
            kp2_next_uncommon.append(list(kp2_next[k, :]))

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    kp2_next_common = kp2_next[idx2]
    kp3_common = kp3[idx2]

    return idx1, kp2_next_common, kp3_common, np.array(kp2_next_uncommon), np.array(kp3_uncommon)


def remove_dup_kp(kp1, kp2, cloud=None):
    '''
    kp1, kp2: shape [N, 2]
    cloud: shape [N, 3]
    :return:
    '''
    # Remove duplicate points
    if cloud is None:
        kp_pair = np.hstack((kp1, kp2))
    else:
        kp_pair = np.hstack((kp1, kp2, cloud))

    kp_pair_uniq = np.unique(kp_pair, axis=0)
    kp1_uniq = kp_pair_uniq[:, :2]
    kp2_uniq = kp_pair_uniq[:, 2:4]
    if cloud is not None:
        cloud = kp_pair_uniq[:, 4:]

    return kp1_uniq, kp2_uniq, cloud


def save_points(filename, point_3d):
    """
    Saves the reconstructed 3D points to ply files using Open3D
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_3d)
    o3d.io.write_point_cloud(filename, pcd)


def save_color_cloud(final_point_cloud, point_colours, filepath):
    """
    Function to generate an .ply 3D object file that can be rendered in MeshLab
    """
    output_points = final_point_cloud.reshape(-1, 3)
    output_colors = point_colours.reshape(-1, 3)
    mesh = np.hstack([output_points, output_colors])
    mesh_mean = np.mean(mesh[:, :3], axis=0)
    diff = mesh[:, :3] - mesh_mean
    distance = LA.norm(diff, axis=1)  # same result as the above line
    index = np.where(distance < np.mean(distance) + 1.5)
    mesh = mesh[index]

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    with open(filepath, 'w') as f:
        f.write(ply_header % dict(vert_num=len(mesh)))
        np.savetxt(f, mesh, '%f %f %f %d %d %d')
    print("Point cloud was generated and saved!")


class StereoSfM:
    def __init__(self, args, savedir):
        self.args = args
        self.size = tuple([int(x) for x in args.size.split('x')])
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

        with open(args.stereo_calib, 'r') as f:
            self.calibration = json.load(f)

        self.E = np.array(self.calibration['general']['essential'])
        self.R = np.array(self.calibration['general']['rotation'])
        self.t = np.array(self.calibration['general']['translation'])

        self.K_l = np.array(self.calibration['left']['matrix'])
        self.K_r = np.array(self.calibration['right']['matrix'])

        self.distort_l = np.array(self.calibration['left']['distortion'])
        self.distort_r = np.array(self.calibration['right']['distortion'])

        self.proj_l = np.array(self.calibration['left']['projection'])
        self.proj_r = np.array(self.calibration['right']['projection'])

        self.M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        self.M_r = np.hstack((self.R, self.t))

        self.P_l = np.dot(self.K_l, self.M_l)
        self.P_r = np.dot(self.K_r, self.M_r)

        self.old_left = None
        self.old_left_gray = None
        self.old_right = None
        self.old_right_gray = None

        self.kp_old_l = None
        self.kp_old_r = None
        self.old_cloud = None

        self.new_left = None
        self.new_left_gray = None
        self.new_right = None
        self.new_right_gray = None

        self.result_cloud = None
        self.result_colors = None

        self.pnp_method = args.pnp_method

        # https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
        self.nfeatures = 0  # default 0
        self.nOctaveLayers = 3  # default 3.
        self.contrastThreshold = 0.04  # default 0.04, larger is more strict
        self.sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.nOctaveLayers,
            contrastThreshold=self.contrastThreshold,
        )

        self.grid_size = args.grid_size
        self.max_points = args.max_points


    def initialize_cloud(self, imgl, imgr):
        namel = imgl.split('/')[-1].split('.')[-2][-2:]
        self.old_left = cv2.imread(imgl)
        self.old_left = cv2.resize(self.old_left, self.size)
        self.old_left_gray = cv2.cvtColor(self.old_left, cv2.COLOR_BGR2GRAY)

        self.old_right = cv2.imread(imgr)
        self.old_right = cv2.resize(self.old_right, self.size)
        self.old_right_gray = cv2.cvtColor(self.old_right, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.feature_detection(self.old_left, f'{namel}_ultra_sift_keypoints')
        kp2, des2 = self.feature_detection(self.old_right, f'{namel}_wide_sift_keypoints')
        pts_l, pts_r = self.feature_matching(kp1, kp2, des1, des2)

        pts_l_norm = cv2.undistortPoints(pts_l, cameraMatrix=self.K_l, distCoeffs=self.distort_l)  # World coordinate
        pts_r_norm = cv2.undistortPoints(pts_r, cameraMatrix=self.K_r, distCoeffs=self.distort_r)

        self.kp_old_l = pts_l
        self.kp_old_r = pts_r
        kp1 = pts_l_norm.squeeze()
        kp2 = pts_r_norm.squeeze()

        kp1, kp2, cloud = triangulate(self.M_l, self.M_r, kp1, kp2)
        cloud = cv2.convertPointsFromHomogeneous(cloud.T)
        cloud = cloud.squeeze()

        self.kp_old_l, self.kp_old_r, cloud = self.filter_outliers(self.kp_old_l, self.kp_old_r, cloud)

        error_l, repro_pts = get_reproj_err(cloud, self.kp_old_l, self.M_l, self.K_l, self.distort_l)
        error_r, repro_pts = get_reproj_err(cloud, self.kp_old_r, self.M_r, self.K_r, self.distort_r)

        print('pose1 reproj error_l: ', error_l.mean())
        print('pose1 reproj error_r: ', error_r.mean())

        print("Number of 3D points: ", cloud.shape[0])

        # Remove duplicate points
        self.kp_old_l, self.kp_old_r, cloud = remove_dup_kp(self.kp_old_l, self.kp_old_r, cloud)
        self.old_cloud = cloud
        self.result_cloud = cloud

        kp_for_intensity = np.array(self.kp_old_r, dtype=np.int32)
        self.result_colors = []
        for intensity in kp_for_intensity:
            self.result_colors.append(self.old_right[intensity[1], intensity[0], :])
        self.result_colors = np.array(self.result_colors)

        cloud_file = os.path.join(self.savedir, 'scene_point_cloud.ply')
        save_color_cloud(self.result_cloud, self.result_colors, cloud_file)


    def update_frame(self, imgl, imgr):
        namel = imgl.split('/')[-1].split('.')[-2][-2:]
        self.new_left = cv2.imread(imgl)
        self.new_left = cv2.resize(self.new_left, self.size)
        self.new_left_gray = cv2.cvtColor(self.new_left, cv2.COLOR_BGR2GRAY)

        self.new_right = cv2.imread(imgr)
        self.new_right = cv2.resize(self.new_right, self.size)
        self.new_right_gray = cv2.cvtColor(self.new_right, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.feature_detection(self.old_left)
        kp2, des2 = self.feature_detection(self.new_left, f'{namel}_ultra_sift_keypoints')
        old_kp_l, new_kp_l = self.feature_matching(kp1, kp2, des1, des2)

        # Remove duplicate points
        old_kp_l, new_kp_l, _ = remove_dup_kp(old_kp_l, new_kp_l)

        index, old_kp_l_common, new_kp_l_common, old_kp_l_uncommon, new_kp_l_uncommon = trifocal_view(self.kp_old_l, old_kp_l, new_kp_l)
        print("Number of common points: ", old_kp_l_common.shape[0])
        cloud = self.old_cloud[index]

        self.draw_tri_matches(old_kp_l_common, self.kp_old_r[index], new_kp_l_common, 'feature_matching')

        new_kp_l_common, old_kp_l_common, cloud = self.feature_point_binning(new_kp_l_common, old_kp_l_common, cloud)
        print("Number of points after binning: ", old_kp_l_common.shape[0])

        rot, trans, new_kp_l_common, old_kp_l_common, cloud = pnp_ransac(
            cloud, new_kp_l_common, old_kp_l_common, self.K_l, self.distort_l, self.args
        )
        print("Number of inliers: ", cloud.shape[0])

        r = R.from_matrix(rot)
        angles = r.as_euler('xyz', degrees=True)

        trans_mat_new = np.hstack((rot, trans))
        error1, kp_projected = get_reproj_err(cloud, new_kp_l_common, trans_mat_new, self.K_l, self.distort_l)
        print('pose2 reproj error_l: ', error1.mean())

        print("estimated transformation is:")
        print(trans_mat_new)
        print("Euler angels (xyz) is:")
        print(angles)
        print('------- finished camera transformation estimation -------')

        return trans_mat_new

    def feature_detection(self, img, file_name=None):
        # Turning RGB images to gray-scale images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # SIFT for feature extraction:
        kp, des = self.sift.detectAndCompute(img_gray, None)
        des = np.float32(des)

        return kp, des

    def feature_matching(self, kp1, kp2, des1, des2):
        # FLANN for featuer matching:
        FLANN_INDEX_KDTREE = 1
        idx_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(idx_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for _ in range(len(matches))]
        # ratio test as per Lowe's paper
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good_matches.append(m)

        kp1_good = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        kp2_good = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return kp1_good, kp2_good

    def filter_outliers(self, kp_l, kp_r, cloud):
        error_l, _ = get_reproj_err(cloud, kp_l, self.M_l, self.K_l, self.distort_l)
        error_r, _ = get_reproj_err(cloud, kp_r, self.M_r, self.K_r, self.distort_r)

        idx_l = error_l < self.args.reprojection_error_stereo
        idx_r = error_r < self.args.reprojection_error_stereo
        idx = idx_l & idx_r
        kp_l, kp_r, cloud = kp_l[idx, :], kp_r[idx, :], cloud[idx, :]

        idx_near = cloud[:, 2] < self.args.max_depth
        kp_l, kp_r, cloud = kp_l[idx_near, :], kp_r[idx_near, :], cloud[idx_near, :]

        return kp_l, kp_r, cloud

    def feature_point_binning(self, kp_new, kp_old, cloud):
        w, h = self.size
        errors, _ = get_reproj_err(cloud, kp_old, self.M_l, self.K_l, self.distort_l)

        kp_old_bin = []
        kp_new_bin = []
        cloud_bin = []
        for y in range(0, h, self.grid_size):
            for x in range(0, w, self.grid_size):
                idx = (kp_new[:, 0] >= x) & (kp_new[:, 0] < x+self.grid_size) & (kp_new[:, 1] >= y) & (kp_new[:, 1] < y+self.grid_size)
                kp_grid = kp_new[idx, :]
                if kp_grid.shape[0] == 0:
                    continue
                if kp_grid.shape[0] <= self.max_points:
                    kp_old_bin.append(kp_old[idx, :])
                    kp_new_bin.append(kp_new[idx, :])
                    cloud_bin.append(cloud[idx, :])
                else:
                    idx2 = np.argsort(errors[idx])[:self.max_points]
                    kp_old_bin.append(kp_old[idx, :][idx2, :])
                    kp_new_bin.append(kp_new[idx, :][idx2, :])
                    cloud_bin.append(cloud[idx, :][idx2, :])

        kp_old_bin = np.vstack(kp_old_bin)
        kp_new_bin = np.vstack(kp_new_bin)
        cloud_bin = np.vstack(cloud_bin)

        return kp_new_bin, kp_old_bin, cloud_bin

    def draw_tri_matches(self, old_kp_l_common, old_kp_r_common, new_kp_l_common, filename):
        # initialize the output visualization image
        h, w = self.old_left.shape[:2]
        vis = np.zeros((2 * h, 2 * w, 3), dtype="uint8")
        vis[:h, :w] = self.old_left
        vis[:h, w:] = self.old_right
        vis[h:, :w] = self.new_left
        vis[h:, w:] = np.array([255, 255, 255], dtype=np.uint8)

        num_points = old_kp_l_common.shape[0]

        for i in range(num_points):
            # old left-right
            ptA = (int(old_kp_l_common[i][0]), int(old_kp_l_common[i][1]))
            ptB = (int(old_kp_r_common[i][0]) + w, int(int(old_kp_r_common[i][1])))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

            # left-left
            ptA = (int(old_kp_l_common[i][0]), int(old_kp_l_common[i][1]))
            ptB = (int(new_kp_l_common[i][0]), int(new_kp_l_common[i][1]) + h)
            cv2.line(vis, ptA, ptB, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(self.savedir, f'{filename}.png'), vis)


class FaceLandmarkRecon:
    def __init__(self, args, savedir, transform_front, pupil_dist_measure=None):
        self.size = tuple([int(x) for x in args.size.split('x')])
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

        with open(args.front_calib, 'r') as f:
            self.calibration = json.load(f)

        self.transformation = transform_front
        self.K = np.array(self.calibration['matrix'])
        self.distort = np.array(self.calibration['distortion'])

        self.M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        self.M_r = self.transformation[:3, :4]
        self.P_l = np.dot(self.K, self.M_l)
        self.P_r = np.dot(self.K, self.M_r)

        self.pupil_dist_measure = pupil_dist_measure
        self.face = args.face
        self.point_3d = None
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        # For rotated image
        focal_length = self.K[1, 1]
        self.pcf = PCF(
            near=1,
            far=10000,
            frame_height=self.size[0],
            frame_width=self.size[1],
            fy=focal_length,
        )
        self.camera_matrix = np.array(
            [[focal_length, 0, self.size[1]/2], [0, focal_length, self.size[0]/2], [0, 0, 1]],
            dtype="double",
        )
        self.dist_coeff = np.zeros((4, 1))

    def scale_estimate(self, imgl, imgr):
        namel = imgl.split('/')[-1].split('.')[-2][-2:]
        namer = imgr.split('/')[-1].split('.')[-2][-2:]

        left_org = cv2.imread(imgl, cv2.IMREAD_UNCHANGED)
        left_org = cv2.resize(left_org, self.size)
        left_bgr = cv2.rotate(left_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

        right_org = cv2.imread(imgr, cv2.IMREAD_UNCHANGED)
        right_org = cv2.resize(right_org, self.size)
        right_bgr = cv2.rotate(right_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        h, w = right.shape[:2]
        left_center_idx = [144, 145, 153, 158, 159, 160]
        right_center_idx = [373, 374, 380, 385, 386, 387]

        results = self.face_mesh.process(left)
        preds = results.multi_face_landmarks[0].landmark
        preds_3d = np.array([(lm.x, lm.y, lm.z) for lm in preds])
        preds_3d = preds_3d[:468, :]
        pred_2d = preds_3d.copy()[:, :2]
        pred_2d_rot = pred_2d.copy()[:, ::-1]
        pred_2d_rot[:, 0] = h - h * pred_2d_rot[:, 0]
        pred_2d_rot[:, 1] = w * pred_2d_rot[:, 1]
        save_image(os.path.join(self.savedir, f'{namel}_face_landmarks.jpg'), left_org, pred_2d_rot, 3)
        eye_center_2d = np.vstack((pred_2d_rot[left_center_idx, :].mean(0), pred_2d_rot[right_center_idx, :].mean(0)))
        pts_l = np.expand_dims(pred_2d_rot, axis=1)

        metric_landmarks_l, pose_transform_mat_l = get_metric_landmarks(preds_3d.copy().T, self.pcf)
        metric_landmarks_l = metric_landmarks_l.T

        results = self.face_mesh.process(right)
        preds = results.multi_face_landmarks[0].landmark
        preds_3d = np.array([(lm.x, lm.y, lm.z) for lm in preds])
        preds_3d = preds_3d[:468, :]

        pred_2d = preds_3d.copy()[:, :2]
        pred_2d_rot = pred_2d.copy()[:, ::-1]
        pred_2d_rot[:, 0] = h - h * pred_2d_rot[:, 0]
        pred_2d_rot[:, 1] = w * pred_2d_rot[:, 1]
        save_image(os.path.join(self.savedir, f'{namer}_face_landmarks.jpg'), right_org, pred_2d_rot, 3)
        eye_center_2d = np.vstack((pred_2d_rot[left_center_idx, :].mean(0), pred_2d_rot[right_center_idx, :].mean(0)))

        metric_landmarks_r, pose_transform_mat_r = get_metric_landmarks(preds_3d.copy().T, self.pcf)
        metric_landmarks_r = metric_landmarks_r.T

        pose_transform_mat_r[1:3, :] = -pose_transform_mat_r[1:3, :]
        mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat_r[:3, :3])
        mp_translation_vector = pose_transform_mat_r[:3, 3, None]

        half_wide = np.max(metric_landmarks_r[:, 0])
        top = np.max(metric_landmarks_r[:, 1])
        bottom = np.min(metric_landmarks_r[:, 1])
        nose_tip = np.max(metric_landmarks_r[:, 2])
        mid_idx = (metric_landmarks_r[:, 0] < 0.5 * half_wide) & (metric_landmarks_r[:, 0] > -0.5 * half_wide) & \
                  (metric_landmarks_r[:, 1] < 0.5 * top) & (metric_landmarks_r[:, 1] > 0.7 * bottom) & \
                  (metric_landmarks_r[:, 2] > 0.25 * nose_tip)

        scale_factor = 0.01
        pt3D_adjust = scale_factor * metric_landmarks_r.copy()
        pt3D_adjust[:, :2] = pt3D_adjust[:, 1::-1]
        pt3D_adjust[:, 2] *= -1

        register_errors = []
        pupil_dist_estimates = []

        angle_range = range(-20, 20, 1)
        interval = 0.5

        for i_rot in angle_range:

            rot_y = interval * i_rot
            head_rot = R.from_euler('y', rot_y, degrees=True)
            head_R = head_rot.as_matrix()

            metric_landmarks_r_rot = np.matmul(head_R, metric_landmarks_r.T)
            metric_landmarks_r_rot = metric_landmarks_r_rot.T

            pred_2d_reproj, jacobian = cv2.projectPoints(
                metric_landmarks_r_rot,
                mp_rotation_vector,
                mp_translation_vector,
                self.camera_matrix,
                self.dist_coeff,
            )
            pred_2d_reproj = pred_2d_reproj.squeeze()
            pred_2d_rot = pred_2d_reproj.copy()[:, ::-1]
            pred_2d_rot[:, 0] = h - pred_2d_rot[:, 0]
            pred_2d_rot[:, 1] = pred_2d_rot[:, 1]
            pts_r = np.expand_dims(pred_2d_rot, axis=1)

            pts_l_norm = cv2.undistortPoints(pts_l, cameraMatrix=self.K, distCoeffs=self.distort)
            pts_r_norm = cv2.undistortPoints(pts_r, cameraMatrix=self.K, distCoeffs=self.distort)
            point_4d_hom = cv2.triangulatePoints(self.M_l, self.M_r, pts_l_norm, pts_r_norm)
            point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
            point_3d = point_4d[:3, :].T

            rmsd, register_error, register_points = self.point_register(pt3D_adjust[mid_idx], point_3d[mid_idx])
            register_errors.append(register_error.mean())

            pupil_dist_estimate = LA.norm(point_3d[left_center_idx, :].mean(0) - point_3d[right_center_idx, :].mean(0))
            pupil_dist_estimates.append(pupil_dist_estimate)
            print(f"Rotation {rot_y} degree, register error {register_error.mean():.6f}, RMSD {rmsd:.6f}")

        register_errors = np.asarray(register_errors)
        idx_min = np.argmin(register_errors)
        pupil_dist_estimate = pupil_dist_estimates[idx_min]
        print(f"Estimated head rotation: {interval * angle_range[idx_min]} degree")
        print(f"Estimated pupil distance: {pupil_dist_estimate * 1000.} mm")

        output = {
            'Name': self.face,
            'PD_estimate': pupil_dist_estimate,
        }

        if self.pupil_dist_measure != None:
            pupil_dist_err = pupil_dist_estimate - self.pupil_dist_measure
            pupil_dist_err_perc = pupil_dist_err / self.pupil_dist_measure
            print(f"PD measurement (GT): {self.pupil_dist_measure * 1000.} mm")
            print(f"PD estimation error (absolute): {pupil_dist_err * 1000.} mm")
            print(f"PD estimation error (relative): {pupil_dist_err_perc * 100.} %")
            output1 = {
                'PD_ground_truth': self.pupil_dist_measure,
                'error': pupil_dist_err,
                'error_percent': pupil_dist_err_perc,
            }
            output.update(output1)

        out_file = os.path.join(self.savedir, f'pupil_dist_estimate_results.json')
        with open(out_file, 'w') as f:
            json.dump(output, f, cls=NumpyArrayEncoder, indent=4)

    def point_register(self, frozen_pts, mobile_pts):
        num_pts = mobile_pts.shape[0]
        rmsd, R, t, scale = Superpose3D(frozen_pts, mobile_pts, None, allow_rescale=True, report_quaternion=False)

        Rt = np.hstack((R, np.expand_dims(t, 1)))
        reproj = np.dot(Rt, np.hstack((mobile_pts * scale, np.ones((num_pts, 1)))).T).T
        reproj = reproj[:, :3]
        reproj_err = reproj - frozen_pts
        reproj_err_l2 = LA.norm(reproj_err, axis=1)

        return rmsd, reproj_err_l2, reproj


parser = argparse.ArgumentParser(description='Swing for True Scale. ISMAR 2023.')
parser.add_argument("--stereo_calib", type=str, action="store", default='params/s22_rear_stereo_params.json', help="Back-facing camera parameter json file")
parser.add_argument("--front_calib", type=str, action="store", default='params/s22_front_params.json', help="Front-facing camera parameter json file")
parser.add_argument("--rear_to_front", type=str, action="store", default='params/s22_rear_to_front.txt', help="Transformation from rear to front camera")
parser.add_argument("--input_dir", type=str, action="store", default='data/swing1', help="input data folder")
parser.add_argument("--size", type=str, action="store", default='1920x1440', help="Image size")
parser.add_argument("--pnp_method", type=str, action="store", default='iterative', help="PnP method", choices=['iterative', 'epnp', 'p3p', 'sqpnp'])
parser.add_argument("--reprojection_error_stereo", type=float, action="store", default=8, help="PnP RANSAC parameter")
parser.add_argument("--reprojection_error", type=float, action="store", default=12, help="PnP RANSAC parameter")
parser.add_argument("--iterations_count", type=int, action="store", default=100, help="PnP RANSAC parameter")
parser.add_argument("--confidence", type=float, action="store", default=0.99, help="PnP RANSAC parameter")
parser.add_argument("--grid_size", type=int, action="store", default=120, help="Grid size for feature point binning")
parser.add_argument("--max_points", type=int, action="store", default=10, help="max number of points in each grid")
parser.add_argument("--max_depth", type=float, action="store", default=100., help="Depth threshold of filtering 3D points")
parser.add_argument("--face", type=str, action="store", default='Rui', help="whose face")
args = parser.parse_args()

def main(args):
    dir_l = os.path.join(args.input_dir, 'ultra')
    dir_r = os.path.join(args.input_dir, 'wide')
    dir_f = os.path.join(args.input_dir, 'front')

    pupil_dist_measures = {
        'Rui': 0.0757,
        'Jack': 0.0647,
    }
    pupil_dist_measure = pupil_dist_measures[args.face] if args.face in pupil_dist_measures else None

    f_i, f_j = 1, 2
    savedir = os.path.join(args.input_dir, 'pd_estimate_results')

    sfm = StereoSfM(args, savedir)
    sfm.initialize_cloud(os.path.join(dir_l, f'ultra_{f_i:02d}.jpg'), os.path.join(dir_r, f'wide_{f_i:02d}.jpg'))
    transform_rear = sfm.update_frame(os.path.join(dir_l, f'ultra_{f_j:02d}.jpg'), os.path.join(dir_r, f'wide_{f_j:02d}.jpg'))

    rear_front_trans = np.loadtxt(args.rear_to_front)
    transform_front = convert_trans_matrix(rear_front_trans, transform_rear)

    face_recon = FaceLandmarkRecon(args, savedir, transform_front, pupil_dist_measure)
    face_recon.scale_estimate(os.path.join(dir_f, f'front_{f_i:02d}.jpg'), os.path.join(dir_f, f'front_{f_j:02d}.jpg'))


if __name__ == "__main__":
    main(args)

