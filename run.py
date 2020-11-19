import cv2
import numpy as np

from model.peaky_rf import PeakyRandomForest

import multiprocessing

num_cores = multiprocessing.cpu_count()

pk_rf = PeakyRandomForest(5, 2, cv2.xfeatures2d.DAISY_create(), use_conf_model=True)
pk_rf.load('pk_rf.pkl')


def drawing_homo(kps1, kps2, des1, des2, img1, img2):
    def compute_homography(matched_kp1, matched_kp2):
        matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
        matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

        H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                        matched_pts2[:, [1, 0]],
                                        cv2.RANSAC)
        inliers = inliers.flatten()
        return H, inliers

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(des1, des2)

    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kps1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kps2[idx] for idx in matches_idx]

    H, inliers = compute_homography(m_kp1, m_kp2)

    matches = np.array(matches)[inliers.astype(bool)].tolist()

    matched_image = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, singlePointColor=(255, 255, 255),
                                    matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image

def read_img(path):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    if W > H:
        scale = 320 / W
    else:
        scale = 240 / H
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def main():
    img1 = read_img('1.ppm')
    img2 = read_img('2.ppm')

    pred_kps1, pred_des1 = pk_rf.pred_ms_kps(img1, scales=[1, 1.5, 2])
    pred_kps2, pred_des2 = pk_rf.pred_ms_kps(img2, scales=[1, 1.5, 2])

    matched_image = drawing_homo(pred_kps1, pred_kps2, pred_des1, pred_des2, img1, img2)
    cv2.imwrite('match_out.png', matched_image)


if __name__ == '__main__':
    main()
