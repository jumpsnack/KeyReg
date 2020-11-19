import cv2
import numpy as np
from core.config import cfg
from joblib import load
import copy

from multiprocessing.dummy import Pool as ThreadPool

import multiprocessing

num_cores = multiprocessing.cpu_count()
pool = ThreadPool(num_cores)
from functools import partial


class PeakyRandomForest(object):
    def __init__(self, n_layers, n_width, detector, use_conf_model=False):
        self.n_layers = n_layers
        self.n_width = n_width
        self.detector = detector
        self.L = []
        self.use_conf_model = use_conf_model

    def load(self, path):
        pickle = load(path)
        self.n_layers = pickle['n_layers']
        self.n_width = pickle['n_width']
        self.L = pickle['L']
        for rfs in self.L:
            for rf in rfs:
                rf.n_jobs = 1
        self.use_conf_model = pickle['use_conf_model']
        self.conf_model = pickle['conf_model']
        for rf in self.conf_model:
            rf.n_jobs = 1

    def kps_nms_conf(self, kps_list, n_wnd, size, conf):
        if isinstance(kps_list, np.ndarray):
            kps_list = [cv2.KeyPoint(k[0], k[1], 10) for k in kps_list]
        H, W = size

        n_hor_grid = int(W / n_wnd)
        n_ver_grid = int(H / n_wnd)

        subset = dict()
        for k, c in zip(kps_list, conf):
            pt = k.pt

            id = int(pt[0] / n_wnd) + int(pt[1] / n_wnd) * n_hor_grid
            if id in subset.keys():
                subset[id].append((k, c))
            else:
                subset[id] = []
                subset[id].append((k, c))

        suppressed_kps = []
        suppressed_conf = []
        for key in subset.keys():
            cands = subset[key]
            if len(cands) > 1:
                max_k = None
                max_c = [-1, -1]
                for k, c in cands:
                    if c[1] > max_c[1]:
                        max_c = c
                        max_k = k
                suppressed_kps += [max_k]
                suppressed_conf += [max_c]
            else:
                suppressed_kps += [k for k, c in cands]
                suppressed_conf += [c for k, c in cands]

        return suppressed_kps, suppressed_conf

    def _get_init_kps(self, img):
        H, W = img.shape[:2]

        hor = list(range(int(cfg.common.n_grid / 2), W, cfg.common.n_grid))
        ver = list(range(int(cfg.common.n_grid / 2), H, cfg.common.n_grid))

        cv_kps = []
        for h in hor:
            for v in ver:
                cv_kps.append(cv2.KeyPoint(h, v, 1, 0, 0, 0, 0))

        return cv_kps

    def pred_cscd(self, img):
        def get_near_kps(kps, w_size, size):
            out_kps_set = []
            passed_kps = []

            def _job(out_kps_set, passed_kps, k):
                org_pt = copy.deepcopy(k.pt)

                nn_kps = []
                for v in range(int(-w_size / 3), int(w_size / 3) + 1, int(w_size / 3)):
                    for h in range(int(-w_size / 3), int(w_size / 3) + 1, int(w_size / 3)):
                        new_k = cv2.KeyPoint(org_pt[0] + h, org_pt[1] + v, k.size, k.angle, k.response, k.octave,
                                             k.class_id)
                        nn_kps.append(new_k)

                inv_cnt = 0
                for nn_k in nn_kps:
                    if nn_k.pt[0] <= 0 or nn_k.pt[1] <= 0: inv_cnt += 1
                    if nn_k.pt[0] >= size[0] or nn_k.pt[1] >= size[1]: inv_cnt += 1
                if inv_cnt == 0:
                    out_kps_set.append(nn_kps)
                    passed_kps.append(k)

            func = partial(_job, out_kps_set, passed_kps)
            pool.map(func, kps)
            return passed_kps, out_kps_set

        def get_near_des(img, nn_kps_set):
            daisy = cv2.xfeatures2d.DAISY_create()
            out = daisy.compute(img, sum(nn_kps_set, []))[1].reshape(-1, 9 * 200)
            return out

        center_kps = self._get_init_kps(img)

        layer = self.L[0]
        _layer_pred = []

        H, W = img.shape[:2]

        center_kps, nn_kps = get_near_kps(center_kps, cfg.common.n_grid, (W, H))
        nn_des = get_near_des(img, nn_kps)

        def _pred_job(nn_des, rf):
            return rf.predict(nn_des)

        func = partial(_pred_job, nn_des)
        _layer_pred = pool.map(func, layer)
        n_disp = np.mean(_layer_pred, axis=0)
        center_kps = cv2.KeyPoint_convert(center_kps)

        new_center_kps = np.asarray(center_kps) + n_disp * cfg.common.n_grid
        mask_a = np.logical_and(new_center_kps[:, 0] > 0, new_center_kps[:, 1] > 0)
        mask_b = np.logical_and(new_center_kps[:, 0] < W - 1, new_center_kps[:, 1] < H - 1)
        mask = np.logical_and(mask_a, mask_b)
        new_center_kps = new_center_kps[mask]
        new_center_kps = [cv2.KeyPoint(k[0], k[1], 10) for k in new_center_kps]

        for i in range(1, len(self.L), 1):
            _layer_pred = []
            layer = self.L[i]

            center_kps, nn_kps = get_near_kps(new_center_kps, cfg.common.n_grid, (W, H))
            nn_des = get_near_des(img, nn_kps)

            def _pred_job(nn_des, rf):
                return rf.predict(nn_des)

            func = partial(_pred_job, nn_des)
            _layer_pred = pool.map(func, layer)
            n_disp = np.mean(_layer_pred, axis=0)
            kps = [k.pt for k in center_kps]

            new_center_kps = np.asarray(kps) + n_disp * cfg.common.n_grid
            mask_a = np.logical_and(new_center_kps[:, 0] > 0, new_center_kps[:, 1] > 0)
            mask_b = np.logical_and(new_center_kps[:, 0] < W - 1, new_center_kps[:, 1] < H - 1)
            mask = np.logical_and(mask_a, mask_b)
            new_center_kps = new_center_kps[mask]
            new_center_kps = [cv2.KeyPoint(k[0], k[1], 10) for k in new_center_kps]

        return new_center_kps

    def pred_ms_kps(self, img, scales=[1, 1.5, 2]):
        H, W = img.shape[:2]

        def _job(scale):
            rescaled_img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
            rescaled_kps = self.pred_cscd(rescaled_img)

            np_kps = np.asarray([k.pt for k in rescaled_kps])
            np_kps *= (1 / scale)
            mask_a = np.logical_and(np_kps[:, 0] > 0, np_kps[:, 1] > 0)
            mask_b = np.logical_and(np_kps[:, 0] < W - 1, np_kps[:, 1] < H - 1)
            mask = np.logical_and(mask_a, mask_b)
            np_kps = np_kps[mask]
            out_kps = [cv2.KeyPoint(k[0], k[1], 10) for k in np_kps]
            return out_kps

        pred_kps = pool.map(_job, scales)
        pred_kps = sum(pred_kps, [])

        _, desc = self.detector.compute(img, pred_kps)

        _conf_pred = []
        for rf in self.conf_model:
            conf_pred = rf.predict_proba(desc)
            _conf_pred.append(conf_pred)
        _conf_pred = np.mean(_conf_pred, axis=0)

        pred_kps, pred_conf = self.kps_nms_conf(pred_kps, 3, img.shape[:2], _conf_pred)
        pred_kps = [pred_kps[i] for i in range(len(pred_kps)) if np.argmax(pred_conf, axis=1)[i]]

        _, desc = self.detector.compute(img, pred_kps)

        return pred_kps, desc
