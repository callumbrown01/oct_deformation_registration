import cv2
import numpy as np

class TrackingAlgorithms:
    def __init__(self):
        self.algorithms = {
            'TVL1': self.tvl1,
            'DIS': self.dis,
            'Farneback': self.farneback,
            'PCAFlow': self.pcaflow,
            'Lucas-Kanade': self.lucas_kanade,
            'DeepFlow': self.deepflow
        }

    def tvl1(self, img1, img2, params=None):
        # Use provided params or defaults
        if params is None:
            params = {
                'tau': 0.25,
                'lambda_': 0.15,
                'theta': 0.3,
                'nscales': 4,
                'warps': 5,
                'epsilon': 0.02
            }
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
            tau=params.get('tau', 0.25),
            lambda_=params.get('lambda_', 0.15),
            nscales=params.get('nscales', 3),
            warps=params.get('warps', 3),
            epsilon=params.get('epsilon', 0.02)
        )
        tvl1.setTheta(params.get('theta', 0.3))
        flow = tvl1.calc(img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0, None)
        return flow

    def dis(self, img1, img2, params=None):
        # Use provided params or defaults
        if params is None:
            params = {
                'finest_scale': 0,
                'patch_size': 4,
                'patch_stride': 2,
                'grad_descent_iter': 12,
                'var_refine_iter': 3,
                'var_refine_alpha': 20.0,
                'var_refine_delta': 5.0,
                'var_refine_gamma': 10.0,
                'use_mean_normalization': True,
                'use_spatial_propagation': False
            }
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(params.get('finest_scale', 0))
        dis.setPatchSize(params.get('patch_size', 4))
        dis.setPatchStride(params.get('patch_stride', 2))
        dis.setGradientDescentIterations(params.get('grad_descent_iter', 8))
        dis.setVariationalRefinementIterations(params.get('var_refine_iter', 5))
        dis.setVariationalRefinementAlpha(params.get('var_refine_alpha', 40.0))
        dis.setVariationalRefinementDelta(params.get('var_refine_delta', 2.0))
        dis.setVariationalRefinementGamma(params.get('var_refine_gamma', 5.0))
        dis.setUseMeanNormalization(params.get('use_mean_normalization', True))
        dis.setUseSpatialPropagation(params.get('use_spatial_propagation', False))
        flow = dis.calc(img1, img2, None)
        return flow

    def farneback(self, img1, img2, params=None):
        if params is None:
            params = {
                'pyr_scale': 0.7,
                'levels': 3,
                'winsize': 15,
                'iterations': 1,
                'poly_n': 7,
                'poly_sigma': 1.1,
                'flags': 0
            }
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=params.get('pyr_scale', 0.3),
            levels=params.get('levels', 4),
            winsize=params.get('winsize', 27),
            iterations=params.get('iterations', 1),
            poly_n=params.get('poly_n', 5),
            poly_sigma=params.get('poly_sigma', 1.1),
            flags=params.get('flags', 0)
        )
        return flow

    def pcaflow(self, img1, img2, params=None):
        pca = cv2.optflow.createOptFlow_PCAFlow()
        flow = pca.calc(img1, img2, None)
        return flow

    def lucas_kanade(self, img1, img2, params=None):
        h, w = img1.shape
        y, x = np.mgrid[0:h:10, 0:w:10]
        pts = np.stack((x, y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts, None,
            winSize=(27, 27), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05)
        )
        flow_sparse = np.zeros_like(pts)
        valid = status.flatten() == 1
        flow_sparse[valid] = nextPts[valid] - pts[valid]
        flow = np.zeros((h, w, 2), dtype=np.float32)
        for i, pt in enumerate(pts.reshape(-1, 2)):
            x0, y0 = int(pt[0]), int(pt[1])
            if 0 <= x0 < w and 0 <= y0 < h:
                flow[y0, x0] = flow_sparse[i, 0]
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        return flow

    def deepflow(self, img1, img2, params=None):
        df = cv2.optflow.createOptFlow_DeepFlow()
        flow = df.calc(img1, img2, None)
        return flow

    def get_algorithm_names(self):
        return list(self.algorithms.keys())

    def run(self, name, img1, img2, params=None):
        if name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        return self.algorithms[name](img1, img2, params)
