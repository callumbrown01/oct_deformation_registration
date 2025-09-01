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
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0, None)
        return flow

    def dis(self, img1, img2, params=None):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(img1, img2, None)
        return flow

    def farneback(self, img1, img2, params=None):
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=0.3, levels=4, winsize=27, iterations=1,
            poly_n=5, poly_sigma=1.1, flags=0
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
