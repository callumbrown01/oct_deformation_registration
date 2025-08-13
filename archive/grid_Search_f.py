import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class OpticalFlowPage(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Available algorithms
        self.algorithms = ['Farneback', 'Lucas-Kanade', 'Speckle Tracking', 'TVL1']
        try:
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        except Exception:
            self.algorithms.remove('TVL1')
            self.tvl1 = None

        # Controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(fill='x', pady=10)
        ttk.Button(ctrl, text="Select Folder", command=self.load_images).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Previous Image", command=self.prev_image).pack(side='left', padx=5)
        ttk.Button(ctrl, text="Next Image", command=self.next_image).pack(side='left', padx=5)

        ttk.Label(ctrl, text="Algorithm:").pack(side='left', padx=(20,5))
        self.selected_alg = tk.StringVar(value=self.algorithms[0])
        alg_combo = ttk.Combobox(ctrl, textvariable=self.selected_alg,
                                 values=self.algorithms, state='readonly', width=15)
        alg_combo.pack(side='left', padx=5)
        alg_combo.bind('<<ComboboxSelected>>', lambda e: self._compute_all_and_show())

        ttk.Label(ctrl, text="Grid Step:").pack(side='left', padx=(20,5))
        self.grid_step = tk.IntVar(value=20)
        ttk.Scale(ctrl, from_=1, to=50, variable=self.grid_step,
                  command=lambda e: self.show_flow()).pack(side='left', padx=5)
        self.step_label = ttk.Label(ctrl, text=str(self.grid_step.get()))
        self.step_label.pack(side='left')
        self.grid_step.trace_add("write", lambda *args: self.step_label.config(text=str(self.grid_step.get())))

        # Display options: computed trail/net and NPY-based
        self.show_trail_var = tk.BooleanVar(value=True)
        self.show_net_var   = tk.BooleanVar(value=False)
        self.show_npy_trail_var = tk.BooleanVar(value=False)
        self.show_npy_net_var   = tk.BooleanVar(value=False)
        self.show_net_error_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Show Computed Trail", variable=self.show_trail_var,
                       command=self.show_flow).pack(side='left', padx=(20,5))
        ttk.Checkbutton(ctrl, text="Show Computed Net", variable=self.show_net_var,
                       command=self.show_flow).pack(side='left', padx=5)
        ttk.Checkbutton(ctrl, text="Show NPY Trail", variable=self.show_npy_trail_var,
                       command=self.show_flow).pack(side='left', padx=5)
        ttk.Checkbutton(ctrl, text="Show NPY Net", variable=self.show_npy_net_var,
                       command=self.show_flow).pack(side='left', padx=5)
        ttk.Checkbutton(ctrl, text="Show Net Error", variable=self.show_net_error_var,
                        command=self.show_flow).pack(side='left', padx=5)

        # Canvas for display
        self.canvas = tk.Canvas(self, bg='#f0f0f0')
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        self.canvas.bind('<Configure>', lambda e: self.show_flow())

        # Internal state
        self.images = []
        self.flows_inc = []      # computed incremental flows
        self.net_flow = None     # computed net flow
        self.npy_inc = []        # loaded incremental dx/dy as npy per frame
        self.npy_net = None      # loaded or computed net from npy
        self.index = 0

    @staticmethod
    def _numeric_key(fn):
        name = os.path.splitext(fn)[0]
        m = re.search(r"(\d+)", name)
        return (0, int(m.group(1))) if m else (1, name.lower())

    def load_images(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        files = os.listdir(folder)
        imgs = sorted([f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))],
                      key=self._numeric_key)
        self.images = [cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                       for f in imgs]

        # Load .npy dx_/dy_ files for incremental displacement
        dx_files = sorted([f for f in files if f.startswith('dx_') and f.endswith('.npy')], key=self._numeric_key)
        dy_files = sorted([f for f in files if f.startswith('dy_') and f.endswith('.npy')], key=self._numeric_key)
        self.npy_inc = [None]
        for dx_fn, dy_fn in zip(dx_files, dy_files):
            dx = np.load(os.path.join(folder, dx_fn))
            dy = np.load(os.path.join(folder, dy_fn))
            self.npy_inc.append(np.dstack((dx, dy)))
        # NPY net will be derived dynamically from trail, not precomputed
        self.npy_net = None
        self.index = 0
        self._compute_all()
        self.show_flow()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_flow()

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_flow()

    def _compute_all(self):
        alg = self.selected_alg.get()
        # Computed incremental flows
        self.flows_inc = [None]
        for i in range(1, len(self.images)):
            f = self.compute_flow(self.images[i-1], self.images[i], alg)
            self.flows_inc.append(f)

    def _compute_all_and_show(self):
        if self.images:
            self._compute_all()
            self.show_flow()

    def compute_flow(self, img1, img2, method, **params):
        if method == 'Farneback':
            return cv2.calcOpticalFlowFarneback(
                img1, img2, None,
                pyr_scale  = params.get('pyr_scale', 0.5),
                levels     = params.get('levels', 3),
                winsize    = params.get('winsize', 15),
                iterations = params.get('iterations', 3),
                poly_n     = params.get('poly_n', 5),
                poly_sigma = params.get('poly_sigma', 1.2),
                flags      = params.get('flags', 0)
            )
        if method == 'TVL1' and self.tvl1:
            return self.tvl1.calc(img1, img2, None)
        if method == 'Lucas-Kanade':
            p0 = cv2.goodFeaturesToTrack(img1, maxCorners=200,
                                         qualityLevel=0.01, minDistance=7)
            if p0 is None:
                return None
            p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
            pts0 = p0[st.flatten()==1].reshape(-1,2)
            pts1 = p1[st.flatten()==1].reshape(-1,2)
            disp = pts1 - pts0
            return np.stack((pts0[:,0], pts0[:,1], disp[:,0], disp[:,1]), axis=1)
        if method == 'Speckle Tracking':
            h, w = img1.shape
            step = self.grid_step.get()
            pts = [(x,y) for y in range(0,h,step) for x in range(0,w,step)]
            pts1 = []
            half = 5
            tpl_h, tpl_w = 2*half+1, 2*half+1
            for x0,y0 in pts:
                tpl = img1[max(y0-half,0):y0+half+1, max(x0-half,0):x0+half+1]
                win = img2[max(y0-step,0):y0+step+1, max(x0-step,0):x0+step+1]
                if win.shape[0]<tpl_h or win.shape[1]<tpl_w:
                    pts1.append((x0,y0))
                    continue
                _,_,_,mx = cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))
                dx = mx[0] - (win.shape[1]//2 - half)
                dy = mx[1] - (win.shape[0]//2 - half)
                pts1.append((x0+dx, y0+dy))
            disp = np.array(pts1) - np.array(pts)
            return np.concatenate((np.array(pts), disp), axis=1)
        return None

    def show_flow(self):
        self.canvas.delete('all')
        if not self.images:
            return

        # Fit raw image to canvas
        img = self.images[self.index]
        h_img, w_img = img.shape
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(c_w/w_img, c_h/h_img)
        nw, nh = int(w_img*scale), int(h_img*scale)
        x_off, y_off = (c_w-nw)//2, (c_h-nh)//2
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh))
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(x_off, y_off, anchor='nw', image=self.photo)

        step = self.grid_step.get()

        # Computed trail
        if self.show_trail_var.get() and self.index>0:
            color='blue'
            if hasattr(self.flows_inc[1], 'ndim') and self.flows_inc[1].ndim==3:
                for y0 in range(0,h_img,step):
                    for x0 in range(0,w_img,step):
                        x_prev, y_prev = float(x0), float(y0)
                        for t in range(1, self.index+1):
                            f = self.flows_inc[t]
                            yi = int(np.clip(round(y_prev),0,h_img-1))
                            xi = int(np.clip(round(x_prev),0,w_img-1))
                            dx, dy = f[yi, xi, 0], f[yi, xi, 1]
                            x_next, y_next = x_prev+dx, y_prev+dy
                            x1,y1 = x_prev*scale+x_off, y_prev*scale+y_off
                            x2,y2 = x_next*scale+x_off, y_next*scale+y_off
                            self.canvas.create_line(x1,y1,x2,y2, fill=color)
                            x_prev, y_prev = x_next, y_next
            else:
                for x0, y0, dx, dy in self.flows_inc[self.index].astype(float):
                    x_prev, y_prev = x0, y0
                    for t in range(1, self.index+1):
                        f = self.flows_inc[t]
                        mask = (f[:,0]==y_prev)&(f[:,1]==x_prev)
                        if not mask.any(): break
                        idx = np.where(mask)[0][0]
                        dx_t, dy_t = f[idx,2], f[idx,3]
                        x_next, y_next = x_prev+dx_t, y_prev+dy_t
                        x1,y1 = x_prev*scale+x_off, y_prev*scale+y_off
                        x2,y2 = x_next*scale+x_off, y_next*scale+y_off
                        self.canvas.create_line(x1,y1,x2,y2, fill=color)
                        x_prev, y_prev = x_next, y_next

        # Computed net head‑to‑tail from image0 to final image
        if self.show_net_var.get() and len(self.flows_inc) > 1:
            color = 'red'
            n_frames = len(self.flows_inc)  # total frames = images count

            for y0 in range(0, h_img, step):
                for x0 in range(0, w_img, step):
                    x_prev, y_prev = float(x0), float(y0)

                    # integrate every incremental flow from frame1 → final
                    for t in range(1, n_frames):
                        f = self.flows_inc[t]
                        if f is None:
                            break

                        if hasattr(f, 'ndim') and f.ndim == 3:
                            yi = int(np.clip(round(y_prev), 0, h_img-1))
                            xi = int(np.clip(round(x_prev), 0, w_img-1))
                            dx, dy = f[yi, xi, 0], f[yi, xi, 1]
                        else:
                            # sparse case (Lucas‑Kanade)
                            pts0 = f[:, :2]
                            disps = f[:, 2:]
                            # find nearest point
                            dists = np.hypot(pts0[:,0]-x_prev, pts0[:,1]-y_prev)
                            idx = np.argmin(dists)
                            dx, dy = disps[idx]

                        x_prev += dx
                        y_prev += dy

                    # draw one net arrow from (x0,y0) to final (x_prev,y_prev)
                    x1 = x0*scale + x_off
                    y1 = y0*scale + y_off
                    x2 = x_prev*scale + x_off
                    y2 = y_prev*scale + y_off
                    self.canvas.create_line(x1, y1, x2, y2, fill=color)

        # NPY-based trail
        if self.show_npy_trail_var.get() and self.npy_inc and self.index > 0:
            color = 'green'
            for y0 in range(0, h_img, step):
                for x0 in range(0, w_img, step):
                    x_prev, y_prev = float(x0), float(y0)
                    # walk through each incremental npy flow up to current frame
                    for t in range(1, self.index+1):
                        f = self.npy_inc[t]
                        if f is None:
                            break
                        yi = int(np.clip(round(y_prev), 0, h_img-1))
                        xi = int(np.clip(round(x_prev), 0, w_img-1))
                        # invert both x and y displacements
                        dx, dy = -f[yi, xi, 0], -f[yi, xi, 1]
                        x_next, y_next = x_prev + dx, y_prev + dy
                        x1 = x_prev*scale + x_off
                        y1 = y_prev*scale + y_off
                        x2 = x_next*scale + x_off
                        y2 = y_next*scale + y_off
                        self.canvas.create_line(x1, y1, x2, y2, fill=color)
                        x_prev, y_prev = x_next, y_next


        # NPY‑based net + net‑error metrics
        if (self.show_npy_net_var.get() or self.show_net_error_var.get()) and len(self.npy_inc) > 1:
            total_err = 0.0
            total_pct = 0.0
            count     = 0
            color     = 'lime'
            n_frames  = len(self.npy_inc)

            for y0 in range(0, h_img, step):
                for x0 in range(0, w_img, step):
                    # actual NPY net endpoint
                    xa, ya = float(x0), float(y0)
                    for t in range(1, n_frames-1):
                        f = self.npy_inc[t]
                        if f is None: break
                        yi = int(np.clip(round(ya), 0, h_img-1))
                        xi = int(np.clip(round(xa), 0, w_img-1))
                        dx, dy = -f[yi, xi, 0], -f[yi, xi, 1]
                        xa += dx; ya += dy

                    # computed net endpoint
                    xc, yc = float(x0), float(y0)
                    for t in range(1, len(self.flows_inc)):
                        f = self.flows_inc[t]
                        if f is None: break
                        if hasattr(f, 'ndim') and f.ndim == 3:
                            yi = int(np.clip(round(yc), 0, h_img-1))
                            xi = int(np.clip(round(xc), 0, w_img-1))
                            dx, dy = f[yi, xi, 0], f[yi, xi, 1]
                        else:
                            pts0, disp = f[:, :2], f[:, 2:]
                            dists = np.hypot(pts0[:,0]-xc, pts0[:,1]-yc)
                            idx = np.argmin(dists)
                            dx, dy = disp[idx]
                        xc += dx; yc += dy

                    # draw NPY net if requested
                    x1, y1 = x0*scale + x_off, y0*scale + y_off
                    x2, y2 = (x0+ (xa - x0))*scale + x_off, (y0+ (ya - y0))*scale + y_off
                    if self.show_npy_net_var.get():
                        self.canvas.create_line(x1, y1, x2, y2, fill=color)

                    # draw error line and accumulate metrics
                    if self.show_net_error_var.get():
                        err = np.hypot(xc - xa, yc - ya)
                        total_err += err
                        mag = max(np.hypot(ya - y0, xa - x0), 1e-6)
                        total_pct += (err/mag)*100.0
                        count += 1
                        x3, y3 = xc*scale + x_off, yc*scale + y_off
                        self.canvas.create_line(x2, y2, x3, y3, fill='pink')

            # after looping over all grid points, display the summary
            if self.show_net_error_var.get() and count > 0:
                avg_err = total_err / count
                avg_pct = total_pct / count
                cum_err = total_err
                self.canvas.create_text(
                    10, 10, anchor='nw',
                    text=(f"Avg error: {avg_err:.2f}px\nCumulative error: {cum_err:.2f}px\n" 
                          f"Avg % error: {avg_pct:.1f}%\nSamples: {count}"),
                    fill='black', font=('TkDefaultFont', 10, 'bold')
                )

    def _load_folder_data(self, folder):
        """
        Load grayscale images and their dx_/dy_ .npy flows from a single folder.
        Returns (images_list, npy_inc_list).
        """
        files = os.listdir(folder)
        # images
        imgs = sorted(
            [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))],
            key=self._numeric_key
        )
        images = [cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                    for f in imgs]

        # .npy flows
        dxs = sorted([f for f in files if f.startswith('dx_') and f.endswith('.npy')],
                        key=self._numeric_key)
        dys = sorted([f for f in files if f.startswith('dy_') and f.endswith('.npy')],
                        key=self._numeric_key)
        npy_inc = [None]
        for dx_fn, dy_fn in zip(dxs, dys):
            dx = np.load(os.path.join(folder, dx_fn))
            dy = np.load(os.path.join(folder, dy_fn))
            npy_inc.append(np.dstack((dx, dy)))

        return images, npy_inc

    @staticmethod
    def _parameter_grid(param_defs):
        """Yield every combination of params from a dict of lists."""
        keys = list(param_defs.keys())
        for vals in itertools.product(*(param_defs[k] for k in keys)):
            yield dict(zip(keys, vals))

    @staticmethod
    def _mean_endpoint_error(computed_flow, gt_flow):
        """
        Compute mean endpoint error between computed_flow (H×W×2) and inverted gt_flow (because
        your display code does -gt). Excludes any pixels whose computed endpoint falls
        outside the image extents.
        """
        if computed_flow is None or gt_flow is None:
            return np.nan

        H, W, _ = computed_flow.shape
        # flatten flows
        c = computed_flow.reshape(-1, 2)      # computed dx,dy
        g = gt_flow.reshape(-1, 2)            # ground truth dx,dy (we'll invert below)

        # build pixel coordinates
        ys, xs = np.indices((H, W))
        ys = ys.reshape(-1)
        xs = xs.reshape(-1)

        # compute final positions
        final_x = xs + c[:,0]
        final_y = ys + c[:,1]

        # keep only those inside [0, W-1] and [0, H-1]
        mask = (
            (final_x >= 0) & (final_x < W) &
            (final_y >= 0) & (final_y < H)
        )
        if not np.any(mask):
            return np.nan

        # endpoint error = L2 between computed (dx,dy) and inverted gt (-gt_dx, -gt_dy)
        diffs = c[mask] - (-g[mask])
        errs  = np.linalg.norm(diffs, axis=1)
        return np.mean(errs)


    def grid_search_multi(self, root_folder, algorithm, param_defs, csv_path):
        """
        For each parameter combo in param_defs:
            • run every subfolder under root_folder as an instance
            • compute per‑pair endpoint errors vs. that subfolder’s .npy
            • average within each subfolder, then average across subfolders
            • write one row/combo → mean_error into csv_path
        """
        # find subfolders
        subs = [os.path.join(root_folder, d)
                for d in os.listdir(root_folder)
                if os.path.isdir(os.path.join(root_folder, d))]

        # prepare CSV
        cols = ['algorithm'] + list(param_defs.keys()) + ['mean_error']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()

            for params in self._parameter_grid(param_defs):
                sub_means = []
                for sub in subs:
                    imgs, npy = self._load_folder_data(sub)
                    errs = []
                    for i in range(1, len(imgs)):
                        flow = self.compute_flow(imgs[i-1], imgs[i], algorithm, **params)
                        errs.append(self._mean_endpoint_error(flow, npy[i]))
                    sub_means.append(np.nanmean(errs))

                overall = np.nanmean(sub_means)
                row = {'algorithm': algorithm, **params, 'mean_error': overall}
                w.writerow(row)

        print(f"[+] grid search done → {csv_path}")


# ——— USAGE ———
if __name__ == "__main__":
    root = r"oct_dataset"
    farneback_grid = {
        'pyr_scale':  [0.3, 0.5, 0.7],
        'levels':     [1,   3,   5],
        'winsize':    [9,  15,  21],
        'iterations': [1,   3,   5],
        'poly_n':     [5,   7],
        'poly_sigma': [1.1, 1.5]
    }

    page = OpticalFlowPage(parent=None)
    page.grid_search_multi(
        root_folder=root,
        algorithm='Farneback',
        param_defs=farneback_grid,
        csv_path='farneback_results.csv'
    )


