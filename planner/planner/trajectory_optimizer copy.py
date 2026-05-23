#!/usr/bin/env python3

import os
import csv
import numpy as np

import rclpy
from rclpy.node import Node


class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        # ---- Parameters --------------------------------------------------
        self.declare_parameter('map_name', '')
        self.declare_parameter('input_csv', 'centerline.csv')
        self.declare_parameter('output_csv', 'global_waypoints.csv')
        self.declare_parameter('safety_margin', 0.20)   # [m]   clearance from each wall
        self.declare_parameter('v_max',         6.0)    # [m/s] vehicle speed cap
        self.declare_parameter('a_lat_max',     6.0)    # [m/s^2] lateral grip limit
        self.declare_parameter('a_long_max',    4.0)    # [m/s^2] longitudinal accel limit
        self.declare_parameter('target_ds',     0.25)   # [m]   uniform ds for QP input/output

        map_name      = self.get_parameter('map_name').value
        input_csv     = self.get_parameter('input_csv').value
        output_csv    = self.get_parameter('output_csv').value
        safety_margin = self.get_parameter('safety_margin').value
        v_max         = self.get_parameter('v_max').value
        a_lat_max     = self.get_parameter('a_lat_max').value
        a_long_max    = self.get_parameter('a_long_max').value
        target_ds     = self.get_parameter('target_ds').value

        if not map_name:
            self.get_logger().error('[TrajectoryOptimizer] map_name parameter is required!')
            return

        # ---- I/O paths ---------------------------------------------------
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        map_dir  = os.path.join(pkg_root, 'stack_master', 'maps', map_name)
        in_path  = os.path.join(map_dir, input_csv)
        out_path = os.path.join(map_dir, output_csv)

        if not os.path.exists(in_path):
            self.get_logger().error(f'[TrajectoryOptimizer] input not found: {in_path}')
            return

        # ---- Load + optimize + save -------------------------------------
        self.get_logger().info(f'[TrajectoryOptimizer] loading: {in_path}')
        x_c, y_c, w_r, w_l = self._load_centerline(in_path)

        self.get_logger().info(
            f'[TrajectoryOptimizer] optimizing on {len(x_c)} centerline points '
            f'(margin={safety_margin}, v_max={v_max}, a_lat={a_lat_max}, '
            f'a_long={a_long_max}, target_ds={target_ds})'
        )
        x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new = self._optimize(
            x_c, y_c, w_r, w_l,
            safety_margin=safety_margin,
            v_max=v_max,
            a_lat_max=a_lat_max,
            a_long_max=a_long_max,
            target_ds=target_ds,
        )

        self._save_global_waypoints(out_path, x_opt, y_opt, w_r_new, w_l_new, psi, kappa, vx)
        self.get_logger().info(
            f'[TrajectoryOptimizer] saved {len(x_opt)} pts → {out_path} '
            f'(v_min={vx.min():.2f}, v_max={vx.max():.2f} m/s, '
            f'|kappa|max={np.max(np.abs(kappa)):.3f})'
        )

    # ======================================================================
    #                       STUDENT IMPLEMENTATION
    # ======================================================================
    @staticmethod
    def _optimize(x_c, y_c, w_r, w_l,
                  safety_margin, v_max, a_lat_max, a_long_max, target_ds):
        """
        Minimum-curvature trajectory optimization using trajectory_planning_helpers.

        Parameters
        ----------
        x_c, y_c   : (N,) centerline coordinates (closed loop, no duplicate end)
        w_r, w_l   : (N,) track half-widths to the right / left walls
        safety_margin : [m] keep this far from each wall
        v_max, a_lat_max, a_long_max : vehicle limits
        target_ds  : [m] desired arc-length spacing for the optimized output

        Returns
        -------
        x_opt, y_opt, psi, kappa, vx : (M,) arrays for the optimized raceline
        w_r_new, w_l_new             : (M,) remaining clearance to the original walls
        """
        import trajectory_planning_helpers as tph

        # Step 1. Apply safety margin; clamp to a minimum feasible gap (0.05 m)
        w_r_s = np.maximum(w_r - safety_margin, 0.05)
        w_l_s = np.maximum(w_l - safety_margin, 0.05)
        reftrack = np.column_stack([x_c, y_c, w_r_s, w_l_s])   # (N, 4)

        # Step 2. Resample to uniform arc-length spacing
        #   tph QP requires evenly-spaced support points
        reftrack = tph.interp_track.interp_track(
            track=reftrack, stepsize=target_ds)
        N = len(reftrack)

        # Step 3. Fit cubic splines → unit normals + QP system matrix M
        #   calc_splines detects a closed path automatically when first==last point
        path_cl    = np.vstack([reftrack[:, :2], reftrack[0:1, :2]])   # (N+1, 2)
        el_lengths = np.hypot(np.diff(path_cl[:, 0]), np.diff(path_cl[:, 1]))
        _, _, M, normvec = tph.calc_splines.calc_splines(
            path=path_cl, el_lengths=el_lengths, use_dist_scaling=False)

        # Step 4. Solve convex minimum-curvature QP (Heilmeier et al. 2019)
        #   kappa_bound: max curvature from vehicle kinematics = tan(max_steer)/wheelbase
        kappa_bound = np.tan(0.4) / 0.33          # ≈ 1.21 rad/m
        alpha, curv_err = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack, normvectors=normvec, A=M,
            kappa_bound=kappa_bound, w_veh=0.3, closed=True, print_debug=True)
        print(f'[tph-opt] alpha min={alpha.min():.3f}  max={alpha.max():.3f}  '
              f'std={alpha.std():.3f}  |>0.1m|={(np.abs(alpha) > 0.1).sum()}/{N}  '
              f'curv_err={curv_err:.4f}')

        # Smooth alpha to avoid arc-length collapse on the inside of tight curves.
        from scipy.ndimage import gaussian_filter1d
        alpha = gaussian_filter1d(alpha, sigma=3.0, mode='wrap')
        alpha = np.clip(alpha, -(reftrack[:, 2] - 0.05), reftrack[:, 3] - 0.05)

        # Step 5. Reconstruct the raceline at uniform spacing.
        raceline_out    = tph.create_raceline.create_raceline(
            refline=reftrack[:, :2], normvectors=normvec,
            alpha=alpha, stepsize_interp=target_ds)
        raceline_interp = raceline_out[0]          # (M, 2) [x, y]

        # Step 6. Remaining wall clearances
        #   alpha > 0 = shifted in +normvec direction (LEFT of travel)
        #   → right clearance increases, left clearance decreases
        w_r_opt = np.maximum(reftrack[:, 2] + alpha, 0.0)
        w_l_opt = np.maximum(reftrack[:, 3] - alpha, 0.0)
        s_ref   = np.linspace(0.0, 1.0, N,                    endpoint=False)
        s_rl    = np.linspace(0.0, 1.0, len(raceline_interp), endpoint=False)
        w_r_rl  = np.interp(s_rl, s_ref, w_r_opt)
        w_l_rl  = np.interp(s_rl, s_ref, w_l_opt)

        # Resample to strictly uniform spacing so centered-difference kappa is stable
        x_opt, y_opt, w_r_new, w_l_new = TrajectoryOptimizer._resample_uniform(
            raceline_interp[:, 0], raceline_interp[:, 1], w_r_rl, w_l_rl, target_ds)

        # Step 7. Heading and curvature via centered differences; clip to vehicle limit
        # so numerical noise at non-uniform junctions doesn't drag down the speed profile.
        psi, kappa = TrajectoryOptimizer._geom(x_opt, y_opt)
        kappa = np.clip(kappa, -kappa_bound, kappa_bound)

        # Step 8. Speed profile: cornering limit → forward accel → backward braking
        vx = TrajectoryOptimizer._speed_profile(
            x_opt, y_opt, kappa, v_max, a_lat_max, a_long_max)

        return x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new

    # ======================================================================
    #                            HELPERS
    # ======================================================================
    @staticmethod
    def _load_centerline(path):
        """centerline.csv → (x, y, w_tr_right, w_tr_left) numpy arrays."""
        xs, ys, wrs, wls = [], [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row['x_m']))
                ys.append(float(row['y_m']))
                wrs.append(float(row['w_tr_right_m']))
                wls.append(float(row['w_tr_left_m']))
        x = np.asarray(xs); y = np.asarray(ys)
        wr = np.asarray(wrs); wl = np.asarray(wls)
        # drop duplicate closing point if present
        if len(x) > 1 and np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-3:
            x, y, wr, wl = x[:-1], y[:-1], wr[:-1], wl[:-1]
        return x, y, wr, wl

    @staticmethod
    def _resample_uniform(x, y, w_r, w_l, target_ds):
        """Linear-interp resample of a closed loop onto uniform arc-length spacing."""
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        L = s[-1]
        N_new = max(20, int(round(L / target_ds)))
        s_new = np.linspace(0.0, L, N_new, endpoint=False)
        x_p  = np.concatenate((x,   [x[0]]))
        y_p  = np.concatenate((y,   [y[0]]))
        wr_p = np.concatenate((w_r, [w_r[0]]))
        wl_p = np.concatenate((w_l, [w_l[0]]))
        return (np.interp(s_new, s, x_p),
                np.interp(s_new, s, y_p),
                np.interp(s_new, s, wr_p),
                np.interp(s_new, s, wl_p))

    @staticmethod
    def _geom(x, y):
        """Heading psi and signed curvature kappa via centered differences (closed loop)."""
        dx  = (np.roll(x, -1) - np.roll(x, 1)) * 0.5
        dy  = (np.roll(y, -1) - np.roll(y, 1)) * 0.5
        ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
        psi = np.arctan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        denom[denom < 1e-9] = 1e-9
        kappa = (dx * ddy - dy * ddx) / denom
        return psi, kappa

    @staticmethod
    def _speed_profile(x, y, kappa, v_max, a_lat_max, a_long_max):
        """Point-mass speed profile: cornering limit + fwd/bwd accel smoothing."""
        N = len(x)
        ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        ds[ds < 1e-6] = 1e-6
        v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6)))
        # backward pass: braking limit
        for _ in range(2):
            for i in range(N):
                j = (i - 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[j])
                v[j] = min(v[j], v_cap)
        # forward pass: acceleration limit
        for _ in range(2):
            for i in range(N):
                j = (i + 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[i])
                v[j] = min(v[j], v_cap)
        return v

    @staticmethod
    def _save_global_waypoints(path, x, y, w_r, w_l, psi, kappa, vx):
        header = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m',
                  'psi_rad', 'kappa_radpm', 'vx_mps']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(x)):
                w.writerow([f'{x[i]:.6f}', f'{y[i]:.6f}',
                            f'{w_r[i]:.4f}', f'{w_l[i]:.4f}',
                            f'{psi[i]:.6f}', f'{kappa[i]:.6f}',
                            f'{vx[i]:.4f}'])


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()