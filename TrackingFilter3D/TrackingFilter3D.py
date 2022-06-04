from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
import numpy as np
import logging
import matplotlib.pyplot as plt
import os


class TennisBallTracker3D:
    def __init__(self, frame_rate, filter='EKF'):
        self.delta_t = 1.0 / frame_rate
        self.adjusted_dt = 1.0 / frame_rate

        self.u = np.array([9.81])

        self.filter_type = filter
        if filter == 'EKF':
            self.filter = ExtendedKalmanFilter(dim_x=6, dim_z=3, dim_u=1)
        elif filter == 'UKF':
            points = JulierSigmaPoints(6)
            self.filter = UnscentedKalmanFilter(6, 3, self.delta_t, TennisBallTracker3D.hx, TennisBallTracker3D.fx, points)
        self.filter.x = np.array([0.0, 4.25, 1.0, 0.0, 0.0, 0.0])
        self.filter.R = np.identity(3) * 0.01
        self.filter.Q = np.identity(6)
        self.filter.Q[:3, :3] *= 1.0
        self.filter.Q[3:, 3:] *= 5.0

        self.filter.P[:3, :3] *= 2.0
        self.filter.P[3:, 3:] *= 0.5

        self.state_history = None
        self.confidence_low_history = None
        self.confidence_high_history = None
        self.reprojection_errors = None
        self.reprojection = None

        self.updates_skipped = 0

    def A_B_t(self):
        delta_t = self.adjusted_dt
        A = np.array([
            [1.0, 0.0, 0.0, delta_t, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, delta_t, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, delta_t],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        B = np.array([[0.0, 0.0, -0.5 * delta_t**2, 0.0, 0.0, -delta_t]]).T
        self.updates_skipped = 0
        return A, B

    def HJacobian_at(x):
        p = x[:3]
        p_x, p_y, p_z = p
        p_norm_cubed = np.linalg.norm(p)**3
        H = np.array([
            [p_y**2 + p_z**2, -p_x*p_y, -p_x*p_z],
            [-p_x*p_y, p_x**2 + p_z**2, -p_y*p_z],
            [-p_x*p_z, -p_y*p_z, p_x**2 + p_y**2]
        ]) / p_norm_cubed
        H = np.block([
            [H, np.zeros((3, 3))]
        ])
        return H

    def fx(x, dt):
        px, py, pz = x[:3]
        vx, vy, vz = x[3:]
        x_tp1 = np.array([
            px + dt * vx,
            py + dt * vy,
            pz + dt * vz - 0.5 * 9.81 * dt**2,
            vx,
            vy,
            vz - 9.81 * dt**2
        ])
        return x_tp1

    def hx(x):
        p = x[:3]
        p_norm = np.linalg.norm(p)
        if p_norm == 0.0:
            return np.zeros(p.shape[0])
        return p / p_norm

    def update(self, M, p_2d):
        # decompose the M matrix into K R t
        A = M[:, :3]
        a_1 = A[0]
        a_2 = A[1]
        a_3 = A[2]
        b = M[:, 3]
        rho = 1 / np.linalg.norm(a_3)
        c_x = rho**2 * (a_1 @ a_3)
        c_y = rho**2 * (a_2 @ a_3)
        a_13_cross = np.cross(a_1, a_3)
        a_23_cross = np.cross(a_2, a_3)
        theta = np.arccos(-(a_13_cross @ a_23_cross) / (np.linalg.norm(a_13_cross) * np.linalg.norm(a_23_cross)))
        alpha = rho**2 * np.linalg.norm(a_13_cross) * np.sin(theta)
        beta = rho**2 * np.linalg.norm(a_23_cross) * np.sin(theta)

        r1 = a_23_cross / np.linalg.norm(a_23_cross)
        r3 = rho * a_3
        r2 = np.cross(r3, r1)
        R = np.row_stack((r1, r2, r3))
        K = np.array([
            [alpha, -alpha * (1/np.tan(theta)), c_x],
            [0.0, beta / np.sin(theta), c_y],
            [0.0, 0.0, 1.0],
        ])
        T = rho * np.linalg.inv(K) @ b

        p_2d_homogeneous = np.concatenate((p_2d, np.ones(1)))
        bearing_vec = np.linalg.inv(K @ R) @ (p_2d_homogeneous - T)
        bearing_vec /= np.linalg.norm(bearing_vec)

        self.adjusted_dt = (self.updates_skipped + 1) * self.delta_t
        self.updates_skipped = 0
        A_t, B_t = self.A_B_t()
        self.filter.F = A_t
        self.filter.B = B_t
        self.filter.dt = self.adjusted_dt

        logging.debug(f'Bearing vector = {bearing_vec}')
        logging.debug(f'Predicted bearing vector = {TennisBallTracker3D.hx(self.filter.x)}')
        if self.filter_type == 'EKF':
            self.filter.update(bearing_vec, TennisBallTracker3D.HJacobian_at, TennisBallTracker3D.hx)
        elif self.filter_type == 'UKF':
            self.filter.update(bearing_vec)
        logging.debug(f'Filter state = {self.filter.x}')

        std_bounds = np.sqrt(np.diag(self.filter.P))
        if self.state_history is None:
            self.state_history = self.filter.x
            self.confidence_low_history = np.array([self.filter.x - 2.0 * std_bounds])
            self.confidence_high_history = np.array([self.filter.x + 2.0 * std_bounds])
        else:
            self.state_history = np.row_stack((self.state_history, self.filter.x))
            self.confidence_low_history = np.row_stack((self.confidence_low_history, np.array([self.filter.x - 2.0 * std_bounds])))
            self.confidence_high_history = np.row_stack((self.confidence_high_history, np.array([self.filter.x + 2.0 * std_bounds])))

        # calculate reprojection error
        tennis_ball_est_3d = self.filter.x[:3]
        self.reprojection = M @ np.concatenate((tennis_ball_est_3d, np.ones(1)))
        self.reprojection = (self.reprojection / self.reprojection[-1])[:2]
        logging.debug(f'2D position prediction = {self.reprojection}')
        reprojection_err = np.linalg.norm(p_2d - self.reprojection)
        if self.reprojection_errors is None:
            self.reprojection_errors = np.array([reprojection_err])
        else:
            self.reprojection_errors = np.append(self.reprojection_errors, reprojection_err)
        logging.debug(f'Reprojection error = {reprojection_err}')

        if self.filter_type == 'EKF':
            self.filter.predict(u=self.u)
        elif self.filter_type == 'UKF':
            self.filter.predict()
        
        logging.debug(f'Filter prediction = {self.filter.x_prior}')

    def skip_update(self):
        self.updates_skipped += 1

    def create_plots(self, path):
        times = np.arange(self.state_history.shape[0]) * self.delta_t

        figure = plt.figure(figsize=(15, 10), constrained_layout=True)
        labels = ['p_t_x', 'p_t_y', 'p_t_z']
        mosaic = [[label] for label in labels]
        ax_dict = figure.subplot_mosaic(mosaic)

        for i, label in enumerate(labels):
            ax_dict[label].plot(times, self.state_history[:, i], label=f'Predicted {label}')
            ax_dict[label].fill_between(times, self.confidence_low_history[:, i], self.confidence_high_history[:, i], alpha=0.25)
            ax_dict[label].set_xlabel('Time')
            ax_dict[label].set_ylabel(f'{label}')
            ax_dict[label].set_title(f'Predicted {label} vs. Time')
        
        figure.suptitle('Predicted 3D Position of Tennis Ball Throughout Time')
        plt.savefig(os.path.join(path, 'Position3D.png'))

        figure = plt.figure(figsize=(15, 10), constrained_layout=True)
        labels = ['v_t_x', 'v_t_y', 'v_t_z']
        mosaic = [[label] for label in labels]
        ax_dict = figure.subplot_mosaic(mosaic)

        for i, label in enumerate(labels):
            ax_dict[label].plot(times, self.state_history[:, i+3], label=f'Predicted {label}')
            ax_dict[label].fill_between(times, self.confidence_low_history[:, i+3], self.confidence_high_history[:, i+3], alpha=0.25)
            ax_dict[label].set_xlabel('Time')
            ax_dict[label].set_ylabel(f'{label}')
            ax_dict[label].set_title(f'Predicted {label} vs. Time')

        figure.suptitle('Predicted 3D Velocity of Tennis Ball Throughout Time')
        plt.savefig(os.path.join(path, 'Velocity3D.png'))

        figure = plt.figure(figsize=(15, 10), constrained_layout=True)
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.state_history[:, 0], self.state_history[:, 1], self.state_history[:, 2])
        ax.set_xlabel('p_x')
        ax.set_ylabel('p_y')
        ax.set_zlabel('p_z')
        ax.set_title('3D Position of Tennis Ball')
        plt.savefig(os.path.join(path, 'Position3DVisualization.png'))

        plt.figure(figsize=(15, 10), constrained_layout=True)
        plt.plot(times, self.reprojection_errors)
        plt.xlabel('Time')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error of 3D Position of Tennis Ball Throughout Time')
        plt.savefig(os.path.join(path, 'ReprojectionErrors.png'))
