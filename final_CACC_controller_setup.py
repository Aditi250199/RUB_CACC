# Cooperative Adapative Cruise Control
#
# Controller based on CACC of the paper
# "Combined Longitudinal and Lateral Control of Car-Like Vehicle Platooning With Extended Look-Ahead"
# adapted with abs(v1) instead of v1 for constant velocity headway, and removed poles


import numpy as np
import time


class CACC:
    def __init__(self, k1, k2, headway=0.2, time_gap=0.2) -> None:
        self.k1 = k1
        self.k2 = k2
        self.time_gap = time_gap  # seconds
        self.headway = headway  # meters

    def get_actuating_sig(self, lead, ego):
        # get states
        v0 = lead.vel.linear
        v1 = ego.vel.linear
        theta0 = ego.pos.azimuth - lead.aod

        # calculate error
        ex = ego.pos.dist * np.cos(ego.pos.azimuth) - self.headway - self.time_gap * np.abs(v0)
        ey = ego.pos.dist * np.sin(ego.pos.azimuth)

        # calculate control
        a = (v0 * np.cos(theta0) - v1 + self.k1 * ex) / self.time_gap
        omega = (v0 * np.sin(theta0) + self.k2 * ey) / (self.time_gap * np.abs(v0) + self.headway)

        return a, omega


class extended_CACC:
    def __init__(self, k1, k2, headway=0.2, time_gap=0.2) -> None:
        self.k1 = k1
        self.k2 = k2
        self.time_gap = time_gap  # seconds
        self.headway = headway  # meters
        self.last_kappa0 = 0.0
        self.last_timestamp = time.time()
        self.first_call = True  # flag to detect first run
        self.omega_smoothed = None
        self.smooth_weight = 0.4
        self.set_smooth = False
        self.last_time = time.time()

    def get_actuating_sig(self, lead, ego):

        # get states
        v0 = lead.vel.linear
        omega0 = lead.vel.angular

        if self.set_smooth is True:
            if self.omega_smoothed == None:
                self.omega_smoothed = omega0
            else:
                self.omega_smoothed = omega0 * self.smooth_weight + (1 - self.smooth_weight) * self.omega_smoothed
                omega0 = self.omega_smoothed

    
        v1 = ego.vel.linear
        theta0 = ego.pos.azimuth - lead.aod
        x0 = ego.pos.dist * np.cos(ego.pos.azimuth)
        y0 = ego.pos.dist * np.sin(ego.pos.azimuth)

        if v0 == 0:
            kappa0 = 0
        else:
            kappa0 = omega0 / v0  # curvature of lead vehicle

        if self.first_call:
            self.last_kappa0 = kappa0
            self.last_timestamp = time.time()
            self.first_call = False

        current_time = time.time()
        dt = current_time - self.last_timestamp
        if dt == 0:
            dt = 1e-5  # avoid division by zero
        self.last_timestamp = current_time
        dkappa0 = (kappa0 - self.last_kappa0) / dt
        self.last_kappa0 = kappa0

        # print("v0:", v0, "omega0:", omega0, "v1:", v1)
        # print("theta0:", theta0, "x0:", x0, "y0:", y0)
        # print("kappa0:", kappa0, "dkappa0:", dkappa0)

        r1 = self.headway
        h1 = self.time_gap
        h1v1 = h1 * np.abs(v1)
        hvr2 = (h1v1 + r1) ** 2
        kappa02 = kappa0 ** 2
        sqrt_k = np.sqrt(kappa02 * hvr2 + 1)  # substition variable

        # calculate lookahead point
        mag_s0 = (kappa0 * hvr2) / (1 + sqrt_k)
        sx0 = mag_s0 * np.sin(theta0)
        sy0 = - mag_s0 * np.cos(theta0)

        # calculate position error
        ex = x0 + sx0 - r1 - h1v1
        ey = y0 + sy0

        alpha1 = np.arctan(kappa0 * (r1 + h1v1))
        sin_alpha1 = np.sin(alpha1)
        inv_cos_alpha1 = sqrt_k

        inv_mu1 = 1 / ((1 - sin_alpha1 * np.sin(theta0)) * h1 * (r1 + h1v1))

        sk1 = hvr2 / (sqrt_k * (sqrt_k + 1))

        sa1 = h1 * sin_alpha1
        beta11x = mag_s0 * omega0 * np.cos(theta0) + sk1 * dkappa0 * np.sin(theta0) + (1 - inv_cos_alpha1) * np.cos(
            theta0) * v0
        beta11y = v1 * np.tan(alpha1) + mag_s0 * omega0 * np.sin(theta0) - sk1 * dkappa0 * np.cos(theta0) + (
                    1 - inv_cos_alpha1) * np.sin(theta0) * v0
        beta11 = np.array([[beta11x], [beta11y]])


        # calculate control states
        z11 = ex
        z21 = ey
        z31 = v0 * np.cos(theta0) - v1 * np.cos(alpha1)
        z41 = v0 * np.sin(theta0) - v1 * np.sin(alpha1)

        inv_Gamma12_1 = inv_mu1 * np.array([[r1 + h1v1, 0],
                                            [-sa1 * np.cos(theta0), h1 - sa1 * np.sin(theta0)]])


        vec = np.array([[self.k1 * z11], [self.k2 * z21]]) + inv_cos_alpha1 * np.array([[z31], [z41]]) + beta11


        out = np.matmul(inv_Gamma12_1, vec)
        a = out[0]
        omega = out[1]
        return float(a), float(omega)



