import numpy as np

class Atmosphere:
    def __init__(self, rho0=1.225, scale_height=8400, Cd=0.5, area=1.0, stability_factor=1.0, length=10.0):
        self.rho0 = rho0
        self.scale_height = scale_height
        self.Cd = Cd
        self.area = area
        self.stability_factor = stability_factor
        self.length = length

    def get_forces(self, height, velocity, theta):
        v_mag = np.linalg.norm(velocity)
        if v_mag < 1e-6:
            return np.zeros(2), 0.0

        unit_v = velocity / v_mag
        rho = self.rho0 * np.exp(-max(0, height) / self.scale_height)
        drag_mag = 0.5 * rho * v_mag**2 * self.Cd * self.area
        drag = -drag_mag * unit_v

        vel_angle = np.arctan2(velocity[1], velocity[0])
        alpha = theta - vel_angle

        normal_mag = 0.5 * rho * v_mag**2 * self.area * 2.0 * alpha  # Simple model Cn = 2 * alpha

        normal_dir = np.array([-unit_v[1], unit_v[0]])  # Perpendicular to velocity, assuming convention
        normal_force = normal_mag * normal_dir

        q = 0.5 * rho * v_mag**2
        torque = -self.stability_factor * q * self.area * self.length * alpha

        total_force = drag + normal_force
        return total_force, torque
