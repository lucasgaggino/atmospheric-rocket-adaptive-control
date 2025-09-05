import numpy as np

class Rocket:
    def __init__(self, mass=1000, inertia=10000, max_thrust=15000, engine_gimbal_range=15, engine_distance=5.0, g=9.81):
        self.mass = mass
        self.inertia = inertia
        self.max_thrust = max_thrust
        self.engine_gimbal_range = np.deg2rad(engine_gimbal_range)
        self.engine_distance = engine_distance
        self.g = g

    def get_derivatives(self, state, thrust_fraction, thrust_angle, atmos):
        x, y, vx, vy, theta, omega = state
        thrust_fraction = np.clip(thrust_fraction, 0.5, 1.0)
        thrust_angle = np.clip(thrust_angle, -self.engine_gimbal_range, self.engine_gimbal_range)

        thrust_dir = theta + thrust_angle
        thrust_mag = self.max_thrust * thrust_fraction
        thrust_force = thrust_mag * np.array([np.cos(thrust_dir), np.sin(thrust_dir)])

        gravity_force = np.array([0, -self.mass * self.g])

        velocity = np.array([vx, vy])
        aero_force, aero_torque = atmos.get_forces(y, velocity, theta)

        total_force = thrust_force + gravity_force + aero_force
        accel = total_force / self.mass

        thrust_torque = thrust_mag * self.engine_distance * np.sin(thrust_angle)
        total_torque = thrust_torque + aero_torque
        alpha = total_torque / self.inertia

        dstates = np.array([vx, vy, accel[0], accel[1], omega, alpha])
        return dstates
