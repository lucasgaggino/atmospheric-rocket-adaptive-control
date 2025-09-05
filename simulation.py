import numpy as np
from scipy.integrate import solve_ivp

class Simulation:
    def __init__(self, rocket, atmos, thrust_fraction=0.75, thrust_angle=0.0, t_span=(0, 60), initial_state=None):
        self.rocket = rocket
        self.atmos = atmos
        self.thrust_fraction = thrust_fraction
        self.thrust_angle = thrust_angle
        self.t_span = t_span
        if initial_state is None:
            self.initial_state = np.array([0, 0, 0, 0, np.pi/2, 0])  # x, y, vx, vy, theta, omega
        else:
            self.initial_state = initial_state

    def ode_func(self, t, state):
        return self.rocket.get_derivatives(state, self.thrust_fraction, self.thrust_angle, self.atmos)

    def run(self):
        sol = solve_ivp(self.ode_func, self.t_span, self.initial_state, method='RK45', rtol=1e-6)
        return sol
