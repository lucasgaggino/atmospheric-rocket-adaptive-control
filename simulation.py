import numpy as np
from scipy.integrate import solve_ivp
from rocket import Rocket
from atmosphere import Atmosphere
from avionics import Avionics


class Simulation:
    def __init__(self, rocket, atmos, avionics, t_span=(0, 60), initial_state=None):
        self.rocket:Rocket = rocket
        self.atmos:Atmosphere = atmos
        self.avionics:Avionics = avionics
        self.t_span:tuple[float, float] = t_span
        if initial_state is None:
            self.initial_state = np.array([0, 0, 0, 0, np.pi/2, 0])  # x, y, vx, vy, theta, omega
        else:
            self.initial_state = initial_state

    def ode_func(self, t, state):
        # Estimate dt based on solver's internal timestep (rough approximation)
        dt = 0.01  # Default timestep for fuel consumption calculations
        return self.rocket.get_derivatives(state, self.avionics, self.atmos, t, dt)

    def run(self):
        sol = solve_ivp(self.ode_func, self.t_span, self.initial_state, method='RK45', rtol=1e-6)
        return sol
