import matplotlib.pyplot as plt
from simulation import Simulation
from rocket import Rocket
from atmosphere import Atmosphere
import numpy as np

def main():
    rocket = Rocket()
    atmos = Atmosphere()
    sim = Simulation(rocket, atmos)
    sol = sim.run()

    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Rocket Trajectory')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
