import matplotlib.pyplot as plt
from simulation import Simulation
from rocket import Rocket
from atmosphere import Atmosphere
from avionics import Avionics
import numpy as np

def main():
    # Create rocket and atmosphere instances
    rocket = Rocket()
    atmos = Atmosphere()
    
    # Create avionics system
    avionics = Avionics(target_thrust=0.8, target_gimbal_angle=0.0)
    
    # Create simulation
    sim = Simulation(rocket, atmos, avionics, t_span=(0, 120))
    
    # Run simulation
    print("Running rocket simulation...")
    print(f"Initial mass: {rocket.total_mass:.1f} kg")
    print(f"Initial fuel: {rocket.fuel_mass:.1f} kg")
    print(f"Initial center of mass: {rocket.com:.2f} m from bottom")
    
    # Display avionics status
    avionics_status = avionics.get_status()
    print(f"Avionics - Target thrust: {avionics_status['target_thrust']:.2f}, Target gimbal: {avionics_status['target_gimbal_angle']:.1f}°")
    
    sol = sim.run()
    
    print(f"Final mass: {rocket.total_mass:.1f} kg")
    print(f"Final fuel: {rocket.fuel_mass:.1f} kg")
    print(f"Final center of mass: {rocket.com:.2f} m from bottom")
    print(f"Maximum altitude: {np.max(sol.y[1]):.1f} m")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory plot
    ax1.plot(sol.y[0], sol.y[1])
    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Rocket Trajectory')
    ax1.grid(True)
    
    # Velocity over time
    velocity_mag = np.sqrt(sol.y[2]**2 + sol.y[3]**2)
    ax2.plot(sol.t, velocity_mag)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.grid(True)
    
    # Angle over time
    ax3.plot(sol.t, np.rad2deg(sol.y[4]))
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Rocket Angle vs Time')
    ax3.grid(True)
    
    # Angular velocity over time
    ax4.plot(sol.t, np.rad2deg(sol.y[5]))
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (deg/s)')
    ax4.set_title('Angular Velocity vs Time')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
