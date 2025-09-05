import matplotlib.pyplot as plt
from simulation import Simulation
from rocket import Rocket
from atmosphere import Atmosphere
from pid_avionics import PIDAvionics
from trajectory import ParabolicTrajectory, LinearTrajectory
import numpy as np

def main():
    """Main function to run PID trajectory following simulation"""
    
    print("=== Rocket Simulation with PID Trajectory Following ===\n")
    
    # Create rocket and atmosphere instances
    rocket = Rocket()
    atmos = Atmosphere()
    
    # Create parabolic trajectory: 5km peak height, going left
    trajectory = ParabolicTrajectory(
        peak_height=5000, 
        horizontal_range=10000, 
        direction='left'
    )
    
    print("Trajectory Parameters:")
    print(f"- Peak height: {trajectory.peak_height} m")
    print(f"- Horizontal range: {trajectory.horizontal_range} m")
    print(f"- Direction: {trajectory.direction}")
    print(f"- Flight time: {trajectory.flight_time:.1f} s")
    
    # Create PID avionics system
    pid_avionics = PIDAvionics(
        trajectory=trajectory,
        position_gains={'kp': 0.002, 'ki': 0.0001, 'kd': 0.02},
        velocity_gains={'kp': 0.01, 'ki': 0.001, 'kd': 0.05}
    )
    
    # Display initial rocket status
    print(f"\nRocket Initial Conditions:")
    print(f"- Initial mass: {rocket.total_mass:.1f} kg")
    print(f"- Initial fuel: {rocket.fuel_mass:.1f} kg")
    print(f"- Initial center of mass: {rocket.com:.2f} m from bottom")
    
    # Display avionics status
    avionics_status = pid_avionics.get_status()
    print(f"\nAvionics Configuration:")
    print(f"- Base thrust: {pid_avionics.base_thrust:.2f}")
    print(f"- Base gimbal: {pid_avionics.base_gimbal:.1f}°")
    print(f"- Trajectory duration: {avionics_status['trajectory_duration']:.1f} s")
    
    # Create simulation
    sim_time = min(120, trajectory.flight_time + 20)  # Limit simulation time
    sim = Simulation(rocket, atmos, pid_avionics, t_span=(0, sim_time))
    
    print(f"\n=== Running Simulation for {sim_time:.1f} seconds ===")
    
    # Run simulation
    sol = sim.run()
    
    print(f"\nSimulation Results:")
    print(f"- Final mass: {rocket.total_mass:.1f} kg")
    print(f"- Final fuel: {rocket.fuel_mass:.1f} kg")
    print(f"- Final center of mass: {rocket.com:.2f} m from bottom")
    print(f"- Maximum altitude reached: {np.max(sol.y[1]):.1f} m")
    print(f"- Final horizontal position: {sol.y[0][-1]:.1f} m")
    
    # Get reference trajectory for comparison
    ref_times = np.linspace(0, min(sim_time, trajectory.flight_time), 100)
    ref_positions = [trajectory.get_target_state(t) for t in ref_times]
    ref_x = [pos[0] for pos in ref_positions]
    ref_y = [pos[1] for pos in ref_positions]
    
    # Calculate tracking errors
    pos_errors_x = []
    pos_errors_y = []
    
    for i, t in enumerate(sol.t):
        if t <= trajectory.flight_time:
            x_target, y_target, _, _ = trajectory.get_target_state(t)
            pos_errors_x.append(x_target - sol.y[0][i])
            pos_errors_y.append(y_target - sol.y[1][i])
        else:
            pos_errors_x.append(0)  # No target after trajectory ends
            pos_errors_y.append(0)
    
    if len(pos_errors_x) > 0:
        avg_error_x = np.mean([abs(e) for e in pos_errors_x])
        avg_error_y = np.mean([abs(e) for e in pos_errors_y])
        max_error_x = max([abs(e) for e in pos_errors_x])
        max_error_y = max([abs(e) for e in pos_errors_y])
        
        print(f"\nTracking Performance:")
        print(f"- Average X error: {avg_error_x:.1f} m")
        print(f"- Average Y error: {avg_error_y:.1f} m")
        print(f"- Maximum X error: {max_error_x:.1f} m")
        print(f"- Maximum Y error: {max_error_y:.1f} m")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Trajectory plot
    ax1.plot(ref_x, ref_y, 'r--', linewidth=3, label='Target Trajectory', alpha=0.8)
    ax1.plot(sol.y[0], sol.y[1], 'b-', linewidth=2, label='Actual Trajectory')
    ax1.plot(0, 0, 'go', markersize=8, label='Start')
    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Trajectory Following: PID Control')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Velocity over time
    velocity_mag = np.sqrt(sol.y[2]**2 + sol.y[3]**2)
    ax2.plot(sol.t, sol.y[2], 'r-', label='Horizontal Velocity')
    ax2.plot(sol.t, sol.y[3], 'b-', label='Vertical Velocity')
    ax2.plot(sol.t, velocity_mag, 'g--', alpha=0.7, label='Total Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Components')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Rocket angle and angular velocity
    ax3.plot(sol.t, np.rad2deg(sol.y[4]), 'r-', label='Angle (deg)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(sol.t, np.rad2deg(sol.y[5]), 'b-', alpha=0.7, label='Angular Velocity (deg/s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)', color='r')
    ax3_twin.set_ylabel('Angular Velocity (deg/s)', color='b')
    ax3.set_title('Rocket Orientation')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # 4. Position errors over time (if we have tracking data)
    if len(pos_errors_x) > 0:
        ax4.plot(sol.t, pos_errors_x, 'r-', label='X Position Error')
        ax4.plot(sol.t, pos_errors_y, 'b-', label='Y Position Error')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Position Error (m)')
        ax4.set_title('Position Tracking Errors')
        ax4.grid(True)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No tracking data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Position Tracking Errors')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
