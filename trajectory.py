import numpy as np
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self):
        """Base trajectory class"""
        pass
    
    def get_target_state(self, time):
        """
        Get target state at given time
        Returns: (x_target, y_target, vx_target, vy_target)
        """
        raise NotImplementedError("Subclasses must implement get_target_state")
    
    def get_duration(self):
        """Get total trajectory duration"""
        raise NotImplementedError("Subclasses must implement get_duration")

class ParabolicTrajectory(Trajectory):
    def __init__(self, peak_height=5000, horizontal_range=10000, direction='left'):
        """
        Create a parabolic trajectory
        
        Parameters:
        - peak_height: Maximum altitude in meters
        - horizontal_range: Total horizontal distance in meters
        - direction: 'left' or 'right' for trajectory direction
        """
        super().__init__()
        self.peak_height = peak_height
        self.horizontal_range = horizontal_range
        self.direction = direction
        
        # Calculate trajectory parameters
        # For a parabola: y = ax^2 + bx + c
        # Peak at x = horizontal_range/2, y = peak_height
        # Start at (0,0), end at (horizontal_range, 0)
        
        if direction == 'left':
            # Trajectory goes from (0,0) to (-horizontal_range, 0) with peak at (-horizontal_range/2, peak_height)
            self.x_start = 0
            self.x_end = -horizontal_range
            self.x_peak = -horizontal_range / 2
        else:
            # Trajectory goes from (0,0) to (horizontal_range, 0) with peak at (horizontal_range/2, peak_height)
            self.x_start = 0
            self.x_end = horizontal_range
            self.x_peak = horizontal_range / 2
        
        # Parabola coefficients: y = a(x - x_peak)^2 + peak_height
        # Since it passes through (x_start, 0): 0 = a(x_start - x_peak)^2 + peak_height
        self.a = -peak_height / (self.x_start - self.x_peak)**2
        
        # Calculate flight time assuming constant horizontal velocity
        # Estimate flight time based on ballistic trajectory
        self.flight_time = 2 * np.sqrt(2 * peak_height / 9.81)  # Approximate ballistic flight time
        self.horizontal_velocity = (self.x_end - self.x_start) / self.flight_time
        
    def get_target_state(self, time):
        """Get target position and velocity at given time"""
        if time > self.flight_time:
            # After trajectory ends, maintain final position
            x_target = self.x_end
            y_target = 0
            vx_target = 0
            vy_target = 0
        else:
            # Current position along trajectory
            x_target = self.x_start + self.horizontal_velocity * time
            y_target = self.a * (x_target - self.x_peak)**2 + self.peak_height
            
            # Target velocities (derivatives of position)
            vx_target = self.horizontal_velocity
            vy_target = 2 * self.a * (x_target - self.x_peak) * self.horizontal_velocity
        
        return x_target, y_target, vx_target, vy_target
    
    def get_duration(self):
        """Get trajectory duration"""
        return self.flight_time
    
    def plot_trajectory(self, num_points=100):
        """Plot the trajectory for visualization"""
        times = np.linspace(0, self.flight_time, num_points)
        positions = [self.get_target_state(t) for t in times]
        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Target Trajectory')
        plt.xlabel('Horizontal Position (m)')
        plt.ylabel('Altitude (m)')
        plt.title(f'Parabolic Trajectory (Peak: {self.peak_height}m, Direction: {self.direction})')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()
        
        return x_vals, y_vals

class LinearTrajectory(Trajectory):
    def __init__(self, start_pos=(0, 0), end_pos=(1000, 1000), duration=60):
        """
        Create a linear trajectory from start to end position
        
        Parameters:
        - start_pos: (x, y) starting position
        - end_pos: (x, y) ending position  
        - duration: Time to complete trajectory in seconds
        """
        super().__init__()
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.duration = duration
        
        # Calculate constant velocity
        self.velocity = (self.end_pos - self.start_pos) / duration
        
    def get_target_state(self, time):
        """Get target position and velocity at given time"""
        if time > self.duration:
            # After trajectory ends, maintain final position
            x_target, y_target = self.end_pos
            vx_target = vy_target = 0
        else:
            # Linear interpolation
            position = self.start_pos + self.velocity * time
            x_target, y_target = position
            vx_target, vy_target = self.velocity
            
        return x_target, y_target, vx_target, vy_target
    
    def get_duration(self):
        """Get trajectory duration"""
        return self.duration
