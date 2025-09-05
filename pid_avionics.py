import numpy as np
from trajectory import Trajectory

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        """
        PID Controller implementation
        
        Parameters:
        - kp: Proportional gain
        - ki: Integral gain  
        - kd: Derivative gain
        - output_limits: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        
    def update(self, error, current_time):
        """
        Update PID controller with current error
        
        Parameters:
        - error: Current error value
        - current_time: Current time in seconds
        
        Returns:
        - control_output: PID controller output
        """
        if self.previous_time is None:
            self.previous_time = current_time
            dt = 0.01  # Default dt for first iteration
        else:
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 0.01  # Prevent division by zero
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        derivative_term = self.kd * derivative
        
        # Calculate output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Store values for next iteration
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller internal state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

class PIDAvionics:
    def __init__(self, trajectory, 
                 position_gains={'kp': 0.01, 'ki': 0.001, 'kd': 0.1},
                 velocity_gains={'kp': 0.05, 'ki': 0.005, 'kd': 0.2},
                 min_thrust=0.5, max_thrust=1.0, max_gimbal_angle=15.0):
        """
        Avionics system with PID controllers for trajectory following
        
        Parameters:
        - trajectory: Trajectory object to follow
        - position_gains: PID gains for position control {'kp': , 'ki': , 'kd': }
        - velocity_gains: PID gains for velocity control {'kp': , 'ki': , 'kd': }
        - min_thrust: Minimum thrust fraction
        - max_thrust: Maximum thrust fraction
        - max_gimbal_angle: Maximum gimbal angle in degrees
        """
        self.trajectory = trajectory
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.max_gimbal_angle = max_gimbal_angle
        
        # PID controllers for position (x, y)
        self.position_x_pid = PIDController(
            kp=position_gains['kp'], 
            ki=position_gains['ki'], 
            kd=position_gains['kd'],
            output_limits=(-max_gimbal_angle*0.8, max_gimbal_angle*0.8)
        )
        
        self.position_y_pid = PIDController(
            kp=position_gains['kp']*2, 
            ki=position_gains['ki']*2, 
            kd=position_gains['kd']*2,
            output_limits=(-0.4, 0.4)  # Thrust adjustment range
        )
        
        # PID controllers for velocity (vx, vy)
        self.velocity_x_pid = PIDController(
            kp=velocity_gains['kp'], 
            ki=velocity_gains['ki'], 
            kd=velocity_gains['kd'],
            output_limits=(-max_gimbal_angle*0.3, max_gimbal_angle*0.3)
        )
        
        self.velocity_y_pid = PIDController(
            kp=velocity_gains['kp'], 
            ki=velocity_gains['ki'], 
            kd=velocity_gains['kd'],
            output_limits=(-0.3, 0.3)  # Velocity-based thrust adjustment
        )
        
        # Base control values - need higher thrust for trajectory following
        self.base_thrust = 0.85  # Nominal thrust level (higher for better control)
        self.base_gimbal = 0.0   # Nominal gimbal angle
        
        # Control outputs
        self.commanded_thrust = self.base_thrust
        self.commanded_gimbal_angle = self.base_gimbal
        
        # Tracking variables
        self.current_error = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}
        
    def update(self, time, rocket_state):
        """
        Update avionics commands based on trajectory and current rocket state
        
        Parameters:
        - time: Current simulation time
        - rocket_state: Current rocket state [x, y, vx, vy, theta, omega]
        
        Returns:
        - thrust_fraction: Commanded thrust fraction
        - gimbal_angle_rad: Commanded gimbal angle in radians
        """
        if rocket_state is None:
            return self.base_thrust, np.deg2rad(self.base_gimbal)
            
        # Current rocket state
        x, y, vx, vy, theta, omega = rocket_state
        
        # Get target state from trajectory
        x_target, y_target, vx_target, vy_target = self.trajectory.get_target_state(time)
        
        # Calculate position errors
        error_x = x_target - x
        error_y = y_target - y
        
        # Calculate velocity errors
        error_vx = vx_target - vx
        error_vy = vy_target - vy
        
        # Store current errors for monitoring
        self.current_error = {
            'x': error_x, 'y': error_y, 
            'vx': error_vx, 'vy': error_vy
        }
        
        # Update PID controllers
        # Position control
        pos_x_correction = self.position_x_pid.update(error_x, time)
        pos_y_correction = self.position_y_pid.update(error_y, time)
        
        # Velocity control
        vel_x_correction = self.velocity_x_pid.update(error_vx, time)
        vel_y_correction = self.velocity_y_pid.update(error_vy, time)
        
        # Combine corrections
        # Gimbal angle control (primarily for horizontal position/velocity)
        gimbal_correction = pos_x_correction + vel_x_correction
        self.commanded_gimbal_angle = np.clip(
            self.base_gimbal + gimbal_correction, 
            -self.max_gimbal_angle, 
            self.max_gimbal_angle
        )
        
        # Thrust control (primarily for vertical position/velocity)
        thrust_correction = pos_y_correction + vel_y_correction
        self.commanded_thrust = np.clip(
            self.base_thrust + thrust_correction,
            self.min_thrust,
            self.max_thrust
        )
        
        return self.commanded_thrust, np.deg2rad(self.commanded_gimbal_angle)
    
    def get_status(self):
        """Get avionics system status"""
        return {
            'commanded_thrust': self.commanded_thrust,
            'commanded_gimbal_angle': self.commanded_gimbal_angle,
            'position_error_x': self.current_error['x'],
            'position_error_y': self.current_error['y'],
            'velocity_error_x': self.current_error['vx'],
            'velocity_error_y': self.current_error['vy'],
            'trajectory_duration': self.trajectory.get_duration()
        }
    
    def reset_controllers(self):
        """Reset all PID controllers"""
        self.position_x_pid.reset()
        self.position_y_pid.reset()
        self.velocity_x_pid.reset()
        self.velocity_y_pid.reset()
    
    def tune_gains(self, position_gains=None, velocity_gains=None):
        """
        Update PID gains during runtime
        
        Parameters:
        - position_gains: New position PID gains {'kp': , 'ki': , 'kd': }
        - velocity_gains: New velocity PID gains {'kp': , 'ki': , 'kd': }
        """
        if position_gains:
            self.position_x_pid.kp = position_gains.get('kp', self.position_x_pid.kp)
            self.position_x_pid.ki = position_gains.get('ki', self.position_x_pid.ki)
            self.position_x_pid.kd = position_gains.get('kd', self.position_x_pid.kd)
            
            self.position_y_pid.kp = position_gains.get('kp', self.position_y_pid.kp)
            self.position_y_pid.ki = position_gains.get('ki', self.position_y_pid.ki)
            self.position_y_pid.kd = position_gains.get('kd', self.position_y_pid.kd)
        
        if velocity_gains:
            self.velocity_x_pid.kp = velocity_gains.get('kp', self.velocity_x_pid.kp)
            self.velocity_x_pid.ki = velocity_gains.get('ki', self.velocity_x_pid.ki)
            self.velocity_x_pid.kd = velocity_gains.get('kd', self.velocity_x_pid.kd)
            
            self.velocity_y_pid.kp = velocity_gains.get('kp', self.velocity_y_pid.kp)
            self.velocity_y_pid.ki = velocity_gains.get('ki', self.velocity_y_pid.ki)
            self.velocity_y_pid.kd = velocity_gains.get('kd', self.velocity_y_pid.kd)
