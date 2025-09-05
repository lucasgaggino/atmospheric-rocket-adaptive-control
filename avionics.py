import numpy as np

class Avionics:
    def __init__(self, target_thrust=0.75, target_gimbal_angle=0.0, 
                 min_thrust=0.5, max_thrust=1.0, max_gimbal_angle=15.0):
        """
        Avionics system that controls rocket thrust and gimbal angle.
        
        Parameters:
        - target_thrust: Desired thrust fraction (0.5 to 1.0)
        - target_gimbal_angle: Desired gimbal angle in degrees (-15 to +15)
        - min_thrust: Minimum allowable thrust fraction
        - max_thrust: Maximum allowable thrust fraction  
        - max_gimbal_angle: Maximum gimbal angle in degrees
        """
        self.target_thrust = target_thrust
        self.target_gimbal_angle = target_gimbal_angle
        
        # System limits
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.max_gimbal_angle = max_gimbal_angle
        
        # Current commanded values (with limits applied)
        self.commanded_thrust = self._limit_thrust(target_thrust)
        self.commanded_gimbal_angle = self._limit_gimbal_angle(target_gimbal_angle)
        
    def _limit_thrust(self, thrust):
        """Apply thrust limits"""
        return np.clip(thrust, self.min_thrust, self.max_thrust)
    
    def _limit_gimbal_angle(self, angle):
        """Apply gimbal angle limits (in degrees)"""
        return np.clip(angle, -self.max_gimbal_angle, self.max_gimbal_angle)
    
    def set_thrust(self, thrust_fraction):
        """Set target thrust fraction"""
        self.target_thrust = thrust_fraction
        self.commanded_thrust = self._limit_thrust(thrust_fraction)
        
    def set_gimbal_angle(self, angle_degrees):
        """Set target gimbal angle in degrees"""
        self.target_gimbal_angle = angle_degrees
        self.commanded_gimbal_angle = self._limit_gimbal_angle(angle_degrees)
        
    def get_commands(self):
        """Get current thrust and gimbal commands"""
        return self.commanded_thrust, np.deg2rad(self.commanded_gimbal_angle)
    
    def update(self, time, rocket_state=None):
        """
        Update avionics commands based on time and rocket state.
        Currently just holds the set values, but can be extended for feedback control.
        
        Parameters:
        - time: Current simulation time
        - rocket_state: Current rocket state [x, y, vx, vy, theta, omega] (optional)
        
        Returns:
        - thrust_fraction: Commanded thrust fraction (0.5 to 1.0)
        - gimbal_angle_rad: Commanded gimbal angle in radians
        """
        # For now, just return the held values
        # Future implementations can add feedback control logic here
        return self.get_commands()
    
    def get_status(self):
        """Get avionics system status"""
        return {
            'target_thrust': self.target_thrust,
            'commanded_thrust': self.commanded_thrust,
            'target_gimbal_angle': self.target_gimbal_angle,
            'commanded_gimbal_angle': self.commanded_gimbal_angle,
            'thrust_limited': abs(self.target_thrust - self.commanded_thrust) > 1e-6,
            'gimbal_limited': abs(self.target_gimbal_angle - self.commanded_gimbal_angle) > 1e-6
        }
