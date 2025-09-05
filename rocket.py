import numpy as np
from avionics import Avionics
from atmosphere import Atmosphere

class Rocket:
    def __init__(self, dry_mass=500, fuel_mass=500, max_thrust=15000, engine_gimbal_range=15, 
                 rocket_length=20.0, fuel_tank_length=10.0, g=9.81, specific_impulse=300):
        # Mass properties
        self.dry_mass = dry_mass
        self.initial_fuel_mass = fuel_mass
        self.fuel_mass = fuel_mass
        self.total_mass = dry_mass + fuel_mass
        
        # Engine properties
        self.max_thrust = max_thrust
        self.engine_gimbal_range = np.deg2rad(engine_gimbal_range)
        self.specific_impulse = specific_impulse
        self.g = g
        
        # Geometry properties
        self.rocket_length = rocket_length
        self.fuel_tank_length = fuel_tank_length
        
        # Center of mass calculation (measured from bottom of rocket)
        self.dry_com = rocket_length * 0.6  # Dry mass CoM at 60% of rocket length
        self.fuel_com_empty = rocket_length * 0.4  # Bottom of fuel tank
        self.fuel_com_full = rocket_length * 0.4 + fuel_tank_length * 0.5  # Middle of fuel tank
        
        # Engine position (distance from current CoM)
        self.engine_position = 0.0  # At the bottom of the rocket
        
        # Initialize center of mass and inertia
        self.update_center_of_mass()
        self.update_inertia()

    def update_center_of_mass(self):
        """Update center of mass as fuel is consumed"""
        if self.fuel_mass <= 0:
            self.com = self.dry_com
        else:
            fuel_fraction = self.fuel_mass / self.initial_fuel_mass
            # Fuel CoM moves from full position toward empty position as fuel is consumed
            current_fuel_com = self.fuel_com_empty + fuel_fraction * (self.fuel_com_full - self.fuel_com_empty)
            
            # Combined center of mass
            total_mass = self.dry_mass + self.fuel_mass
            self.com = (self.dry_mass * self.dry_com + self.fuel_mass * current_fuel_com) / total_mass
        
        # Update engine distance from current CoM
        self.engine_distance = self.com - self.engine_position
    
    def update_inertia(self):
        """Update moment of inertia based on current mass distribution"""
        # Simplified inertia calculation - treat as point masses
        dry_inertia = self.dry_mass * (self.dry_com - self.com)**2
        
        if self.fuel_mass > 0:
            fuel_fraction = self.fuel_mass / self.initial_fuel_mass
            current_fuel_com = self.fuel_com_empty + fuel_fraction * (self.fuel_com_full - self.fuel_com_empty)
            fuel_inertia = self.fuel_mass * (current_fuel_com - self.com)**2
        else:
            fuel_inertia = 0
            
        # Add base rotational inertia
        base_inertia = (self.dry_mass + self.fuel_mass) * (self.rocket_length / 6)**2
        self.inertia = base_inertia + dry_inertia + fuel_inertia
    
    def consume_fuel(self, dt, thrust_mag):
        """Consume fuel based on thrust and time step"""
        if self.fuel_mass > 0 and thrust_mag > 0:
            # Mass flow rate based on thrust and specific impulse
            mass_flow_rate = thrust_mag / (self.specific_impulse * self.g)
            fuel_consumed = mass_flow_rate * dt
            
            self.fuel_mass = max(0, self.fuel_mass - fuel_consumed)
            self.total_mass = self.dry_mass + self.fuel_mass
            
            # Update CoM and inertia
            self.update_center_of_mass()
            self.update_inertia()

    def get_derivatives(self, state, avionics:Avionics, atmos:Atmosphere, time, dt=0.01):
        """
        Get rocket derivatives based on state, avionics, atmosphere, time, and dt
        
        Parameters:
        - state: Current rocket state [x, y, vx, vy, theta, omega]
        - avionics: Avionics system instance
        - atmos: Atmosphere instance
        - time: Current simulation time
        - dt: Time step (optional, default is 0.01 seconds)
        """
        x, y, vx, vy, theta, omega = state
        
        # Get commands from avionics system
        thrust_fraction, thrust_angle = avionics.update(time, state)
        print(f"Thrust fraction: {thrust_fraction}, Thrust angle: {thrust_angle}")

        thrust_dir = theta + thrust_angle
        thrust_mag = self.max_thrust * thrust_fraction if self.fuel_mass > 0 else 0
        thrust_force = thrust_mag * np.array([np.cos(thrust_dir), np.sin(thrust_dir)])

        # Consume fuel and update mass properties
        self.consume_fuel(dt, thrust_mag)

        gravity_force = np.array([0, -self.total_mass * self.g])

        velocity = np.array([vx, vy])
        
        aero_force, aero_torque = atmos.get_forces(y, velocity, theta)
        

        total_force = thrust_force + gravity_force + aero_force
        accel = total_force / self.total_mass
        
        thrust_torque = thrust_mag * self.engine_distance * np.sin(thrust_angle)
        total_torque = thrust_torque + aero_torque
        alpha = total_torque / self.inertia

        dstates = np.array([vx, vy, accel[0], accel[1], omega, alpha])
        return dstates
