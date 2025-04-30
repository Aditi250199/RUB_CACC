# simulate_cacc.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from final_CACC_controller_setup import CACC, extended_CACC


# vehicles.py
import numpy as np

class Position:
    def __init__(self, dist=0.0, azimuth=0.0):
        self.dist = dist
        self.azimuth = azimuth

class Velocity:
    def __init__(self, linear=0.0, angular=0.0):
        self.linear = linear
        self.angular = angular

class LeadVehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.vel = Velocity(v, omega)
        self.aod = theta  # assume AoD same as heading
        self.pos = Position() #initializing with its default values

class EgoVehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.vel = Velocity(0.0, 0.0)
        self.pos = Position()

    # vehicles.py (fix)
    def update_state(self, a, omega, dt):
        self.vel.linear += float(a) * dt
        self.theta += float(omega) * dt
        self.x += self.vel.linear * np.cos(self.theta) * dt
        self.y += self.vel.linear * np.sin(self.theta) * dt

        # Force to float, in case anything becomes numpy array accidentally
        self.x = float(self.x)
        self.y = float(self.y)
        self.theta = float(self.theta)
        self.vel.linear = float(self.vel.linear)


# Load lead car CSV
lead_data = pd.read_csv('lead_vehicle_trajectory.csv')

# Initialize vehicles and controller
ego = EgoVehicle(x=-5, y=0, theta=0)
controller = extended_CACC(k1=1.0, k2= 1.0, headway=2.0, time_gap=1.5)

ego_trajectory = []

# Simulation loop
dt = 0.1  # same as CSV dt
for idx, row in lead_data.iterrows():
    lead = LeadVehicle(x=row['x'], y=row['y'], theta=row['theta'], v=row['v'], omega=row['omega'])

    # Compute relative position
    dx = lead.x - ego.x
    dy = lead.y - ego.y
    dist = np.hypot(dx, dy)
    azimuth = np.arctan2(dy, dx) - ego.theta

    lead.pos.dist = dist
    lead.pos.azimuth = azimuth

    ego.pos.dist = dist
    ego.pos.azimuth = azimuth

    # Controller output
    a, omega = controller.get_actuating_sig(lead, ego)

    # Update ego
    ego.update_state(a, omega, dt)

    # Save trajectory
    ego_trajectory.append((ego.x, ego.y))

# Plot results
ego_trajectory = np.array(ego_trajectory)


plt.figure(figsize=(8,6))
plt.plot(lead_data['x'], lead_data['y'], label='Lead Vehicle', linestyle='--')
plt.plot(ego_trajectory[:,0], ego_trajectory[:,1], label='Ego Vehicle')
plt.legend()
plt.xlabel('X Position [m]')
plt.ylabel('Y Position [m]')
plt.title('Ego Car Following Lead Car Using Extended CACC')
plt.axis('equal')
plt.grid(True)
plt.show()
