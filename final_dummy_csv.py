# generate_lead_csv.py
#generates a dummy trajectory for a lead vehicle moving in a circular path, with a constant velocity and angular velocity

import pandas as pd
import numpy as np

# Create a dummy lead vehicle trajectory following a curve
timesteps = 100
dt = 0.1  #simulation time step size in sec
time_arr = np.linspace(0, (timesteps-1)*dt, timesteps)

#circular trajectory
radius = 20  # meters
omega_const = 0.1  # rad/s #ang vel for lead vehicle
v_const = radius * omega_const #constant linear velocity of the lead vehicle

#Lead Vehicle's Position and Orientation
lead_x = radius * np.sin(omega_const * time_arr) #x-coordinate of the lead vehicle at each timestep along a circular path
lead_y = radius * (1 - np.cos(omega_const * time_arr)) #y-coordinate of the lead vehicle at each timestep along a circular path
lead_theta = omega_const * time_arr  # orientation of lead #heading increases linearly

# Save into a DataFrame
lead_data = pd.DataFrame({
    'time': time_arr,
    'x': lead_x,
    'y': lead_y,
    'theta': lead_theta,
    'v': np.full_like(time_arr, v_const),
    'omega': np.full_like(time_arr, omega_const)
})

# Save CSV
lead_data.to_csv('lead_vehicle_trajectory.csv', index=False)
print("Dummy lead vehicle trajectory CSV created successfully.")
