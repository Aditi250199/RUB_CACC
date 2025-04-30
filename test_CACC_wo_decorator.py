from CACC_controller_setup import CACC, extended_CACC

import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt


class Velocity:
    def __init__(self, linear, angular=0.0):
        self.linear = linear
        self.angular = angular


class Position:
    def __init__(self, azimuth, dist):
        self.azimuth = azimuth
        self.dist = dist


class Vehicle:
    def __init__(self, vel, pos, aod=0.0):
        self.vel = vel
        self.pos = pos
        self.aod = aod


# create vehicle states
lead_vehicle = Vehicle(vel=Velocity(linear=10.0, angular=2.0), pos=Position(azimuth=0.0, dist=0.0), aod=0.0)

ego_vehicle = Vehicle(vel=Velocity(linear=8.0), pos=Position(azimuth=0.0, dist=5.0))

# create controller
# controller = CACC(k1=0.5, k2=0.3)
controller = extended_CACC(k1=0.05, k2=0.05)

# get actuation commands
acceleration, omega = controller.get_actuating_sig(lead_vehicle, ego_vehicle)

print("Acceleration:", acceleration)
print("Omega:", omega)


# simulation
# Initial conditions
lead_position = 0  # Lead car starts at position 0
ego_position = -10  # Ego car starts 10 meters behind the lead
lead_speed = 5  # Lead car moving at 5 m/s
ego_speed = 5  # Ego car starts with the same speed

# Simulation parameters
time_step = 0.1  # Time step for each simulation loop (in seconds)
simulation_duration = 30  # Duration of the simulation in seconds

# To store positions over time for plotting
ego_positions = []
lead_positions = []
times = []

# Simple follow behavior (ego adjusts speed to follow the lead)
for t in range(int(simulation_duration / time_step)):
    # Update positions based on speed
    lead_position += lead_speed * time_step
    ego_position += ego_speed * time_step

    # If the ego car is too far behind, it accelerates to catch up
    if lead_position - ego_position > 5:
        ego_speed = 6  # Ego car speeds up to close the gap
    else:
        ego_speed = 5  # Ego car maintains normal speed

    # Store the positions and time
    ego_positions.append(ego_position)
    lead_positions.append(lead_position)
    times.append(t * time_step)

    # Wait for the next time step
    time.sleep(time_step)

# Now ego_positions, lead_positions, and times are ready for plotting

# Plot the positions of the ego car and the lead car over time
plt.figure(figsize=(10, 6))
plt.plot(times, ego_positions, label='Ego Position', color='blue', linestyle='-', marker='o')
plt.plot(times, lead_positions, label='Lead Position', color='red', linestyle='--', marker='x')

# Adding titles and labels
plt.title("Ego vs Lead Car Positions Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
