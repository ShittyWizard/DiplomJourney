import math

eps = 0.01
eps_beta = 0.1
# Parameters
L = 0.5

# Constraints
delta_t = 0.05

beta_max = math.radians(60)
delta_beta = math.radians(1)
beta_acc_max = math.radians(400)

# Velocity per second
v_max = 1
# For example: for 1 second vector of velocities could be [0, 0.02, 0.04, 0.06, 0.08, 0.1]
delta_v = 0.01
v_acc_max = 0.5

# Initial conditions
x_0 = 0
y_0 = 0
phi_0 = math.pi * 5/6

# Target
x_t = -1
y_t = -5
