import math

eps = 0.001
eps_beta = math.radians(5)
# Parameters
L = 0.5

# Constraints
delta_t = 0.05

beta_max = math.radians(60)
delta_beta = math.radians(1)
beta_acc_max = math.radians(400)

# Velocity per second
v_max = 1
v_min = 0.4
delta_v = 0.005
v_acc_max = 0.5

# Initial conditions
x_0 = 0
y_0 = 0
phi_0 = 0

# Target
x_t = 1
y_t = 5
