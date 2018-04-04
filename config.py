import math


eps = 0.5
eps_beta = 0.1
# Parameters
L = 0.5
r = 0.05

# Constraints
delta_t = 0.5
delta_v = 0.2
delta_beta = math.pi / 24
beta_max = math.pi / 6
w_max = 20
v_max = w_max * r  # 1 m/s

# Initial conditions
x_0 = 0
y_0 = 0
coord_0 = [x_0, y_0]
phi_0 = 0

# Target
x_t = 5
y_t = 5
coord_t = [x_t, y_t]
phi_t = math.pi/2
