import math

eps = 0.01
# Parameters
L = 1
r = 0.1

# Constraints
beta_max = math.pi / 6
beta_v_max = math.pi / 12
w_max = 10
v_max = w_max * r  # 1 m/s

# Initial conditions
x_0 = 0
y_0 = 0
coord_0 = [x_0, y_0]
phi_0 = math.pi/6

# Target
x_t = 10
y_t = 10
coord_t = [x_t, y_t]
phi_t = 0