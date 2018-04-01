from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps_beta

# Actual
beta = 0
v = 0
phi = phi_0
x = x_0
y = y_0
coord_actual = [x, y]

t = 0
delta_t = 0.5
delta_v = 0.2
delta_beta = math.pi / 18

prediction_horizon = 5

# Generate array from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v, v_max + delta_v, delta_v)
vector_v = np.round(vector_v, 3)

# Generate array from minimal possible angle to maximum possible angle
vector_beta = np.arange(-beta_max, beta_max + delta_beta, delta_beta)
vector_beta = np.round(vector_beta, 3)

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]

predicted_vector_x = [x]
predicted_vector_y = [y]
predicted_vector_phi = [phi]


# Function for getting distance from robot's actual position to line from initial position to target
def get_distance(x_a, y_a):
    distance = (abs((y_t - y_0) * x_a - (x_t - x_0) * y_a + x_t * y_0 - y_t * x_0)
                / (math.sqrt((y_t - y_0) ** 2 + (x_t - x_0) ** 2)))
    return distance


def saturation(value, value_max):
    if value > value_max:
        value = value_max
    elif value < -value_max:
        value = -value_max
    return value


# TODO: predictive model
def predict_velocity(vector_velocity):
    return saturation(vector_velocity[random.randint(0, size(vector_velocity))], v_max)


def predict_beta(vector_angle_beta):
    return saturation(vector_angle_beta[random.randint(0, size(vector_angle_beta))], beta_max)


def v_x(time, velocity):
    return velocity * cos(phi)


def v_y(time, velocity):
    return velocity * sin(phi)


def v_phi(time, velocity, beta):
    return (velocity / L) * math.tan(beta)


# Calculation of trajectory
# If d_phi < 0, then left turn.
# If d_phi > 0, then right_turn.
# If d_phi = 0, then beta need to be 0 and we control only velocity


# Integration
while t < 100:
    # beta = predict_beta(vector_beta)
    d_phi = phi - arctan(y_t / x_t)
    if d_phi < - eps_beta:
        beta = beta_max
    elif d_phi > eps_beta:
        beta = -beta_max
    else:
        beta = 0
    v = 0.5
    phi += sp.quad(v_phi, t, t + delta_t, args=(v, beta))[0]
    x += sp.quad(v_x, t, t + delta_t, args=(v,))[0]
    y += sp.quad(v_y, t, t + delta_t, args=(v,))[0]
    t += delta_t
    result_vector_x.append(x)
    result_vector_y.append(y)
    result_vector_phi.append(phi)

plt.figure(1)
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.plot(result_vector_x, result_vector_y)

plt.figure(2)
plt.xlabel("Time")
plt.ylabel("Angle Phi")
plt.plot(np.linspace(0, 100, np.size(result_vector_phi)), result_vector_phi)
