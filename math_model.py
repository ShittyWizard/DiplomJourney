from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max

# Actual
beta = 0
v = 0
phi = phi_0
x = x_0
y = y_0
coord_actual = [x, y]

t = 0
delta_t = 0.5
delta_v = 0.1
delta_beta = math.pi / 24

criterion = 1

# Generate array from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v, v + delta_v * 6, delta_v)
vector_v = np.round(vector_v, 1)

# Generate array from minimal possible angle to maximum possible angle
vector_beta = np.arange(beta - delta_beta * 5, beta + delta_beta * 6, delta_beta)
vector_beta = np.round(vector_beta, 3)

# Generate vectors with coordinates for plotting
vector_x = [x]
vector_y = [y]
vector_phi = [phi]


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
def predict_velocity(vector_v):
    return saturation(vector_v[random.randint(0, size(vector_v))], v_max)


def predict_beta(vector_beta):
    return saturation(vector_beta[random.randint(0, size(vector_beta))], beta_max)


def v_x(time, velocity):
    return velocity * cos(phi)


def v_y(time, velocity):
    return velocity * sin(phi)


def v_phi(time, velocity):
    return (velocity / L) * math.tan(beta)


# Integration
while t < 100:
    beta = predict_beta(vector_beta)
    v = predict_velocity(vector_v)
    phi += sp.quad(v_phi, t, t + delta_t, args=(v,))[0]
    x += sp.quad(v_x, t, t + delta_t, args=(v,))[0]
    y += sp.quad(v_y, t, t + delta_t, args=(v,))[0]
    t += delta_t
    print(beta, v)
    vector_x.append(x)
    vector_y.append(y)
    vector_phi.append(phi)

plt.figure(1)
plt.subplot(211)
plt.plot(vector_x, vector_y, 'bo')

plt.subplot(212)
plt.plot(np.linspace(0, 100, np.size(vector_phi)), vector_phi)
plt.show()
