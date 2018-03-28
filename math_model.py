import matplotlib.pyplot as plt
import scipy.integrate as scipy
import numpy as np

from scipy import *
from config import phi_0, r, L, y_t, y_0, x_t, x_0

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
delta_beta = math.pi/24

criterion = 1

# Generate array from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v - delta_v * 5, v + delta_v * 6, delta_v)
vector_v = np.round(vector_v, 1)


# Generate array from minimal possible angle to maximum possible angle
vector_beta = np.arange(beta - delta_beta * 5, beta + delta_beta*6, delta_beta)
vector_beta = np.round(vector_beta, 2)


vector_x = [x]
vector_y = [y]
vector_phi = [phi]


def get_distance(x_a, y_a):
    distance = (abs((y_t - y_0) * x_a - (x_t - x_0) * y_a + x_t * y_0 - y_t * x_0)
                / (math.sqrt((y_t - y_0) ** 2 + (x_t - x_0) ** 2)))
    return distance


# TODO: predictive model
def predict_velocity(vector_v):
    return vector_v[10]


def predict_beta(vector_beta):
    return 5


def v_x(time):
    return predict_velocity(vector_v) * cos(phi)


def v_y(time):
    return predict_velocity(vector_v) * sin(phi)


def v_phi(time):
    return (predict_velocity(vector_v) / L) * tan(beta)


# Integration
while t != 10:
    x += scipy.quad(v_x, t, t + delta_t)[0]
    y += scipy.quad(v_y, t, t + delta_t)[0]
    phi += scipy.quad(v_phi, t, t + delta_t)[0]
    t += delta_t
    vector_x.append(x)
    vector_y.append(y)
    vector_phi.append(phi)

plt.plot(vector_x, vector_y)
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.show()
