from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps_beta, eps

# Actual
beta = 0
v = 0
phi = float(phi_0)
x = x_0
y = y_0
coord_actual = [x, y]

t = 0
delta_t = 0.5
delta_v = 0.2
delta_beta = math.pi / 18

# Prediction horizon * delta_t = 2.5s
prediction_horizon = 3

# Generate vector from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v, v_max + delta_v, delta_v)
vector_v = np.round(vector_v, 3)
# [0.0 0.2 0.4 0.6 0.8 1.0]  6


# Generate vector from minimal possible angle to maximum possible angle
vector_beta = np.arange(-beta_max, beta_max + delta_beta, delta_beta)
vector_beta = np.round(vector_beta, 3)
# [-0.524 -0.349 -0.175  0.000  0.175  0.349  0.524]  7


#  Generate vector with distance from robot's actual position to line from initial position to target
vector_distance = np.arange(0, 0, 0.1)

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]

# Generate vectors with coordinates for calculation of trajectory
predicted_vector_x = [x]
predicted_vector_y = [y]
predicted_vector_phi = [phi]


def is_on_target(actual_x, actual_y, target_x, target_y):
    if (target_x - actual_x) ^ 2 + (target_y - actual_y) ^ 2 <= eps:
        return True


# Function for getting distance from robot's actual position to line from initial position to target
def get_distance_from_line(x_a, y_a):
    distance = (abs((y_t - y_0) * x_a - (x_t - x_0) * y_a + x_t * y_0 - y_t * x_0)
                / (math.sqrt((y_t - y_0) ** 2 + (x_t - x_0) ** 2)))
    return distance


def get_distance_from_target(x_a, y_a):
    return math.sqrt((x_t - x_a) ^ 2 + (y_t - y_a) ^ 2)


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


def v_x(time, _velocity):
    return _velocity * cos(phi)


def v_y(time, _velocity):
    return _velocity * sin(phi)


def v_phi(time, _velocity, angle_beta):
    return (_velocity / L) * math.tan(angle_beta)


def control_criterion(_predicted_x, _predicted_y, _predicted_phi):
    distance_from_line = get_distance_from_line(_predicted_x, _predicted_y)
    distance_from_target = get_distance_from_target(predicted_vector_x, _predicted_y)
    # Add angle criterion
    return distance_from_line + distance_from_target


# Integration
def integrate_velocity(velocity_function, velocity_value, t_start, t_stop):
    return sp.quad(velocity_function, t_start, t_stop, args=(velocity_value,))[0]


def integrate_angle(angle_function, velocity_value, angle_value, t_start, t_stop):
    return sp.quad(angle_function, t_start, t_stop, args=(velocity_value, angle_value))[0]


def coordinate_x(_v):
    return integrate_velocity(v_x, _v, t, t + delta_t)


def coordinate_y(_v):
    return integrate_velocity(v_y, _v, t, t + delta_t)


def angle_phi(_v, _beta):
    return integrate_angle(v_phi, _v, _beta, t, t + delta_t)


def iteration_of_predict(_initial_coordinates, _v, _angle):
    _x = coordinate_x(_v)
    _y = coordinate_y(_v)
    _phi = angle_phi(_v, _angle)
    return [_initial_coordinates[0] + _x, _initial_coordinates[1] + _y, _initial_coordinates[2] + _phi]


# while not is_on_target(x, y, x_t, y_t):
initial_coordinates_0 = [x_0, y_0, phi_0]
initial_coordinates_1 = []
initial_coordinates_2 = []
initial_coordinates_3 = []

for i in range(prediction_horizon):
    if i == 0:
        for velocity in vector_v:
            for angle in vector_beta:
                initial_coordinates_1.append(iteration_of_predict(initial_coordinates_0, velocity, angle))
        print("first done")
    elif i == 1:
        t += delta_t
        for velocity in vector_v:
            for angle in vector_beta:
                for coordinates in initial_coordinates_1:
                    initial_coordinates_2.append(iteration_of_predict(coordinates, velocity, angle))
        print("second done")
    elif i == 2:
        t += delta_t
        for velocity in vector_v:
            for angle in vector_beta:
                for coordinates in initial_coordinates_2:
                    initial_coordinates_3.append(iteration_of_predict(coordinates, velocity, angle))
        print("third done")

# Plotting

for coordinates in initial_coordinates_3:
    predicted_vector_x.append(coordinates[0])
    predicted_vector_y.append(coordinates[1])


plt.figure(1)
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.scatter(predicted_vector_x, predicted_vector_y)
plt.show()


# plt.figure(1)
# plt.xlabel("Coordinate X")
# plt.ylabel("Coordinate Y")
# plt.plot(result_vector_x, result_vector_y)
#
# plt.figure(2)
# plt.xlabel("Time")
# plt.ylabel("Angle Phi")
# plt.plot(np.linspace(0, 100, np.size(result_vector_phi)), result_vector_phi)
#
# plt.show()
