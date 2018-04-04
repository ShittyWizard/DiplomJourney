from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps, delta_t, delta_beta, delta_v

# Actual
beta = 0
v = 0
phi = phi_0
x = x_0
y = y_0
coord_actual = [x, y]

t = 0

# Prediction horizon * delta_t = 1.5s
prediction_horizon = 3

# Generate vector from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v, v_max + delta_v, delta_v)
vector_v = np.round(vector_v, 3)
print(vector_v)

# Generate vector from minimal possible angle to maximum possible angle
vector_beta = np.arange(-beta_max, beta_max + delta_beta, delta_beta)
vector_beta = np.round(vector_beta, 3)
print(vector_beta)

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]

optimal_criterion = 0

# Stack with coordinates for optimal trajectory
optimal_trajectory = [0]


def is_on_target(actual_x, actual_y, target_x, target_y):
    if (target_x - actual_x) ^ 2 + (target_y - actual_y) ^ 2 <= eps:
        return True


# Function for getting distance from robot's actual position to line from initial position to target
def get_distance_from_line(x_a, y_a):
    distance = (abs((y_t - y_0) * x_a - (x_t - x_0) * y_a + x_t * y_0 - y_t * x_0)
                / (math.sqrt((y_t - y_0) ** 2 + (x_t - x_0) ** 2)))
    return distance


def get_distance_from_target(x_a, y_a):
    return math.sqrt((x_t - x_a) ** 2 + (y_t - y_a) ** 2)


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


def v_x(time, _velocity, _phi):
    return _velocity * cos(_phi)


def v_y(time, _velocity, _phi):
    return _velocity * sin(_phi)


def v_phi(time, _velocity, angle_beta):
    return (_velocity / L) * math.tan(angle_beta)


# Criterion for optimizing movement
def control_criterion(predicted_coordinates):
    angle_from_line = arctan(x_t / y_t) - predicted_coordinates[2]
    distance_from_target = get_distance_from_target(predicted_coordinates[0], predicted_coordinates[1])
    return distance_from_target ** 2 + 10 * angle_from_line ** 2


# Integration
def integrate_velocity(velocity_function, velocity_value, _phi, t_start, t_stop):
    return sp.quad(velocity_function, t_start, t_stop, args=(velocity_value, _phi))[0]


def integrate_angle(angle_function, velocity_value, angle_value, t_start, t_stop):
    return sp.quad(angle_function, t_start, t_stop, args=(velocity_value, angle_value))[0]


def coordinate_x(_v, _phi):
    return integrate_velocity(v_x, _v, _phi, t, t + delta_t)


def coordinate_y(_v, _phi):
    return integrate_velocity(v_y, _v, _phi, t, t + delta_t)


def angle_phi(_v, _beta):
    return integrate_angle(v_phi, _v, _beta, t, t + delta_t)


def iteration_of_predict(_initial_coordinates, _v, _angle):
    _phi = angle_phi(_v, _angle)
    _x = coordinate_x(_v, _initial_coordinates[2] + _phi)
    _y = coordinate_y(_v, _initial_coordinates[2] + _phi)
    return [_initial_coordinates[0] + _x, _initial_coordinates[1] + _y, _initial_coordinates[2] + _phi]


# while not is_on_target(x, y, x_t, y_t):
initial_coordinates_0 = [x, y, phi]
initial_coordinates_1 = []
initial_coordinates_2 = []
initial_coordinates_3 = []

for i in range(prediction_horizon):
    if i == 0:
        for velocity in vector_v:
            for angle in vector_beta:
                initial_coordinates_1.append(
                    iteration_of_predict(initial_coordinates_0, velocity, angle))
        print("First layer done")
    elif i == 1:
        t += delta_t
        for velocity in vector_v:
            for angle in vector_beta:
                for coordinates in initial_coordinates_1:
                    initial_coordinates_2.append(iteration_of_predict(coordinates, velocity, angle))
        print("Second layer done")
    elif i == 2:
        t += delta_t
        i = 0
        for velocity in vector_v:
            for angle in vector_beta:
                for coordinates in initial_coordinates_2:
                    initial_coordinates_3.append(iteration_of_predict(coordinates, velocity, angle))
                    if i == 0:
                        optimal_criterion = control_criterion(coordinates)
                        i += 1
                    if control_criterion(coordinates) < optimal_criterion:
                        optimal_trajectory.pop()
                        optimal_criterion = control_criterion(coordinates)
                        optimal_trajectory.append(coordinates)
        print("Third layer done")
        print("Criterion = " + str(optimal_criterion))
        print("Optimal trajectory = " + str(optimal_trajectory))

print([x_0, y_0], [x_t, y_t])

plt.figure(1)
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.grid()
plt.plot([x_0, x_t], [y_0, y_t])
plt.quiver(optimal_trajectory[0][0], optimal_trajectory[0][1], L * cos(optimal_trajectory[0][2]),
           L * sin(optimal_trajectory[0][2]))

plt.show()

# Plotting
# Generate vectors with coordinates for calculation of trajectory
# predicted_vector_x_1 = [x]
# predicted_vector_y_1 = [y]
# predicted_vector_phi_1 = [phi]
#
# for coordinates in initial_coordinates_1:
#     predicted_vector_x_1.append(coordinates[0])
#     predicted_vector_y_1.append(coordinates[1])
#     predicted_vector_phi_1.append(coordinates[2])
#
# predicted_vector_x_2 = [x]
# predicted_vector_y_2 = [y]
# predicted_vector_phi_2 = [phi]
#
# for coordinates in initial_coordinates_2:
#     predicted_vector_x_2.append(coordinates[0])
#     predicted_vector_y_2.append(coordinates[1])
#     predicted_vector_phi_2.append(coordinates[2])
#
# predicted_vector_x_3 = [x]
# predicted_vector_y_3 = [y]
# predicted_vector_phi_3 = [phi]
#
# for coordinates in initial_coordinates_3:
#     predicted_vector_x_3.append(coordinates[0])
#     predicted_vector_y_3.append(coordinates[1])
#     predicted_vector_phi_3.append(coordinates[2])
#
# plt.figure(1)
# plt.xlabel("Coordinate X")
# plt.ylabel("Coordinate Y")
# plt.grid()
# plt.scatter(predicted_vector_x_3, predicted_vector_y_3, 2, linewidths=2, c='blue', edgecolors='blue',
#             label=r'Possible coordinates for 3 next steps(1,5s)')
# plt.scatter(predicted_vector_x_2, predicted_vector_y_2, 2, linewidths=8, c='green', edgecolors='green',
#             label=r'Possible coordinates for 2 next steps(1,0s)')
# plt.scatter(predicted_vector_x_1, predicted_vector_y_1, 2, linewidths=8, c='red', edgecolors='red',
#             label=r'Possible coordinates for 1 next step(0,5s)')
#
# plt.legend(fontsize=15)  # legend(loc='upper left')
# plt.title(
#     r'$\varphi_0 = \pi/2' + ', ' + r'\beta_{max} = \pi /6' + ', ' + r'L = 0.5m' + ', ' +
#     r'r = 0.05m' + ', ' + r'V_{max} = 1m/s$', fontsize=20)

plt.show()
