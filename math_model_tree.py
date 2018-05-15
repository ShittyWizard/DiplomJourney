import time

from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from CoordinateTree import CoordinateTree
from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps, delta_t, delta_beta, delta_v

np.set_printoptions(threshold=np.nan)

# Actual
beta = 0
v = 0
phi = phi_0
x = x_0
y = y_0
coord_actual = [x, y]

# Prediction horizon * delta_t = 1.5s
prediction_horizon = 3

# Generate vector from minimal possible velocity to maximum possible velocity
vector_v = np.arange(v, v_max + delta_v, delta_v)
vector_v = np.round(vector_v, 3)
print("Vector of velocities - " + str(vector_v))
print()

# Generate vector from minimal possible angle to maximum possible angle
vector_beta = np.arange(-beta_max, beta_max + delta_beta, delta_beta)
vector_beta = np.round(vector_beta, 3)
print("Vector of beta angles - " + str(vector_beta))
print()

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]


def is_on_target(actual_x, actual_y, target_x, target_y):
    if (target_x - actual_x) ** 2 + (target_y - actual_y) ** 2 <= eps:
        return True
    else:
        return False


# Function for getting distance from robot's actual position to line from initial position to target
def get_distance_from_line(x_a, y_a):
    if x_a == x_0 and y_a == y_0:
        distance = 1000
    else:
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


def v_x(time, _velocity, _phi):
    return _velocity * cos(_phi)


def v_y(time, _velocity, _phi):
    return _velocity * sin(_phi)


def v_phi(time, _velocity, angle_beta):
    return (_velocity / L) * math.tan(angle_beta)


# Criterion for optimizing movement
def control_criterion(predicted_coordinates):
    angle_from_line = (arctan(x_t / y_t) - predicted_coordinates[2])
    distance_from_target = get_distance_from_target(predicted_coordinates[0], predicted_coordinates[1])
    distance_from_line = get_distance_from_line(predicted_coordinates[0], predicted_coordinates[1])
    return 1000 * distance_from_target + 10 * angle_from_line ** 2 + 10 * distance_from_line ** 2


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


def iteration_of_predict(_global_coordinates, _v, _angle):
    _phi = angle_phi(_v, _angle)
    _x = coordinate_x(_v, _global_coordinates[2] + _phi)
    _y = coordinate_y(_v, _global_coordinates[2] + _phi)
    return [_global_coordinates[0] + _x, _global_coordinates[1] + _y, _global_coordinates[2] + _phi]


size_max_1 = size(vector_beta) * size(vector_v)
size_max_2 = pow(size_max_1, 2)
size_max_3 = pow(size_max_1, 3)

plt.figure(1)
plt.grid()
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.title(r'$\beta_{max} = $' + str(beta_max) + '  ' + r'$\varphi_0 = $' + str(phi_0) + ' ')

t = 0

plt.plot([x_0, x_t], [y_0, y_t], 'b', linewidth=3)

# Stack with coordinates for optimal trajectory
optimal_trajectory = [0]
optimal_criterion = control_criterion([x_0, y_0, phi_0])
result_v = 0
result_beta = 0


def predictive_control(_initial_x, _initial_y, _initial_phi, _initial_velocity, _target_x, _target_y):
    global optimal_trajectory
    global optimal_criterion
    global t
    global result_v
    global result_beta

    initial_coordinates = [_initial_x, _initial_y, _initial_phi]
    global_coordinates = CoordinateTree(size_max_1)

    result_x = 0
    result_y = 0
    result_phi = 0
    field_x = []
    field_y = []
    t += delta_t
    start = time.time()
    for i in range(prediction_horizon):
        if i == 0:
            j = 0
            for velocity in vector_v:
                for angle in vector_beta:
                    temp0 = iteration_of_predict(initial_coordinates, velocity, angle)
                    global_coordinates[j] = temp0
                    j += 1
            print("First layer done. Time = " + str(time.time() - start))
        elif i == 1:
            j = size_max_1
            for velocity in vector_v:
                for angle in vector_beta:
                    temp1 = iteration_of_predict(global_coordinates[(j - size_max_1) // size_max_1],
                                                 velocity, angle)
                    global_coordinates[j] = temp1
                j += 1
            print("Second layer done.Time = " + str(time.time() - start))
        elif i == 2:
            j = size_max_1 + size_max_2
            for velocity in vector_v:
                for angle in vector_beta:
                    temp2 = iteration_of_predict(
                        global_coordinates[size_max_1 + ((j - (size_max_1 + size_max_2)) // size_max_1)],
                        velocity, angle)
                    global_coordinates[j] = temp2
                    field_x.append(temp2[0])
                    field_y.append(temp2[1])
                    if control_criterion(temp2) < optimal_criterion:
                        optimal_trajectory.pop()
                        optimal_trajectory.append([global_coordinates[global_coordinates.get_index_of_parent(j)[1]],
                                                   global_coordinates[global_coordinates.get_index_of_parent(j)[0]],
                                                   global_coordinates[j]])
                        result_v = velocity
                        result_beta = angle
                        optimal_criterion = control_criterion(temp2)
                    j += 1
            print("Third layer done.Time = " + str(time.time() - start))
            print("Absolute time = " + str(t))
            print()

            plt.scatter(field_x, field_y, color='g', alpha=0.3)

            plt.quiver(_initial_x, _initial_y, L * cos(_initial_phi), L * sin(_initial_phi), pivot='middle')
            plt.quiver(optimal_trajectory[0][0][0], optimal_trajectory[0][0][1], L * cos(optimal_trajectory[0][0][2]),
                       L * sin(optimal_trajectory[0][0][2]), pivot='middle', alpha=0.2)

            plt.quiver(optimal_trajectory[0][1][0], optimal_trajectory[0][1][1], L * cos(optimal_trajectory[0][1][2]),
                       L * sin(optimal_trajectory[0][1][2]), pivot='middle', alpha=0.2)

            plt.quiver(optimal_trajectory[0][2][0], optimal_trajectory[0][2][1], L * cos(optimal_trajectory[0][2][2]),
                       L * sin(optimal_trajectory[0][2][2]), pivot='middle', alpha=0.2)

            result_trajectory_x = [optimal_trajectory[0][0][0], optimal_trajectory[0][1][0],
                                   optimal_trajectory[0][2][0]]
            result_trajectory_y = [optimal_trajectory[0][0][1], optimal_trajectory[0][1][1],
                                   optimal_trajectory[0][2][1]]
            result_trajectory_phi = [optimal_trajectory[0][0][2], optimal_trajectory[0][1][2],
                                     optimal_trajectory[0][2][2]]

            result_x = result_trajectory_x[2]
            result_y = result_trajectory_y[2]
            result_phi = result_trajectory_phi[2]

            plt.plot(
                [_initial_x, result_trajectory_x[1], result_trajectory_x[2]],
                [_initial_y, result_trajectory_y[1], result_trajectory_y[2]],
                'r',
                linewidth=3)

            print("Now I'm here - x : " + str(result_x) + ' y: ' + str(result_y))
            print("Distance from line: " + str(get_distance_from_line(result_x, result_y)))
            print()
    return [result_x, result_y, result_phi, result_v, result_beta]


p = 1
coordinates = [x_0, y_0, phi_0, v, beta]
k = 0
x_previous = coordinates[0]
y_previous = coordinates[1]
while not is_on_target(x, y, x_t, y_t):
    coordinates = predictive_control(x, y, phi, v, x_t, y_t)
    x = coordinates[0]
    y = coordinates[1]
    phi = coordinates[2]
    v = coordinates[3]
    beta = coordinates[4]
    if x == x_previous and y == y_previous:
        k += 1
    if k == 2:
        print("Recursive error")
        break
    x_previous = x
    y_previous = y
    print("Iteration number = " + str(p))
    p += 1

plt.show()
