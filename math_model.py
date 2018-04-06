import time

from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps, delta_t, delta_beta, delta_v

np.set_printoptions(threshold=np.nan)

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

# Generate vector from minimal possible velocity to mpltimum possible velocity
vector_v = np.arange(v, v_max + delta_v, delta_v)
vector_v = np.round(vector_v, 3)
print("Vector of velocities - " + str(vector_v))
print()

# Generate vector from minimal possible angle to mpltimum possible angle
vector_beta = np.arange(-beta_max, beta_max + delta_beta, delta_beta)
vector_beta = np.round(vector_beta, 3)
print("Vector of beta angles - " + str(vector_beta))
print()

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]

# Stack with coordinates for optimal trajectory
optimal_trajectory = [0]
optimal_criterion = 1000000


def is_on_target(actual_x, actual_y, target_x, target_y):
    if (target_x - actual_x) ^ 2 + (target_y - actual_y) ^ 2 <= eps:
        return True


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


def saturation(value, value_mplt):
    if value > value_mplt:
        value = value_mplt
    elif value < -value_mplt:
        value = -value_mplt
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
    print("Angle_crit = " + str(angle_from_line ** 2), "Line_crit = " + str(distance_from_line ** 2),
          "Target_crit = " + str(distance_from_target ** 2))
    return 10*distance_from_target ** 2 + 10 * angle_from_line ** 2 + 10 * distance_from_line ** 2


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


initial_coordinates = [x, y, phi]

size_mplt_1 = size(vector_beta) * size(vector_v)
size_mplt_2 = pow(size_mplt_1, 2)
size_mplt_3 = pow(size_mplt_1, 3)

global_coordinates = np.empty([size_mplt_3, 3], tuple)

start = time.time()
for i in range(prediction_horizon):
    if i == 0:
        j = 0
        m = 0
        while j < size_mplt_3:
            for velocity in vector_v:
                for angle in vector_beta:
                    temp0 = iteration_of_predict(initial_coordinates, velocity, angle)
                    for k in range(size_mplt_2):
                        global_coordinates[j + k][0] = np.array(temp0)
                    j += size_mplt_2
        print("First layer done. Time = " + str(time.time() - start))
    elif i == 1:
        j = 0
        t += delta_t
        while j < size_mplt_3:
            for velocity in vector_v:
                for angle in vector_beta:
                    temp1 = iteration_of_predict(global_coordinates[j][0], velocity, angle)
                    for k in range(size_mplt_1):
                        global_coordinates[j + k][1] = np.array(temp1)
                    j += size_mplt_1
        print("Second layer done.Time = " + str(time.time() - start))
    elif i == 2:
        j = 0
        t += delta_t
        while j < size_mplt_3:
            for velocity in vector_v:
                for angle in vector_beta:
                    temp2 = iteration_of_predict(global_coordinates[j][1], velocity, angle)
                    global_coordinates[j][2] = np.array(temp2)
                    if control_criterion(temp2) < optimal_criterion:
                        optimal_trajectory.pop()
                        optimal_trajectory.append(global_coordinates[j])
                        optimal_criterion = control_criterion(temp2)
                    j += 1
        print("Third layer done.Time = " + str(time.time() - start))
print("Optimal_criterion = " + str(optimal_criterion))
print("Optimal trajectory = " + str(optimal_trajectory))

plt.figure(1)

ticks_x = np.arange(0, x_t + 0.1, 0.1)
ticks_y = np.arange(0, y_t + 0.1, 0.1)

plt.xticks(ticks_x)
plt.yticks(ticks_y)

# And a corresponding grid
plt.grid(which='both')

plt.plot([x_0, x_t], [y_0, y_t])
plt.quiver(x_0, y_0, L * cos(phi_0), L * sin(phi_0), pivot='middle')
plt.quiver(optimal_trajectory[0][0][0], optimal_trajectory[0][0][1], L * cos(optimal_trajectory[0][0][2]),
           L * sin(optimal_trajectory[0][0][2]), pivot='middle')

plt.quiver(optimal_trajectory[0][1][0], optimal_trajectory[0][1][1], L * cos(optimal_trajectory[0][1][2]),
           L * sin(optimal_trajectory[0][1][2]), pivot='middle')

plt.quiver(optimal_trajectory[0][2][0], optimal_trajectory[0][2][1], L * cos(optimal_trajectory[0][2][2]),
           L * sin(optimal_trajectory[0][2][2]), pivot='middle')

plt.show()

# plt.legend(fontsize=15)  # legend(loc='upper left')
# plt.title(
#     r'$\varphi_0 = \pi/2' + ', ' + r'\beta_{mplt} = \pi /6' + ', ' + r'L = 0.5m' + ', ' +
#     r'r = 0.05m' + ', ' + r'V_{mplt} = 1m/s$', fontsize=20)
#
# plt.show()
