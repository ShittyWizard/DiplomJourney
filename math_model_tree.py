import time

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp
import sys as sys
from CoordinateTree import CoordinateTree
from config import phi_0, L, y_t, y_0, x_t, x_0, beta_max, v_max, eps, delta_t, delta_beta, delta_v, v_acc_max, \
    beta_acc_max

np.set_printoptions(threshold=np.nan)

# Actual
beta = 0
v = 0
phi = phi_0
x = x_0
y = y_0

# Prediction horizon * delta_t = 1.5s
prediction_horizon = 3

# Generate vectors with coordinates for plotting
result_vector_x = [x]
result_vector_y = [y]
result_vector_phi = [phi]

# Radius of U-turn
radius_u_turn = L / sin(beta_max)


# Function which return True/False (Is on target?) and distance from target
def is_on_target(actual_x, actual_y, target_x, target_y):
    if (target_x - actual_x) ** 2 + (target_y - actual_y) ** 2 <= eps:
        return [True, (target_x - actual_x) ** 2 + (target_y - actual_y) ** 2]
    else:
        return [False, (target_x - actual_x) ** 2 + (target_y - actual_y) ** 2]


# Function for getting distance from robot's actual position to line from initial position to target
def get_distance_from_line(x_a, y_a):
    if x_a == x_0 and y_a == y_0:
        distance = 1000
    else:
        distance = (abs((y_t - y_0) * x_a - (x_t - x_0) * y_a + x_t * y_0 - y_t * x_0)
                    / (math.sqrt((y_t - y_0) ** 2 + (x_t - x_0) ** 2)))
    return distance ** 2


def get_distance_from_target(x_a, y_a):
    return math.sqrt((x_t - x_a) ** 2 + (y_t - y_a) ** 2)


def v_x(time, _velocity, _phi):
    return _velocity * cos(_phi)


def v_y(time, _velocity, _phi):
    return _velocity * sin(_phi)


def v_phi(time, _velocity, angle_beta):
    return (_velocity / L) * math.tan(angle_beta)


# Criterion for optimizing movement
def control_criterion(predicted_coordinates):
    distance_from_target = get_distance_from_target(predicted_coordinates[0], predicted_coordinates[1])
    distance_from_line = get_distance_from_line(predicted_coordinates[0], predicted_coordinates[1])
    return 10000 * distance_from_target + 10000 * distance_from_line


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
    return [_global_coordinates[0] + _x, _global_coordinates[1] + _y, _global_coordinates[2] + _phi, _v, _angle]


def new_target(actual_x, actual_y, actual_phi, target_x, target_y):
    global x_t, y_t, x_0, y_0, phi_0
    print("Previous target: " + str(x_t) + " " + str(y_t))
    x_t = target_x
    y_t = target_y
    x_0 = actual_x
    y_0 = actual_y
    phi_0 = actual_phi
    plot_from_actual_to_target(actual_x, actual_y, actual_phi, target_x, target_y)
    print("New target: " + str(x_t) + " " + str(y_t))


def plot_from_actual_to_target(initial_x, initial_y, initial_phi, target_x, target_y):
    # Line from initial point to 1st target
    plt.plot([initial_x, target_x], [initial_y, target_y], 'b', linewidth=2, alpha=0.5)
    plt.plot([initial_x, target_x], [initial_y, target_y], 'bo')
    # Initial position
    plt.quiver(initial_x, initial_y, cos(initial_phi), sin(initial_phi), pivot='middle')


def turn_left(actual_x, actual_y, actual_phi, distance, actual_velocity):
    global v
    if math.pi / 2 <= actual_phi <= 3 * math.pi / 2:
        if actual_phi <= math.pi:
            # \ |
            temp_phi = actual_phi - math.pi / 2
            target_x = actual_x - distance * cos(temp_phi) - radius_u_turn * sin(temp_phi)
            target_y = actual_y - distance * sin(temp_phi) + radius_u_turn * cos(temp_phi)
        else:
            # / |
            temp_phi = actual_phi - math.pi
            target_x = actual_x + distance * sin(temp_phi) - radius_u_turn * cos(temp_phi)
            target_y = actual_y - distance * cos(temp_phi) - radius_u_turn * sin(temp_phi)
    else:
        if actual_phi <= 2 * math.pi:
            # | \
            temp_phi = actual_phi - 3 * math.pi / 2
            target_x = actual_x + distance * cos(temp_phi) + radius_u_turn * sin(temp_phi)
            target_y = actual_y + distance * sin(temp_phi) - radius_u_turn * cos(temp_phi)
        else:
            # | /
            temp_phi = actual_phi
            target_x = actual_x - distance * sin(temp_phi) + radius_u_turn * cos(temp_phi)
            target_y = actual_y + distance * cos(temp_phi) + radius_u_turn * sin(temp_phi)
    delta_teta = math.pi / 2
    print("Delta teta: " + str(delta_teta))
    v = slow_down(actual_velocity, delta_teta)
    new_target(actual_x, actual_y, actual_phi, target_x, target_y)
    print("Turning left...")
    print()


def turn_right(actual_x, actual_y, actual_phi, distance, actual_velocity):
    global v
    if math.pi / 2 <= actual_phi <= 3 * math.pi / 2:
        if actual_phi <= math.pi:
            # \ |
            temp_phi = actual_phi - math.pi / 2
            target_x = actual_x + distance * cos(temp_phi) - radius_u_turn * sin(temp_phi)
            target_y = actual_y + distance * sin(temp_phi) + radius_u_turn * cos(temp_phi)
        else:
            # / |
            temp_phi = actual_phi - math.pi
            target_x = actual_x - distance * sin(temp_phi) - radius_u_turn * cos(temp_phi)
            target_y = actual_y + distance * cos(temp_phi) - radius_u_turn * sin(temp_phi)
    else:
        if actual_phi <= 2 * math.pi:
            # | \
            temp_phi = actual_phi - 3 * math.pi / 2
            target_x = actual_x - distance * cos(temp_phi) + radius_u_turn * sin(temp_phi)
            target_y = actual_y - distance * sin(temp_phi) - radius_u_turn * cos(temp_phi)
        else:
            # | /
            temp_phi = actual_phi
            target_x = actual_x + distance * sin(temp_phi) + radius_u_turn * cos(temp_phi)
            target_y = actual_y - distance * cos(temp_phi) + radius_u_turn * sin(temp_phi)
    delta_teta = math.pi / 2
    print("Delta teta: " + str(delta_teta))
    v = slow_down(actual_velocity, delta_teta)
    new_target(actual_x, actual_y, actual_phi, target_x, target_y)
    print("Turning right...")
    print()


# For future update with autonomnous avoiding barriers (now it handles by operator)
def move_forward(actual_x, actual_y, actual_phi, distance):
    target_x = actual_x + distance * cos(actual_phi)
    target_y = actual_y + distance * sin(actual_phi)
    new_target(actual_x, actual_y, actual_phi, target_x, target_y)


# Angle teta from 0 to 90 degrees
def slow_down(actual_velocity, delta_teta):
    vector_velocities = vector_of_velocities(actual_velocity)
    temp = 10
    if abs(delta_teta) < math.radians(10):
        velocity = actual_velocity
    else:
        velocity = actual_velocity * (1 - delta_teta * (4 / 3 * math.pi))
        print("New velocity: " + str(velocity))
        for v_ in vector_velocities:
            if abs(v_ - velocity) < temp:
                velocity = v_
                temp = abs(v_ - velocity)
    return velocity


def vector_of_velocities(actual_velocity):
    vector_velocities = []
    for i in range(1 + 2 * int((v_acc_max * delta_t) / delta_v)):
        possible_velocity = actual_velocity + delta_v * (i - (v_acc_max * delta_t) / delta_v)
        if (not possible_velocity < 0) and (not possible_velocity > v_max):
            vector_velocities.append(possible_velocity)
    return vector_velocities


def vector_of_beta_angles(actual_beta):
    vector_beta_angles = []
    for i in range(1 + 2 * int((math.degrees(beta_acc_max) * delta_t) / math.degrees(delta_beta))):
        possible_angle = actual_beta + delta_beta * (
                i - (math.degrees(beta_acc_max) * delta_t) / math.degrees(delta_beta))
        if abs(possible_angle) > beta_max:
            print(".")
        else:
            vector_beta_angles.append(possible_angle)
    return vector_beta_angles


def predictive_control(_initial_x, _initial_y, _initial_phi, _target_x, _target_y, _vector_v,
                       _vector_beta):
    global optimal_trajectory, optimal_criterion
    global result_trajectory_x, result_trajectory_y, result_trajectory_phi
    global t, m, time_arr_for_plotting
    global result_v, result_beta
    global recursive, need_scatter
    global result_trajectory_v, result_trajectory_beta, result_trajectory_angle_speed

    size_max_1 = size(_vector_beta) * size(_vector_v)
    size_max_2 = size_max_1 * size_max_1
    size_max_3 = size_max_1 * size_max_1 * size_max_1

    initial_coordinates = [_initial_x, _initial_y, _initial_phi]
    global_coordinates = CoordinateTree(size_max_1)

    result_x = 0
    result_y = 0
    result_phi = 0
    field_x_1 = []
    field_y_1 = []

    field_x_2 = []
    field_y_2 = []

    field_x_3 = []
    field_y_3 = []

    t += delta_t
    time_arr_for_plotting.append(t)
    start = time.time()
    for i in range(prediction_horizon):
        if i == 0:
            j = 0
            for velocity in _vector_v:
                for angle in _vector_beta:
                    temp0 = iteration_of_predict(initial_coordinates, velocity, angle)
                    global_coordinates[j] = temp0
                    field_x_1.append(temp0[0])
                    field_y_1.append(temp0[1])
                    j += 1
            print("First layer done. Time = " + str(time.time() - start))
        elif i == 1:
            j = size_max_1
            for velocity in _vector_v:
                for angle in _vector_beta:
                    temp1 = iteration_of_predict(global_coordinates[(j - size_max_1) % size_max_1],
                                                 velocity, angle)
                    global_coordinates[j] = temp1
                    field_x_2.append(temp1[0])
                    field_y_2.append(temp1[1])
                    j += 1
            print("Second layer done.Time = " + str(time.time() - start))
        elif i == 2:
            j = size_max_1 + size_max_2
            for velocity in _vector_v:
                for angle in _vector_beta:
                    temp2 = iteration_of_predict(
                        global_coordinates[size_max_1 + ((j - (size_max_1 + size_max_2)) % size_max_1)],
                        velocity, angle)
                    global_coordinates[j] = temp2
                    field_x_3.append(temp2[0])
                    field_y_3.append(temp2[1])
                    if control_criterion(temp2) < optimal_criterion:
                        optimal_trajectory.pop()
                        optimal_trajectory.append([global_coordinates[global_coordinates.get_index_of_parent(j)[1]],
                                                   global_coordinates[global_coordinates.get_index_of_parent(j)[0]],
                                                   global_coordinates[j]])
                        result_v = velocity
                        result_trajectory_v.append(velocity)
                        result_beta = angle
                        optimal_criterion = control_criterion(temp2)
                    j += 1
            print("Third layer done.Time = " + str(time.time() - start))
            print("Absolute time = " + str(t))
            print()

            predicted_trajectory_x = [optimal_trajectory[0][0][0], optimal_trajectory[0][1][0],
                                      optimal_trajectory[0][2][0]]
            # predicted_trajectory_x_anim0.append(predicted_trajectory_x[0])
            # predicted_trajectory_x_anim1.append(predicted_trajectory_x[1])
            # predicted_trajectory_x_anim2.append(predicted_trajectory_x[2])

            predicted_trajectory_y = [optimal_trajectory[0][0][1], optimal_trajectory[0][1][1],
                                      optimal_trajectory[0][2][1]]
            # predicted_trajectory_y_anim0.append(predicted_trajectory_y[0])
            # predicted_trajectory_y_anim1.append(predicted_trajectory_y[1])
            # predicted_trajectory_y_anim2.append(predicted_trajectory_y[2])

            predicted_trajectory_phi = [optimal_trajectory[0][0][2], optimal_trajectory[0][1][2],
                                        optimal_trajectory[0][2][2]]
            # predicted_trajectory_phi_anim0.append(predicted_trajectory_phi[0])
            # predicted_trajectory_phi_anim1.append(predicted_trajectory_phi[1])
            # predicted_trajectory_phi_anim2.append(predicted_trajectory_phi[2])

            result_x = predicted_trajectory_x[0]
            result_y = predicted_trajectory_y[0]
            result_phi = predicted_trajectory_phi[0]

            if m == 2:
                print("Predicted trajectory is in target! x4")
                result_x = predicted_trajectory_x[2]
                result_y = predicted_trajectory_y[2]
                result_phi = predicted_trajectory_phi[2]
                print("Distance from target: " + str(
                    is_on_target(predicted_trajectory_x[1], predicted_trajectory_y[1], x_t, y_t)[1]))
            elif m == 1:
                print("Predicted trajectory is in target! x3")
                result_x = predicted_trajectory_x[1]
                result_y = predicted_trajectory_y[1]
                result_phi = predicted_trajectory_phi[1]
                print("Distance from target: " + str(
                    is_on_target(predicted_trajectory_x[1], predicted_trajectory_y[1], x_t, y_t)[1]))
                m += 1
            elif is_on_target(predicted_trajectory_x[2], predicted_trajectory_y[2], x_t, y_t)[0]:
                print("Predicted trajectory is in target! x2")
                result_x = predicted_trajectory_x[0]
                result_y = predicted_trajectory_y[0]
                result_phi = predicted_trajectory_phi[0]
                print("Distance from target: " + str(
                    is_on_target(predicted_trajectory_x[1], predicted_trajectory_y[1], x_t, y_t)[1]))
                m += 1

            result_trajectory_x.append(result_x)
            result_trajectory_y.append(result_y)
            result_trajectory_phi.append(result_phi)
            result_trajectory_v.append(result_v)
            result_trajectory_beta.append(result_beta)
            result_trajectory_angle_speed.append((result_v / L) * tan(result_beta))

            print()
            print("Now I'm here - x : " + str(result_x) + ' y: ' + str(result_y) + ' v: ' + str(
                result_v) + ' beta: ' + str(math.degrees(result_beta)))
            print()
    optimal_criterion = sys.maxsize
    return [result_x, result_y, result_phi, result_v, result_beta]


def add_plot_polygon(coordinates_of_vertexes):
    barrier_polygon = Polygon(coordinates_of_vertexes, fill=False, hatch='//',
                              edgecolor='black', linewidth='1')
    plt.axes().add_patch(barrier_polygon)


""""
 Initial coordinates format: [initial_x, initial_y, initial_phi, initial_v, initial_beta]
 Target coordinates format: [target_x, target_y]
 Vector v : vector with possible velocities for this iteration
 Vector beta : vector with possible angles for this iteration
"""

need_scatter = False


def math_mpc(initial_coordinates, target_coordinates):
    global x, y, phi, x_t, y_t, v, beta, recursive, need_scatter, p, time_arr_for_plotting
    global result_x_velocity, result_x_acceleration
    global result_y_velocity, result_y_acceleration
    global result_trajectory_v, result_trajectory_beta, result_trajectory_angle_speed
    print("Start MPC modelling... ")
    print()
    p = 1
    coordinates = initial_coordinates
    x = coordinates[0]
    y = coordinates[1]
    phi = coordinates[2]
    v_0 = coordinates[3]
    v = v_0
    beta = initial_coordinates[4]

    x_t = target_coordinates[0]
    y_t = target_coordinates[1]

    x_previous = coordinates[0]
    y_previous = coordinates[1]

    recursive = False

    plot_from_actual_to_target(x_0, y_0, phi_0, x_t, y_t)

    while not is_on_target(x, y, x_t, y_t)[0]:
        vector_velocities = vector_of_velocities(v)
        vector_beta_angles = vector_of_beta_angles(beta)
        previous_v = v

        coordinates = predictive_control(x, y, phi, x_t, y_t, vector_velocities, vector_beta_angles)

        x = coordinates[0]
        y = coordinates[1]
        phi = coordinates[2]
        v = coordinates[3]
        beta = coordinates[4]

        result_x_velocity.append(v * cos(phi))
        result_x_acceleration.append(((v - previous_v) / delta_t) * cos(phi))

        result_y_velocity.append(v * sin(phi))
        result_y_acceleration.append(((v - previous_v) / delta_t) * sin(phi))

        need_scatter = False

        if recursive:
            print("Recursive error.")
            break
        elif x == x_previous and y == y_previous:
            recursive = True
        if p == 40:
            turn_left(x, y, phi, 2, v)
        if p == 70:
            turn_right(x, y, phi, 2, v)
        if p == 90:
            new_target(x, y, phi, -3, -3)
        x_previous = x
        y_previous = y
        print("Iteration number = " + str(p))
        print()
        p += 1

    print("MPC modelling has finished. Waiting for plots...")
    print()

    print("Plots are ready.")


t = 0
dt = delta_t
time_arr_for_plotting = [0]

# Stack with coordinates for optimal trajectory
optimal_trajectory = [0]

result_trajectory_phi = [phi_0]

result_trajectory_x = [x_0]
result_x_velocity = [0]
result_x_acceleration = [0]

result_trajectory_y = [y_0]
result_y_velocity = [0]
result_y_acceleration = [0]

result_trajectory_v = [0]
result_trajectory_beta = [0]
result_trajectory_angle_speed = [0]

optimal_criterion = control_criterion([x_0, y_0, phi_0])
result_v = 0
result_beta = 0

m = 0  # For optimizing finishing

# For animation
# predicted_trajectory_x_anim0 = []
# predicted_trajectory_y_anim0 = []
# predicted_trajectory_phi_anim0 = []
#
# predicted_trajectory_x_anim1 = []
# predicted_trajectory_y_anim1 = []
# predicted_trajectory_phi_anim1 = []
#
# predicted_trajectory_x_anim2 = []
# predicted_trajectory_y_anim2 = []
# predicted_trajectory_phi_anim2 = []

# Plotting
fig1 = plt.figure(1)
fig1.set_dpi(100)

ax1 = plt.axes()
ax1.set_aspect(1)
plt.grid()
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.title(r'$\beta_{max} = $' + str(beta_max) + '  ' + r'$v_{max} = $' + str(v_max) + '  ' + r'$\varphi_0 = $' + str(
    phi_0) + ' ' + r'$ L = $' + str(L))
xdata, ydata = [], []

# MODELLING!
math_mpc([0, 0, math.pi * 5 / 6, 0, 0], [-2, -2])

add_plot_polygon([[-0.5, -1], [-1, -1.9], [-2, -2.2], [-3, -2], [-2, -0.5], [-0.5, -1]])

plt.plot(result_trajectory_x, result_trajectory_y, 'r', linewidth=1)

# -------------------------------------------------------------------

# TODO: Constraints on plots for V_MAX, BETA_MAX, ANGLE_SPEED_MAX

# Plots for X coordinate
fig2 = plt.figure(2)
fig2.set_dpi(100)

ax2 = plt.axes()
ax2.set_aspect(1)
plt.grid()

x_coord = plt.subplot(311)
x_coord.set_xlabel("Time")
x_coord.set_ylabel("Coordinate X")
# plt.plot(time_arr_for_plotting, result_trajectory_x, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_trajectory_x, 'r', linewidth=1)

x_velocity = plt.subplot(312)
x_velocity.set_xlabel("Time")
x_velocity.set_ylabel("X-Axis speed")
# plt.plot(time_arr_for_plotting, result_x_velocity, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_x_velocity, 'r', linewidth=1)
plt.plot(v_max)

x_acceleration = plt.subplot(313)
x_acceleration.set_xlabel("Time")
x_acceleration.set_ylabel("X-Axis acceleration")
# plt.plot(time_arr_for_plotting, result_x_acceleration, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_x_acceleration, 'r', linewidth=1)
plt.plot(v_acc_max)

# ---------------------------------------------------------------------

# Plots for Y coordinate
fig3 = plt.figure(3)
fig3.set_dpi(100)

ax3 = plt.axes()
ax3.set_aspect(1)
ax3.grid()

y_coord = plt.subplot(311)
y_coord.set_xlabel("Time")
y_coord.set_ylabel("Coordinate Y")
# plt.plot(time_arr_for_plotting, result_trajectory_y, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_trajectory_y, 'r', linewidth=1)

y_velocity = plt.subplot(312)
y_velocity.set_xlabel("Time")
y_velocity.set_ylabel("Y-Axis speed")
# plt.plot(time_arr_for_plotting, result_y_velocity, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_y_velocity, 'r', linewidth=1)
plt.plot(v_max)

y_acceleration = plt.subplot(313)
y_acceleration.set_xlabel("Time")
y_acceleration.set_ylabel("Y-Axis acceleration")
# plt.plot(time_arr_for_plotting, result_y_acceleration, 'ro', linewidth=1)
plt.plot(time_arr_for_plotting, result_y_acceleration, 'r', linewidth=1)
plt.plot(v_acc_max)

# -----------------------------------------------------------------------

# Plot for velocity and "beta" angle

fig4 = plt.figure(4)
fig4.set_dpi(100)

ax4 = plt.axes()
ax4.set_aspect(1)
ax4.grid()

velocity = plt.subplot(311)
velocity.set_xlabel("Time")
velocity.set_ylabel("Speed")
# plt.plot(time_arr_for_plotting, result_trajectory_v, 'r', linewidth=1)
plt.plot(time_arr_for_plotting, v_max, 'b-', linewidth=1)

beta_plot = plt.subplot(312)
beta_plot.set_xlabel("Time")
beta_plot.set_ylabel("Wheel turning angle")
plt.plot(time_arr_for_plotting, result_trajectory_beta, 'r', linewidth=1)
plt.plot(beta_max)

angle_speed = plt.subplot(313)
angle_speed.set_xlabel("Time")
angle_speed.set_ylabel("Angle speed")
plt.plot(time_arr_for_plotting, result_trajectory_angle_speed, 'r', linewidth=1)
plt.plot((v_max / L) * tan(beta_max))

# ----------------------------------------------------------------------
plt.show()
"""
ANIMATION
"""
# predicted_position, = plt.plot([], [], 'go', animated=True)
# predicted_line, = plt.plot([], [], 'g', animated=True)
# previous_position, = plt.plot([], [], 'co', animated=True)
# previous_line, = plt.plot([], [], 'c', animated=True)
# main_pos, = plt.plot([], [], 'ro', animated=True)
# plt.plot(result_trajectory_x, result_trajectory_y, 'r', linewidth=1)
#
#
# # Plot barrier (Polygon)
# # add_plot_polygon([[-1.8, -2], [-2, -3], [-3, -4], [-5, -3], [-3, -1], [-1.8, -2]])
#
# def init():
#     return main_pos, predicted_position, predicted_line, previous_position, previous_line,
#
#
# def animate(i):
#     main_pos.set_data(result_trajectory_x[i], result_trajectory_y[i])
#     predicted_position.set_data(
#         [predicted_trajectory_x_anim0[i], predicted_trajectory_x_anim1[i], predicted_trajectory_x_anim2[i]],
#         [predicted_trajectory_y_anim0[i], predicted_trajectory_y_anim1[i], predicted_trajectory_y_anim2[i]])
#     predicted_line.set_data(
#         [result_trajectory_x[i], predicted_trajectory_x_anim0[i], predicted_trajectory_x_anim1[i],
#          predicted_trajectory_x_anim2[i]],
#         [result_trajectory_y[i], predicted_trajectory_y_anim0[i], predicted_trajectory_y_anim1[i],
#          predicted_trajectory_y_anim2[i]])
#     if i >= 2:
#         previous_position.set_data([result_trajectory_x[i - 2], result_trajectory_x[i - 1]],
#                                    [result_trajectory_y[i - 2], result_trajectory_y[i - 1]])
#     previous_line.set_data([result_trajectory_x[i - 2], result_trajectory_x[i - 1]],
#                            [result_trajectory_y[i - 2], result_trajectory_y[i - 1]])
#
#     return predicted_position, predicted_line, previous_position, previous_line, main_pos,
#
#
# plot_from_actual_to_target(x_0, y_0, phi_0, x_t, y_t)
# ani = FuncAnimation(fig, animate, p, init_func=init, blit=True, repeat=True, repeat_delay=100)
# # plt.show()
# ani.save('animation_2.gif', writer='imagemagick', dpi=100, fps=20)
