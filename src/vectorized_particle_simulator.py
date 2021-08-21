import math
import time

import numpy as np
from cma import fmin

EPS = 10 ** -5
PI = np.pi


class Reflector:
    def __init__(self, length, coordinates: np.ndarray):
        self.L = length
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.angle = np.mod(coordinates[2], PI)
        self.R = None
        self.dtheta_cm = None
        self.dtheta_r = None

    def set_angle(self, angle):
        self.angle = np.mod(angle, PI)
        return self

    def will_collide(self, particle_coordinates):

        x = particle_coordinates[:, 0]
        y = particle_coordinates[:, 1]
        theta_p = particle_coordinates[:, 2]

        dx = self.x - x
        dy = self.y - y
        self.R = np.sqrt(dx ** 2 + dy ** 2)
        self.dtheta_cm = np.arctan2(dy, dx) - theta_p

        self.dtheta_r = self.angle - theta_p

        positive_x_direction_after_rotation = np.cos(self.dtheta_cm) > 0

        y_height_after_rotation = np.abs(self.L * np.sin(self.dtheta_r))
        # self.R * np.sin(self.dtheta_cm) + y_height_after_rotation > 0
        # self.R * np.sin(self.dtheta_cm) - y_height_after_rotation < 0

        will_intercept = np.abs(self.R * np.sin(self.dtheta_cm)) < np.abs(
            self.L * np.sin(self.dtheta_r)
        )
        particle_collisions = np.logical_and(
            positive_x_direction_after_rotation, will_intercept
        ).astype(int)
        return particle_collisions

    def get_new_coordinates(self, particle_coordinates, particle_collisions):

        # get only the indicies and angles for the particles which collided with this reflector
        collided_particles_idx = np.nonzero(particle_collisions)[0]
        theta_p = particle_coordinates[collided_particles_idx, 2]

        # calc factor from center, where the particle will hit
        k = (
            self.R[collided_particles_idx]
            * np.sin(self.dtheta_cm[collided_particles_idx])
        ) / (self.L * np.sin(self.dtheta_r[collided_particles_idx]))

        # calc the new coordinates for x and y
        x_new = self.x + k * self.L * np.cos(self.angle) / 2
        y_new = self.y + k * self.L * np.sin(self.angle) / 2
        # theta_p_new = -(2 * self.angle + theta_p)
        theta_p_new = 0
        theta_p_new = np.mod(theta_p_new, PI)

        # # add small pertubations of the coordinates so that the particle doesn't hit the reflector again
        x_new = x_new + EPS * np.cos(theta_p_new)
        y_new = y_new + EPS * np.sin(theta_p_new)

        # set new particle coordinates
        particle_coordinates[collided_particles_idx, 0] = x_new
        particle_coordinates[collided_particles_idx, 1] = y_new
        particle_coordinates[collided_particles_idx, 2] = theta_p_new

        return particle_coordinates


def get_final_coordinates(H, W, particle_coordinates, all_colisions):

    # get indices of active particles
    # not_collided = all_colisions == 0
    noncollided_particles_idx = np.nonzero(all_colisions == 0)[0]
    # x_boundry_not_reached = np.abs(particle_coordinates[noncollided_particles_idx, 0]) != W / 2
    # y_boundry_not_reached = np.abs(particle_coordinates[noncollided_particles_idx, 1]) != H / 2
    # boundries_not_reached = np.logical_and(x_boundry_not_reached, y_boundry_not_reached)
    # active_non_collided_particles = np.logical_and(boundries_not_reached, noncollided_particles_idx)
    # active_non_collided_particle_idx = np.nonzero(active_non_collided_particles)[0]

    theta_p = particle_coordinates[noncollided_particles_idx, 2]
    R = 2 * np.sqrt(H ** 2 + W ** 2)

    # x boundry reached
    x1 = np.clip(R * np.cos(theta_p), -W / 2, W / 2)
    y1 = x1 * np.tan(theta_p)
    x_col_coordinates = np.vstack((x1, y1)).T
    R1_square = x1 ** 2 + y1 ** 2

    # y boundry reached
    y2 = np.clip(R * np.sin(theta_p), -H / 2, H / 2)
    x2 = y2 / (np.tan(theta_p) + 10 ** -5)
    y_col_coordinates = np.vstack((x2, y2)).T
    R2_square = x2 ** 2 + y2 ** 2

    # check which collision occured
    x_collision = np.expand_dims((R1_square < R2_square).astype(int), 1)
    y_collision = np.expand_dims((R1_square > R2_square).astype(int), 1)

    particle_coordinates[noncollided_particles_idx, :2] = (
        x_col_coordinates * x_collision + y_col_coordinates * y_collision
    )

    return particle_coordinates


def get_active_particles_idx(particle_coordinates, H, W):
    x_boundry_not_reached = np.abs(particle_coordinates[:, 0]) != W / 2
    y_boundry_not_reached = np.abs(particle_coordinates[:, 1]) != H / 2
    active_particles = np.logical_and(x_boundry_not_reached, y_boundry_not_reached)
    active_particles_idx = np.nonzero(active_particles)[0]
    return active_particles_idx


def run_one_collision(active_particle_coordinates, reflectors):

    num_active_particles = active_particle_coordinates.shape[0]
    all_colisions = np.zeros(num_active_particles)

    # run through all reflectors
    for reflector in reflectors:

        # check which particles collided with the current reflector
        particle_collisions = reflector.will_collide(active_particle_coordinates)

        # get new coordinates for the particles that collided with the reflector
        active_particle_coordinates = reflector.get_new_coordinates(
            active_particle_coordinates, particle_collisions
        )

        # count all reflections that occured
        all_colisions += particle_collisions

    # get final coordinates of particles that passed through all the reflectors
    active_particle_coordinates = get_final_coordinates(
        H, W, active_particle_coordinates, all_colisions
    )

    return active_particle_coordinates


def run_particles_through_reflectors(H, W, reflectors, particle_coordinates):

    num_particles = particle_coordinates.shape[0]
    trajectories = np.zeros((num_allowed_collisions, num_particles, 2))

    for colission in range(num_allowed_collisions):

        # get the indices of the active particles
        active_particles_idx = get_active_particles_idx(particle_coordinates, H, W)
        num_active_particles = len(active_particles_idx)
        if num_active_particles == 0:
            return trajectories[:colission, :, :]

        # run active particles through system for one collision
        active_particle_coordinates = particle_coordinates[active_particles_idx, :]
        active_particle_coordinates = run_one_collision(
            active_particle_coordinates, reflectors
        )

        # save trajectories
        particle_coordinates[active_particles_idx, :] = active_particle_coordinates
        trajectories[colission, :, :] = particle_coordinates[:, :2]

    return trajectories


def run_simulation(num_particles, H, W, reflectors, verbose=False):

    # np.random.seed(0)
    list_of_trajectories = []

    # for particle in range(num_particles):

    # initialize particle
    particle_coordinates = np.zeros((num_particles, 3))
    particle_coordinates[:, 0] = -W / 2 + EPS
    particle_coordinates[:, 1] = H * (np.random.rand(num_particles) - 0.5)
    particle_coordinates[:, 2] = (np.pi / 1000) * (np.random.rand(num_particles) - 0.5)

    trajectories = run_particles_through_reflectors(
        H, W, reflectors, particle_coordinates
    )
    # trajectory = run_single_particle(H, W, reflectors, particle_coordinates)
    if verbose:
        print(trajectories)
    # list_of_trajectories.append(trajectory)

    return trajectories


def get_reflectors(angles, position_std):
    # np.random.seed(0)
    reflectors = []
    for i in range(num_reflectors):
        reflector_coordinates = position_std * np.random.randn(3)
        reflector_coordinates[0] = np.clip(
            reflector_coordinates[0], -reflectors_W / 2, reflectors_W / 2
        )
        reflector_coordinates[1] = np.clip(
            reflector_coordinates[1], -reflectors_H / 2, reflectors_H / 2
        )
        reflector_coordinates[2] = angles[i]
        reflectors.append(Reflector(length_of_reflector, reflector_coordinates))
    return reflectors


def calculate_loss(trajectories, H, W):
    final_coordinates = trajectories[-1, :, :]
    square_norms = np.linalg.norm(final_coordinates, axis=1) ** 2
    return np.mean(square_norms)


def fun(angles):

    global reflectors

    reflectors = [
        reflector.set_angle(angle) for reflector, angle in zip(reflectors, angles)
    ]
    list_of_trajectories = run_simulation(
        num_particles, H, W, reflectors, verbose=False
    )
    loss = calculate_loss(list_of_trajectories, H, W)

    # np.minimum(np.maximum(angles, -np.pi/2, np.pi/2)

    penalty = np.sum(
        np.maximum(angles - np.pi / 2, 0) ** 2 + np.minimum(angles + np.pi / 2, 0) ** 2
    )

    return loss + penalty


if __name__ == "__main__":

    np.random.seed(0)
    W = 6
    H = 6
    reflectors_W = 4
    reflectors_H = 4

    num_allowed_collisions = 10
    num_reflectors = 2
    num_particles = 1000
    length_of_reflector = 1.0
    position_std = 2.0

    np.random.seed(0)
    angles = np.pi * np.random.randn(num_reflectors)
    reflectors = get_reflectors(angles, position_std)

    fmin(fun, angles, sigma0=1.0, options={"maxfevals": 5000})
