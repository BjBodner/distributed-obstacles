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

        will_intercept = (
            np.abs(
                self.R * np.sin(self.dtheta_cm),
            )
            < np.abs(self.L * np.sin(self.dtheta_r))
        )
        particle_collisions = np.logical_and(
            positive_x_direction_after_rotation,
            will_intercept,
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
