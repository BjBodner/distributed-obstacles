# import math
# import time

import numpy as np

# from cma import fmin
from .utils import parse_coordinates

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

        particles_x, particles_y, particles_angles = parse_coordinates(
            particle_coordinates
        )

        # calculate displacement distance and angles
        x_displacement_from_particles = self.x - particles_x
        y_displacement_from_particles = self.y - particles_y
        self.displacement_distance_from_particles = np.sqrt(
            x_displacement_from_particles ** 2 + y_displacement_from_particles ** 2
        )
        self.displacement_angle_from_particles = np.arctan2(
            y_displacement_from_particles, x_displacement_from_particles
        )

        # rotate system with resepct to each particle angle
        self.relative_displacement_angle_wrt_particles = (
            self.displacement_angle_from_particles - particles_angles
        )
        self.relative_reflector_angle_wrt_particles = self.angle - particles_angles

        # calculate the y coordinates of the upper and lower edges of the reflectors - after rotation
        reflector_height_after_rotation = self.L * np.sin(
            self.relative_reflector_angle_wrt_particles
        )
        displacement_height_after_rotation = (
            self.displacement_distance_from_particles
            * np.sin(self.relative_displacement_angle_wrt_particles)
        )
        y_of_upper_reflector_edge = (
            displacement_height_after_rotation + reflector_height_after_rotation
        )
        y_of_lower_reflector_edge = (
            displacement_height_after_rotation - reflector_height_after_rotation
        )

        # check if the reflector intercepts the positive x half-plane
        # since we rotated the system such that each particle is moving in the +x direction AND centered it in (0,0)
        # if the reflector intercepts the positive x half-plane, this means the particle will intercept it
        reflector_intercepts_positive_x_plane = np.logical_and(
            (y_of_upper_reflector_edge > 0), (y_of_lower_reflector_edge < 0)
        )
        particle_collisions = reflector_intercepts_positive_x_plane.astype(int)

        # reflector_intercepts_positivle_x_plane = np.sign(y_of_upper_reflector_edge) != np.sign(y_of_lower_reflector_edge)

        # furthest_y_of_reflector_wrt_particles = self.R * np.sin(self.dtheta_cm) + self.L * np.sin(np.sin(self.dtheta_r)
        # nearest_y_of_reflector_wrt_particles = self.R * np.sin(self.dtheta_cm) + self.L * np.sin(self.dtheta_r)

        # # self.R * np.sin(self.dtheta_cm) + y_height_after_rotation > 0
        # # self.R * np.sin(self.dtheta_cm) - y_height_after_rotation < 0

        # will_intercept = (
        #     np.abs(
        #         self.R * np.sin(self.dtheta_cm),
        #     )
        #     < np.abs(self.L * np.sin(self.dtheta_r))
        # )
        # particle_collisions = np.logical_and(
        #     positive_x_direction_after_rotation,
        #     will_intercept,
        # ).astype(int)
        return particle_collisions

    def get_new_coordinates(self, particle_coordinates, particle_collisions):

        # get only the indicies and angles for the particles which collided with this reflector
        collided_particles_idx = np.nonzero(particle_collisions)[0]
        new_particle_coordinates = np.copy(particle_coordinates)
        if len(collided_particles_idx) > 0:
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
            theta_p_new = np.mod(theta_p_new, 2 * PI)

            # # add small pertubations of the coordinates so that the particle doesn't hit the reflector again
            x_new = x_new + EPS * np.cos(theta_p_new)
            y_new = y_new + EPS * np.sin(theta_p_new)

            # set new particle coordinates
            new_particle_coordinates[collided_particles_idx, 0] = x_new
            new_particle_coordinates[collided_particles_idx, 1] = y_new
            new_particle_coordinates[collided_particles_idx, 2] = theta_p_new

        return new_particle_coordinates
