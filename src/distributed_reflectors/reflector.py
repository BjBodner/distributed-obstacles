# import math
# import time

from typing import Tuple

import numpy as np

from .utils import parse_coordinates

EPS = 10 ** -5
PI = np.pi


def get_reflectors(
    x_range,
    y_range,
    reflector_position_amplitudes: list,
    reflector_length: float,
    num_reflectors: int,
    random_state: int,
) -> list:
    if random_state is not None:
        np.random.seed(random_state)

    # calculate maximal allowed values for the
    x_allowed_values = [x_range[0] + reflector_length, x_range[1] - reflector_length]
    y_allowed_values = [y_range[0] + reflector_length, y_range[1] - reflector_length]

    # y_boundry

    reflectors = []
    for i in range(num_reflectors):

        # define the coordinates for each selector
        reflector_coordinates = np.zeros(3)
        reflector_coordinates[0] = (
            reflector_position_amplitudes[0] * 2 * (np.random.rand(1) - 0.5)
        )
        reflector_coordinates[0] = np.clip(reflector_coordinates[0], *x_allowed_values)
        reflector_coordinates[1] = (
            reflector_position_amplitudes[1] * 2 * (np.random.rand(1) - 0.5)
        )
        reflector_coordinates[1] = np.clip(reflector_coordinates[1], *y_allowed_values)
        reflector_coordinates[2] = np.mod(
            reflector_position_amplitudes[2] * np.random.rand(1), PI
        )

        # instantiate reflector and add to reflector list
        reflectors.append(Reflector(reflector_length, reflector_coordinates))
    return reflectors


class Reflector:
    def __init__(self, length, coordinates: np.ndarray) -> None:
        self.L = length
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.angle = np.mod(coordinates[2], PI)
        self.R = None
        self.dtheta_cm = None
        self.dtheta_r = None

    def set_angle(self, angle) -> None:
        self.angle = np.mod(angle, PI)
        return self

    def will_collide(self, particle_coordinates: np.ndarray) -> np.ndarray:

        particles_x, particles_y, particles_angles = parse_coordinates(
            particle_coordinates
        )

        # calculate displacement distance and angles and rotate system
        self._calc_displacement_distance_and_angle(particles_x, particles_y)

        # rotate system with resepct to each particle angle
        self._calc_rotated_angles(particles_angles)

        # calculate the y coordinates of the upper and lower edges of the reflectors - after rotation
        self._calc_positions_of_rotated_reflector_edges()

        # check if the reflector intercepts the positive x half-plane
        # since we rotated the system such that each particle is moving in the +x direction AND centered it in (0,0)
        # if the reflector intercepts the positive x half-plane, this means the particle will intercept it
        reflector_intercepts_positive_x_plane = np.logical_and(
            (self.y_of_upper_reflector_edge > 0), (self.y_of_lower_reflector_edge < 0)
        )
        particle_collisions = reflector_intercepts_positive_x_plane.astype(int)

        return particle_collisions

    def get_new_coordinates(
        self, particle_coordinates: np.ndarray, particle_collisions: np.ndarray
    ) -> np.ndarray:

        # get only the indicies and angles for the particles which collided with this reflector
        collided_particles_idx = np.nonzero(particle_collisions)[0]
        new_particle_coordinates = np.copy(particle_coordinates)

        if len(collided_particles_idx) > 0:
            # get only the particle coordinates that collided
            particles_x, particles_y, particles_angles = parse_coordinates(
                particle_coordinates, collided_particles_idx
            )

            # calculate the new coordinates for the particles that collided
            x_new, y_new = self._calc_new_xy(
                particles_x, particles_y, particles_angles, collided_particles_idx
            )
            angle_new = self._calc_new_angle(particles_angles, collided_particles_idx)

            # set the new particle coordinates
            new_particle_coordinates[collided_particles_idx, 0] = x_new
            new_particle_coordinates[collided_particles_idx, 1] = y_new
            new_particle_coordinates[collided_particles_idx, 2] = angle_new

        return new_particle_coordinates

    def _calc_new_xy(
        self,
        particles_x: np.ndarray,
        particles_y: np.ndarray,
        particles_angles: np.ndarray,
        collided_particles_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # calculate the displaced and rotated new x and y (where the particles collide with the reflector)
        rotated_particle_x_dimensionless = (
            -self.y_of_lower_reflector_edge[collided_particles_idx]
        ) / (
            self.y_of_upper_reflector_edge[collided_particles_idx]
            - self.y_of_lower_reflector_edge[collided_particles_idx]
        )
        displaced_rotated_particle_x = (
            rotated_particle_x_dimensionless
            * (
                self.x_of_upper_reflector_edge[collided_particles_idx]
                - self.x_of_lower_reflector_edge[collided_particles_idx]
            )
            + self.x_of_lower_reflector_edge[collided_particles_idx]
        )
        displaced_rotated_particle_y = np.zeros_like(particles_angles)

        # rotate the coordinates back to their original orientation
        particle_displacement_relative_to_original_position = np.sqrt(
            displaced_rotated_particle_x ** 2 + displaced_rotated_particle_y ** 2
        )
        displaced_x_new = particle_displacement_relative_to_original_position * np.cos(
            particles_angles
        )
        displaced_y_new = particle_displacement_relative_to_original_position * np.sin(
            particles_angles
        )

        # add back the displacemet that was initially subtracted
        x_new = displaced_x_new + particles_x
        y_new = displaced_y_new + particles_y

        return x_new, y_new

    def _calc_new_angle(
        self, particles_angles: np.ndarray, collided_particles_idx: np.ndarray
    ) -> np.ndarray:
        rotated_particle_angle_after_reflection = (
            2 * self.relative_reflector_angle_wrt_particles[collided_particles_idx]
        )

        # rotate back the angle
        angle_new = rotated_particle_angle_after_reflection + particles_angles
        angle_new = np.mod(angle_new, 2 * PI)
        return angle_new

    def _calc_displacement_distance_and_angle(
        self, particles_x: np.ndarray, particles_y: np.ndarray
    ) -> None:
        x_displacement_from_particles = self.x - particles_x
        y_displacement_from_particles = self.y - particles_y
        self.displacement_distance_from_particles = np.sqrt(
            x_displacement_from_particles ** 2 + y_displacement_from_particles ** 2
        )
        self.displacement_angle_from_particles = np.arctan2(
            y_displacement_from_particles, x_displacement_from_particles
        )

        # return displacement_distance_from_particles, displacement_angle_from_particles

    def _calc_rotated_angles(self, particles_angles: np.ndarray) -> None:
        self.relative_displacement_angle_wrt_particles = (
            self.displacement_angle_from_particles - particles_angles
        )
        self.relative_reflector_angle_wrt_particles = self.angle - particles_angles

    def _calc_positions_of_rotated_reflector_edges(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # calculate the relative heights and widths after rotation
        self.reflector_height_after_rotation = self.L * np.sin(
            self.relative_reflector_angle_wrt_particles
        )
        self.displacement_height_after_rotation = (
            self.displacement_distance_from_particles
            * np.sin(self.relative_displacement_angle_wrt_particles)
        )
        self.reflector_width_after_rotation = self.L * np.cos(
            self.relative_reflector_angle_wrt_particles
        )
        self.displacement_width_after_rotation = (
            self.displacement_distance_from_particles
            * np.cos(self.relative_displacement_angle_wrt_particles)
        )

        # calculate the coordinates of the upper and lower edges of the reflector
        self.x_of_upper_reflector_edge = (
            self.displacement_width_after_rotation + self.reflector_width_after_rotation
        )
        self.x_of_lower_reflector_edge = (
            self.displacement_width_after_rotation - self.reflector_width_after_rotation
        )
        self.y_of_upper_reflector_edge = (
            self.displacement_height_after_rotation
            + self.reflector_height_after_rotation
        )
        self.y_of_lower_reflector_edge = (
            self.displacement_height_after_rotation
            - self.reflector_height_after_rotation
        )
