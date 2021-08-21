import numpy as np

from .utils import parse_coordinates

# TODO move this to utils


class BoxBoundry:
    """a box centered at zero, with a fixed height and width.
    ranges of allowed values within box are between [-width / 2, width / 2] in the x direction
    and [-height / 2, height / 2] in the y direction
    """

    def __init__(self, height: float, width: float) -> None:

        self._check_inputs(height, width)
        self.height = height
        self.width = width
        self.max_radius = np.sqrt(height ** 2 + width ** 2)

    def _check_inputs(self, height: float, width: float) -> None:
        # check inputs
        if height < 0:
            raise ValueError(
                f"height argument must be positive float, but got value height = {height}"
            )
        if width < 0:
            raise ValueError(
                f"width argument must be positive float, but got value width = {width}"
            )

    def _get_non_collided_particle_indices(
        self, all_colisions: np.ndarray
    ) -> np.ndarray:
        non_collided_particle_indices = np.nonzero(all_colisions == 0)[0]
        return non_collided_particle_indices

    def _calc_x_collision_coordinates(
        self, non_collided_particles_coordinates: np.ndarray
    ) -> np.ndarray:

        particles_x, particles_y, particles_angles = parse_coordinates(
            non_collided_particles_coordinates
        )

        # calc coordinates in case of x collision
        max_x_displacement = self.max_radius * np.cos(particles_angles)
        x_final = particles_x + max_x_displacement
        allowed_x_final = np.clip(x_final, -self.width / 2, self.width / 2)
        allowed_y_displacement = allowed_x_final * np.tan(particles_angles)
        y_final = particles_y + allowed_y_displacement

        # check if a valid x boundry collision occured
        x_boundry_reached = np.abs(allowed_x_final) == (self.width / 2)
        y_value_within_allowed_range = np.abs(y_final) <= (self.height / 2)
        valid_x_boundry_collision = np.logical_and(
            x_boundry_reached, y_value_within_allowed_range
        )
        valid_x_boundry_collision = np.expand_dims(
            valid_x_boundry_collision.astype(int), 1
        )

        # for cases of a valid x collision, calculate the resulting coordinates
        # all other coordinates that did not have this collision will be set to zero
        x_collision_coordinates = np.vstack((allowed_x_final, y_final)).T
        x_collision_coordinates = (
            x_collision_coordinates * valid_x_boundry_collision
        )  # zero out all coordinates that did not have an x collision

        return x_collision_coordinates, valid_x_boundry_collision

    def _calc_y_collision_coordinates(
        self, non_collided_particles_coordinates: np.ndarray
    ) -> np.ndarray:

        particles_x, particles_y, particles_angles = parse_coordinates(
            non_collided_particles_coordinates
        )

        # calc coordinates in case of y collision
        max_y_displacement = self.max_radius * np.sin(particles_angles)
        y_final = particles_y + max_y_displacement
        allowed_y_final = np.clip(y_final, -self.height / 2, self.height / 2)
        allowed_x_displacement = allowed_y_final / np.tan(particles_angles)
        x_final = particles_x + allowed_x_displacement

        # check if a valid y boundry collision occured
        y_boundry_reached = np.abs(allowed_y_final) == (self.height / 2)
        x_value_within_allowed_range = np.abs(x_final) <= (self.width / 2)
        valid_y_boundry_collision = np.logical_and(
            y_boundry_reached, x_value_within_allowed_range
        ).astype(int)
        valid_y_boundry_collision = np.expand_dims(
            valid_y_boundry_collision.astype(int), 1
        )

        # for cases of a valid y collision, calculate the resulting coordinates
        # all other coordinates that did not have this collision will be set to zero
        y_collision_coordinates = np.vstack((x_final, allowed_y_final)).T
        y_collision_coordinates = y_collision_coordinates * valid_y_boundry_collision

        return y_collision_coordinates, valid_y_boundry_collision

    def get_final_coordinates(
        self,
        particles_coordinates: np.ndarray,
        non_collided_particle_indices: np.ndarray,
    ):

        new_particle_coordinates = np.copy(particles_coordinates)
        non_collided_particles_coordinates = particles_coordinates[
            non_collided_particle_indices, :
        ]

        # calculates coordinates in cases of x and y collisions
        # all particles that did not collide with reflectors are expected to reach the boundry
        (
            x_collision_coordinates,
            valid_x_boundry_collision,
        ) = self._calc_x_collision_coordinates(non_collided_particles_coordinates)
        (
            y_collision_coordinates,
            valid_y_boundry_collision,
        ) = self._calc_y_collision_coordinates(non_collided_particles_coordinates)

        # replace the x,y coordinates of the
        new_particle_coordinates[non_collided_particle_indices, :2] = (
            x_collision_coordinates + y_collision_coordinates
        )

        return (
            new_particle_coordinates,
            valid_x_boundry_collision,
            valid_y_boundry_collision,
        )
