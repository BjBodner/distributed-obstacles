import numpy as np


class BoxBoundry:
    """a box centered at zero, with a fixed height and width.
    ranges of allowed values within box are between [-width / 2, width / 2] in the x direction
    and [-height / 2, height / 2] in the y direction
    """

    def __init__(self, height: float, width: float) -> None:
        self.height = height
        self.width = width

    def _get_non_collided_particle_indices(all_colisions: np.ndarray) -> np.ndarray:
        non_collided_particle_indices = np.nonzero(all_colisions == 0)[0]
        return non_collided_particle_indices

    def _calc_distance_to_x_boundry(self, theta_p: np.ndarray) -> np.ndarray:
        pass


def get_final_coordinates(H, W, particle_coordinates, all_colisions):

    # get indices of active particles
    noncollided_particles_idx = np.nonzero(all_colisions == 0)[0]

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
