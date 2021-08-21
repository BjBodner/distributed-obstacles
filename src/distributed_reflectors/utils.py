import numpy as np


def parse_coordinates(particle_coordinates: np.ndarray) -> np.ndarray:
    particles_x = particle_coordinates[:, 0]
    particles_y = particle_coordinates[:, 1]
    particles_angles = particle_coordinates[:, 2]

    return particles_x, particles_y, particles_angles
