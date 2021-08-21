import os
import random
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append("\\".join(os.path.dirname(__file__).split("\\")[:-2]))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


from src.distributed_reflectors.boundries import BoxBoundry

SEED = 0


def create_system(
    initial_angle,
    box_width,
    box_height,
):
    random.seed(SEED)
    np.random.seed(SEED)

    num_particles = 10
    particle_width = 3
    particle_height = 3
    angle_amplitude = 0.1

    # generate box
    box_boundry = BoxBoundry(box_width, box_height)

    # generate particles
    particle_coordinates = np.zeros((num_particles, 3))
    particle_coordinates[:, 0] = -particle_width / 2
    particle_coordinates[:, 1] = particle_height * (np.random.rand(num_particles) - 0.5)
    particle_coordinates[:, 2] = initial_angle + angle_amplitude * (
        np.random.rand(num_particles) - 0.5
    )

    # simulate collisions with reflectors
    non_collided_particle_indices = random.sample(
        range(num_particles), int(num_particles / 2)
    )
    non_collided_particle_indices = np.array(non_collided_particle_indices)
    reflector_collided_particles_indices = [
        i for i in range(num_particles) if i not in non_collided_particle_indices
    ]

    return (
        box_boundry,
        particle_coordinates,
        non_collided_particle_indices,
        reflector_collided_particles_indices,
    )


def test_instantiation():
    box_boundry = BoxBoundry(3.0, 3.0)
    assert isinstance(box_boundry, BoxBoundry)


def test_x_boundry_collisions_are_valid():

    initial_angle = 0.0
    box_width = 4.0
    box_height = 4.0

    (
        box_boundry,
        particle_coordinates,
        non_collided_particle_indices,
        reflector_collided_particles_indices,
    ) = create_system(initial_angle, box_width, box_height)

    # find collisions with boundry
    (
        new_particle_coordinates,
        valid_x_boundry_collision,
        valid_y_boundry_collision,
    ) = box_boundry.get_final_coordinates(
        particle_coordinates, non_collided_particle_indices
    )

    assert np.all(
        np.logical_or(valid_x_boundry_collision, valid_y_boundry_collision)
    )  # either x or y collision occured for all relevant indices
    assert np.all(
        valid_x_boundry_collision != valid_y_boundry_collision
    )  # no x AND y collisions occured
    assert np.all(valid_x_boundry_collision)  # assert only x collisions occured


def test_x_boundry_collisions_have_correct_coordinates():

    initial_angle = 0.0
    box_width = 4.0
    box_height = 4.0

    (
        box_boundry,
        particle_coordinates,
        non_collided_particle_indices,
        reflector_collided_particles_indices,
    ) = create_system(initial_angle, box_width, box_height)

    # find collisions with boundry
    (new_particle_coordinates, _, _,) = box_boundry.get_final_coordinates(
        particle_coordinates, non_collided_particle_indices
    )

    # check coordinates of boundry collision particles
    x_coordinates = new_particle_coordinates[non_collided_particle_indices, 0]
    y_coordinates = new_particle_coordinates[non_collided_particle_indices, 1]
    assert np.all(np.abs(x_coordinates) == (box_width / 2))
    assert np.all(np.abs(y_coordinates) <= (box_height / 2))

    # check coordinates of reflector collided particles are left unchanged
    assert np.all(
        new_particle_coordinates[reflector_collided_particles_indices, :]
        == particle_coordinates[reflector_collided_particles_indices, :]
    )


def test_y_boundry_collisions_are_valid():

    initial_angle = np.pi / 2
    box_width = 4.0
    box_height = 4.0

    (
        box_boundry,
        particle_coordinates,
        non_collided_particle_indices,
        reflector_collided_particles_indices,
    ) = create_system(initial_angle, box_width, box_height)

    # find collisions with boundry
    (
        _,
        valid_x_boundry_collision,
        valid_y_boundry_collision,
    ) = box_boundry.get_final_coordinates(
        particle_coordinates, non_collided_particle_indices
    )

    assert np.all(
        np.logical_or(valid_x_boundry_collision, valid_y_boundry_collision)
    )  # either x or y collision occured for all relevant indices
    assert np.all(
        valid_x_boundry_collision != valid_y_boundry_collision
    )  # no x AND y collisions occured
    assert np.all(valid_y_boundry_collision)  # assert only y collisions occured


def test_y_boundry_collisions_have_correct_coordinates():

    initial_angle = np.pi / 2
    box_width = 4.0
    box_height = 4.0

    (
        box_boundry,
        particle_coordinates,
        non_collided_particle_indices,
        reflector_collided_particles_indices,
    ) = create_system(initial_angle, box_width, box_height)

    # find collisions with boundry
    (new_particle_coordinates, _, _,) = box_boundry.get_final_coordinates(
        particle_coordinates, non_collided_particle_indices
    )

    x_coordinates = new_particle_coordinates[non_collided_particle_indices, 0]
    y_coordinates = new_particle_coordinates[non_collided_particle_indices, 1]
    assert np.all(np.abs(x_coordinates) <= (box_width / 2))
    assert np.all(np.abs(y_coordinates) == (box_height / 2))

    # check coordinates of reflector collided particles are left unchanged
    assert np.all(
        new_particle_coordinates[reflector_collided_particles_indices, :]
        == particle_coordinates[reflector_collided_particles_indices, :]
    )


if __name__ == "__main__":
    test_instantiation()
    test_x_boundry_collisions_are_valid()
    test_x_boundry_collisions_have_correct_coordinates()
    test_y_boundry_collisions_are_valid()
    test_x_boundry_collisions_have_correct_coordinates()
