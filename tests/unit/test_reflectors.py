import os
import random
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append("\\".join(os.path.dirname(__file__).split("\\")[:-2]))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


from src.distributed_reflectors.reflector import Reflector

SEED = 0
PI = np.pi


straight_collision_and_clear_miss = (
    np.array([[-3, 0, 0], [-3, 0, PI / 2]]),
    1.0,
    np.array([0, 0, PI / 2]),
    np.array([1, 0]),
    np.array([[0, 0, PI], [-3, 0, PI / 2]]),
)

barely_misses = (
    np.array([[-1.0, 0, PI / 4], [-1.0, 0, -PI / 4]]),
    1.0,
    np.array([0, 0, PI / 2]),
    np.array([0, 0]),
    np.array([[-1.0, 0, PI / 4], [-1.0, 0, -PI / 4]]),
)

near_misses_but_hit = (
    np.array([[-0.99, 0, PI / 4], [-0.99, 0, -PI / 4]]),
    1.0,
    np.array([0, 0, PI / 2]),
    np.array([1, 1]),
    np.array([[0.0, 0.99, 3 * PI / 4], [0.0, -0.99, -5 * PI / 4]]),
)

hit = (
    np.array([[-1.0, 0, PI / 6], [-1.0, 0, -PI / 6]]),
    1.0,
    np.array([0, 0, PI / 2]),
    np.array([1, 1]),
    np.array([[0, 0.5, 5 * PI / 6], [0.0, -0.5, 7 * PI / 6]]),
)


def test_reflector_instantiation():
    length = 2.0
    coordinates = 5 * np.random.randn(3)
    reflector = Reflector(length, coordinates)
    assert isinstance(reflector, Reflector)
    assert 0 <= reflector.angle <= np.pi


@pytest.mark.parametrize(
    "particle_coordinates,length,reflector_coordinates,expected_collisions,expected_new_coordinates",
    [straight_collision_and_clear_miss, barely_misses, near_misses_but_hit, hit],
)
def test_correct_collision_detections(
    particle_coordinates,
    length,
    reflector_coordinates,
    expected_collisions,
    expected_new_coordinates,
):

    reflector = Reflector(length, reflector_coordinates)
    collisions = reflector.will_collide(particle_coordinates)

    np.testing.assert_equal(collisions, expected_collisions)


@pytest.mark.parametrize(
    "particle_coordinates,length,reflector_coordinates,expected_collisions,expected_new_coordinates",
    [straight_collision_and_clear_miss, barely_misses, near_misses_but_hit, hit],
)
def test_correct_collision_coordinates(
    particle_coordinates,
    length,
    reflector_coordinates,
    expected_collisions,
    expected_new_coordinates,
):

    reflector = Reflector(length, reflector_coordinates)
    particle_collisions = reflector.will_collide(particle_coordinates)
    new_coordinates = reflector.get_new_coordinates(
        particle_coordinates, particle_collisions
    )

    np.testing.assert_equal(new_coordinates, expected_new_coordinates)


if __name__ == "__main__":

    # for particle_coordinates, length, reflector_coordinates, expected_collisions in zip(
    #     particle_coordinates_options,
    #     length_options,
    #     reflector_coordinates_options,
    #     expected_collisions_options,
    # ):
    #     test_correct_collisions(particle_coordinates, length, reflector_coordinates, expected_collisions)
    #     a = 1

    test_correct_collision_coordinates(*straight_collision_and_clear_miss)
    test_reflector_instantiation()
