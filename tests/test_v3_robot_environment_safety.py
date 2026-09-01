import numpy as np

from v3_chan.robot_environment_safety import (
    AxisAlignedBounds,
    ColliderPairSafetyResult,
    _with_contact_result,
    links_are_excluded_self_pair,
    minimum_pair_gap,
    signed_aabb_surface_gap,
)


def _bounds(minimum, maximum):
    return AxisAlignedBounds(np.asarray(minimum), np.asarray(maximum))


def test_signed_aabb_gap_is_positive_when_separated():
    first = _bounds([0, 0, 0], [1, 1, 1])
    second = _bounds([2, 1, 0], [3, 2, 1])
    assert signed_aabb_surface_gap(first, second) == 1.0


def test_signed_aabb_gap_is_nonpositive_on_contact_or_overlap():
    first = _bounds([0, 0, 0], [1, 1, 1])
    touching = _bounds([1, 0, 0], [2, 1, 1])
    overlapping = _bounds([0.75, 0, 0], [2, 1, 1])
    assert signed_aabb_surface_gap(first, touching) == 0.0
    assert signed_aabb_surface_gap(first, overlapping) == -0.25


def test_minimum_pair_gap_preserves_collider_provenance():
    first = _bounds([0, 0, 0], [1, 1, 1])
    close = _bounds([1.1, 0, 0], [2, 1, 1])
    far = _bounds([3, 0, 0], [4, 1, 1])
    result = minimum_pair_gap(
        (("robot", first, "far", far), ("robot", first, "close", close))
    )
    assert result.geometry_valid
    assert np.isclose(result.surface_gap_m, 0.1)
    assert not result.collision
    assert not result.collision_valid
    assert result.first_path == "robot"
    assert result.second_path == "close"


def test_hard_collision_comes_from_contact_not_aabb_overlap():
    aabb_result = ColliderPairSafetyResult(
        geometry_valid=True,
        surface_gap_m=-0.02,
        first_path="aabb_robot",
        second_path="aabb_other",
    )
    no_contact = _with_contact_result(
        aabb_result,
        contact_valid=True,
        contact_pair=None,
    )
    contact = _with_contact_result(
        aabb_result,
        contact_valid=True,
        contact_pair=("contact_robot", "contact_other"),
    )

    assert not no_contact.collision
    assert no_contact.collision_valid
    assert contact.collision
    assert contact.collision_valid
    assert contact.first_path == "contact_robot"


def test_self_pair_exclusions_cover_adjacent_and_gripper_internal_links():
    assert links_are_excluded_self_pair("panda_link6", "panda_link7")
    assert links_are_excluded_self_pair("panda_link5", "panda_link7")
    assert links_are_excluded_self_pair("panda_link6", "panda_hand")
    assert links_are_excluded_self_pair("panda_hand", "panda_leftfinger")
    assert not links_are_excluded_self_pair("panda_link6", "panda_link0")
