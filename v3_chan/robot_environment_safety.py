"""Conservative static/self safety queries over composed Panda colliders.

The hand-to-robot metric uses exact PhysX sphere overlap in
``end_effector_safety_runtime``.  Isaac Sim 4.5 does not expose the same signed
distance query for an arbitrary collider pair through the API used here, so
static and self distance shaping use world-aligned bounds computed from the
actual composed CollisionAPI prims.  The result is conservative and its source
is kept explicit in every checkpoint.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

try:
    from v3_chan.end_effector_safety_geometry import (
        DISTAL_LINK_NAMES,
        MISSING_SURFACE_GAP_M,
    )
except ImportError:
    from end_effector_safety_geometry import (
        DISTAL_LINK_NAMES,
        MISSING_SURFACE_GAP_M,
    )


PANDA_LINK_NAMES = tuple(f"panda_link{index}" for index in range(9)) + (
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)


@dataclass(frozen=True)
class AxisAlignedBounds:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64).reshape(-1)
        maximum = np.asarray(self.maximum, dtype=np.float64).reshape(-1)
        if (
            minimum.size != 3
            or maximum.size != 3
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or np.any(maximum < minimum)
        ):
            raise ValueError("Axis-aligned bounds must be finite ordered 3-D vectors")
        object.__setattr__(self, "minimum", minimum.copy())
        object.__setattr__(self, "maximum", maximum.copy())


@dataclass(frozen=True)
class ColliderPairSafetyResult:
    geometry_valid: bool
    surface_gap_m: float = MISSING_SURFACE_GAP_M
    collision: bool = False
    collision_valid: bool = False
    first_path: str = ""
    second_path: str = ""


@dataclass(frozen=True)
class RobotEnvironmentSafetyResult:
    static: ColliderPairSafetyResult
    self_collision: ColliderPairSafetyResult
    query_time_ms: float = 0.0

    @property
    def geometry_valid(self) -> bool:
        return self.static.geometry_valid and self.self_collision.geometry_valid

    @property
    def collision(self) -> bool:
        return self.static.collision or self.self_collision.collision


def signed_aabb_surface_gap(
    first: AxisAlignedBounds,
    second: AxisAlignedBounds,
) -> float:
    """Return positive separation or conservative negative overlap depth."""

    separation = np.maximum(
        np.maximum(first.minimum - second.maximum, second.minimum - first.maximum),
        0.0,
    )
    if np.any(separation > 0.0):
        return float(np.linalg.norm(separation))
    overlap = np.minimum(first.maximum, second.maximum) - np.maximum(
        first.minimum, second.minimum
    )
    return -float(np.min(np.maximum(overlap, 0.0)))


def minimum_pair_gap(
    pairs: Iterable[tuple[str, AxisAlignedBounds, str, AxisAlignedBounds]],
) -> ColliderPairSafetyResult:
    best: tuple[float, str, str] | None = None
    for first_path, first_bounds, second_path, second_bounds in pairs:
        gap = signed_aabb_surface_gap(first_bounds, second_bounds)
        if not math.isfinite(gap):
            continue
        candidate = (float(gap), str(first_path), str(second_path))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return ColliderPairSafetyResult(geometry_valid=False)
    return ColliderPairSafetyResult(
        geometry_valid=True,
        surface_gap_m=best[0],
        # Aligned bounds may overlap even when the composed colliders do not.
        # PhysX contact reports provide the hard collision label separately.
        collision=False,
        collision_valid=False,
        first_path=best[1],
        second_path=best[2],
    )


def links_are_excluded_self_pair(first: str, second: str) -> bool:
    """Exclude the normal two-hop kinematic neighborhood and gripper pairs."""

    if not first or not second or first == second:
        return True
    chain_index = {
        **{f"panda_link{index}": index for index in range(8)},
        "panda_link8": 8,
        "panda_hand": 8,
        "panda_leftfinger": 9,
        "panda_rightfinger": 9,
    }
    if first in chain_index and second in chain_index:
        if abs(chain_index[first] - chain_index[second]) <= 2:
            return True
    pair = frozenset((first, second))
    exclusions = {
        frozenset(("panda_leftfinger", "panda_rightfinger")),
        frozenset(("panda_link8", "panda_hand")),
    }
    return pair in exclusions


class PandaEnvironmentSafetyRuntime:
    """Discover composed collision prims and evaluate conservative pair gaps."""

    GEOMETRY_SOURCE = "isaac_stage_collisionapi_conservative_world_aabb"

    def __init__(self, *, robot_prim_path: str = "/World/Franka") -> None:
        import omni.usd
        from pxr import Usd, UsdGeom

        self.robot_prim_path = str(robot_prim_path).rstrip("/")
        self._stage = omni.usd.get_context().get_stage()
        self._usd = Usd
        self._bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        self._robot_colliders: dict[str, str] = {}
        self._static_colliders: tuple[str, ...] = ()
        self._contact_reporting_paths: tuple[str, ...] = ()
        self._physx_simulation = None
        self._physics_schema_tools = None
        self._contact_lost_type = 1
        self.refresh()

    @property
    def geometry_valid(self) -> bool:
        protected = any(
            link in DISTAL_LINK_NAMES for link in self._robot_colliders.values()
        )
        self_pair = any(
            not links_are_excluded_self_pair(first, second)
            for first in self._robot_colliders.values()
            for second in self._robot_colliders.values()
        )
        return bool(protected and self._static_colliders and self_pair)

    def refresh(self) -> None:
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics

        robot_root = self._stage.GetPrimAtPath(self.robot_prim_path)
        if not robot_root.IsValid():
            raise RuntimeError(f"Panda prim not found: {self.robot_prim_path}")
        robot: dict[str, str] = {}
        for prim in Usd.PrimRange(robot_root):
            if not _collision_enabled(prim, UsdPhysics):
                continue
            link = _owning_panda_link(str(prim.GetPath()))
            if link:
                robot[str(prim.GetPath())] = link

        static: list[str] = []
        world_root = self._stage.GetPrimAtPath("/World")
        for prim in Usd.PrimRange(world_root):
            path = str(prim.GetPath())
            if path.startswith(self.robot_prim_path) or not _is_static_path(path):
                continue
            if _collision_enabled(prim, UsdPhysics):
                static.append(path)

        self._robot_colliders = robot
        self._static_colliders = tuple(sorted(set(static)))
        if not self.geometry_valid:
            raise RuntimeError(
                "Static/self safety geometry is incomplete: "
                f"robot_colliders={len(robot)} static_colliders={len(static)}"
            )

        reporting_paths = {self.robot_prim_path}
        for collider_path in robot:
            body_path = _rigid_body_ancestor_path(
                self._stage,
                collider_path,
                UsdPhysics,
            )
            if body_path:
                reporting_paths.add(body_path)
        for path in sorted(reporting_paths):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)

        self._contact_reporting_paths = tuple(sorted(reporting_paths))
        self._physx_simulation = get_physx_simulation_interface()
        self._physx_simulation.flush_changes()
        self._physics_schema_tools = PhysicsSchemaTools
        self._contact_lost_type = int(ContactEventType.CONTACT_LOST)

    def evaluate(self) -> RobotEnvironmentSafetyResult:
        started = time.perf_counter()
        self._bbox_cache.Clear()
        bounds = {
            path: self._world_bounds(path)
            for path in (*self._robot_colliders, *self._static_colliders)
        }
        protected = tuple(
            path
            for path, link in self._robot_colliders.items()
            if link in DISTAL_LINK_NAMES and bounds.get(path) is not None
        )
        static_pairs = (
            (robot_path, bounds[robot_path], static_path, bounds[static_path])
            for robot_path in protected
            for static_path in self._static_colliders
            if bounds.get(static_path) is not None
        )
        seen: set[tuple[str, str]] = set()
        self_pairs: list[tuple[str, AxisAlignedBounds, str, AxisAlignedBounds]] = []
        for first_path in protected:
            first_link = self._robot_colliders[first_path]
            for second_path, second_link in self._robot_colliders.items():
                if bounds.get(second_path) is None:
                    continue
                key = tuple(sorted((first_path, second_path)))
                if key in seen or first_path == second_path:
                    continue
                seen.add(key)
                if links_are_excluded_self_pair(first_link, second_link):
                    continue
                self_pairs.append(
                    (first_path, bounds[first_path], second_path, bounds[second_path])
                )
        static_result = minimum_pair_gap(static_pairs)
        self_result = minimum_pair_gap(self_pairs)
        contact_valid, static_contact, self_contact = self._current_contacts()
        static_result = _with_contact_result(
            static_result,
            contact_valid=contact_valid,
            contact_pair=static_contact,
        )
        self_result = _with_contact_result(
            self_result,
            contact_valid=contact_valid,
            contact_pair=self_contact,
        )
        return RobotEnvironmentSafetyResult(
            static=static_result,
            self_collision=self_result,
            query_time_ms=(time.perf_counter() - started) * 1000.0,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "environment_safety_geometry_source": self.GEOMETRY_SOURCE,
            "environment_safety_distance_is_conservative": True,
            "environment_safety_collision_semantics": "physx_contact_report",
            "environment_safety_contact_reporting_paths": self._contact_reporting_paths,
            "environment_safety_robot_collider_count": len(self._robot_colliders),
            "environment_safety_static_collider_paths": self._static_colliders,
            "environment_safety_protected_links": DISTAL_LINK_NAMES,
            "environment_safety_task_objects_excluded": True,
        }

    def _world_bounds(self, path: str) -> AxisAlignedBounds | None:
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        try:
            world_range = self._bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = np.asarray(world_range.GetMin(), dtype=np.float64)
            maximum = np.asarray(world_range.GetMax(), dtype=np.float64)
            return AxisAlignedBounds(minimum, maximum)
        except (RuntimeError, TypeError, ValueError):
            return None

    def _current_contacts(
        self,
    ) -> tuple[bool, tuple[str, str] | None, tuple[str, str] | None]:
        if self._physx_simulation is None or self._physics_schema_tools is None:
            return False, None, None
        try:
            report = self._physx_simulation.get_contact_report()
            headers = report[0]
        except (AttributeError, RuntimeError, TypeError):
            return False, None, None

        static_contact: tuple[str, str] | None = None
        self_contact: tuple[str, str] | None = None
        for header in headers:
            if int(header.type) == self._contact_lost_type:
                continue
            first_path = str(
                self._physics_schema_tools.intToSdfPath(header.collider0)
            )
            second_path = str(
                self._physics_schema_tools.intToSdfPath(header.collider1)
            )
            first_link = _owning_panda_link(first_path)
            second_link = _owning_panda_link(second_path)
            if first_link and second_link:
                if (
                    (first_link in DISTAL_LINK_NAMES or second_link in DISTAL_LINK_NAMES)
                    and not links_are_excluded_self_pair(first_link, second_link)
                ):
                    self_contact = (first_path, second_path)
                continue
            if first_link in DISTAL_LINK_NAMES and _matches_any_path(
                second_path,
                self._static_colliders,
            ):
                static_contact = (first_path, second_path)
            elif second_link in DISTAL_LINK_NAMES and _matches_any_path(
                first_path,
                self._static_colliders,
            ):
                static_contact = (second_path, first_path)
        return True, static_contact, self_contact


def _collision_enabled(prim, usd_physics) -> bool:
    api = usd_physics.CollisionAPI(prim)
    if not api:
        return False
    return api.GetCollisionEnabledAttr().Get() is not False


def _owning_panda_link(path: str) -> str:
    parts = set(str(path).split("/"))
    return next((name for name in PANDA_LINK_NAMES if name in parts), "")


def _is_static_path(path: str) -> bool:
    prefixes = (
        "/World/table",
        "/World/defaultGroundPlane",
        "/World/groundPlane",
    )
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _rigid_body_ancestor_path(stage, collider_path: str, usd_physics) -> str:
    prim = stage.GetPrimAtPath(collider_path)
    while prim.IsValid():
        if prim.HasAPI(usd_physics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return ""


def _matches_any_path(path: str, candidates: Iterable[str]) -> bool:
    path = str(path).rstrip("/")
    for candidate in candidates:
        candidate = str(candidate).rstrip("/")
        if (
            path == candidate
            or path.startswith(candidate + "/")
            or candidate.startswith(path + "/")
        ):
            return True
    return False


def _with_contact_result(
    result: ColliderPairSafetyResult,
    *,
    contact_valid: bool,
    contact_pair: tuple[str, str] | None,
) -> ColliderPairSafetyResult:
    if not contact_valid:
        return replace(result, collision=False, collision_valid=False)
    if contact_pair is None:
        return replace(result, collision=False, collision_valid=True)
    return replace(
        result,
        surface_gap_m=min(float(result.surface_gap_m), 0.0),
        collision=True,
        collision_valid=True,
        first_path=str(contact_pair[0]),
        second_path=str(contact_pair[1]),
    )
