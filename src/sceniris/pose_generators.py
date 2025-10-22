# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
from shapely import MultiPolygon, Polygon
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from scene_synthesizer.utils import sample_volume_mesh, sample_polygon as _sample_polygon

from sceniris.utils import sample_random_z_rotations, check_within_polygon

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from scene_synthesizer.scene import SupportSurface

DEFAULT_REPLISH_SIZE = 2

"""
This file defines new pose generators that support batch generation and caching compared to scene_synthesizer.
"""


def sample_polygon(
    polygon: Polygon, 
    count: int, 
    factor: float = 1.5, 
    max_iter: int = 10, 
    seed: int | None = None
):
    """
    Override the sample_polygon function in scene_synthesizer.utils to support empty polygon.

    Args:
        polygon (Polygon): The polygon to sample from.
        count (int): The number of samples to generate.
        factor (float): The factor to scale the polygon by.
        max_iter (int): The maximum number of iterations to sample from the polygon.
        seed (int | None): The seed to use for the random number generator.
    """
    if polygon.is_empty:
        return np.full((count, 2), np.nan, dtype=np.float32)
    return _sample_polygon(polygon, count, factor, max_iter, seed)

class PoseGenerator:
    def __init__(
        self, 
        seed: int | None = None, 
        replenish_size: int = DEFAULT_REPLISH_SIZE
    ):
        """
        A base class for pose generators.

        Args:
            seed (int | None): The seed to use for the random number generator.
            replenish_size (int): The number of samples to replenish the buffer with.
        """
        self.rng = np.random.default_rng(seed)
        self.replenish_size = replenish_size
        self.sample_buffer: NDArray = np.empty(0, dtype=np.float32)

    def __iter__(self):
        while True:
            sample = self.sample_buffer[0]
            self.sample_buffer = self.sample_buffer[1:]
            yield sample
    
    def sample(self, num_samples: int = 1) -> NDArray:
        """Sample a batch of poses.

        Args:
            num_samples (int): The number of samples to sample.

        Returns:
            NDArray: A batch of poses.
        """
        while len(self.sample_buffer) < num_samples:
            self.replenish()
        
        samples = self.sample_buffer[:num_samples]
        self.sample_buffer = self.sample_buffer[num_samples:]
        return samples
    
    def replenish(self) -> None:
        """
        Replenish the sample buffer with new samples.
        Child classes should implement this method.
        """
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class OrientationGeneratorConst(PoseGenerator):
    def __init__(self, orientation: NDArray | float, seed=None, replenish_size=DEFAULT_REPLISH_SIZE, degrees=False):
        super().__init__(seed, replenish_size)
        if isinstance(orientation, float):
            self.orientation = np.eye(4)
            self.orientation[:3, :3] = R.from_euler('z', orientation, degrees=degrees).as_matrix()
        else:
            self.orientation = orientation
        self.sample_buffer: NDArray | list = np.empty((0, 4, 4), dtype=np.float32)

    def replenish(self) -> None:
        self.sample_buffer = np.concatenate([self.sample_buffer, np.tile(self.orientation, (self.replenish_size, 1))])

    def __next__(self):
        return self.sample(1)[0]

class OrientationGeneratorUniformAroundZ(PoseGenerator):
    def __init__(self, lower=0.0, upper=2.0 * np.pi, seed=None, replenish_size=DEFAULT_REPLISH_SIZE, degrees=False):
        super().__init__(seed, replenish_size)
        self.lower = lower
        self.upper = upper
        if degrees:
            self.lower = np.deg2rad(self.lower)
            self.upper = np.deg2rad(self.upper)
        self.sample_buffer: NDArray | list = np.empty((0, 4, 4), dtype=np.float32)

    def replenish(self) -> None:
        rotation_matrices = sample_random_z_rotations(lower=self.lower, upper=self.upper, num_samples=self.replenish_size)
        self.sample_buffer = np.concatenate([self.sample_buffer, rotation_matrices])

    def __next__(self):
        return self.sample(1)[0]


class OrientationGeneratorStablePoses(PoseGenerator):
    def __init__(self, asset, seed=None, replenish_size: int = DEFAULT_REPLISH_SIZE, z_rotation: bool = True, **kwargs):
        """
        Generator that yields random homogeneous transformations that represent stable poses of an asset.
        Internally, calls sample_stable_pose of asset.

        Args:
            asset (scene_synth.Asset): An asset with a `sample_stable_pose(seed)` function.
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
            replenish_size (int): The number of samples to replenish the buffer with.
            **kwargs: Additional arguments to pass to the `Assets.compute_stable_poses` method.
        """
        super().__init__(seed, replenish_size)
        self.asset = asset
        self.kwargs = kwargs
        self.z_rotation = z_rotation
        self.sample_buffer: NDArray | list = np.empty((0, 4, 4), dtype=np.float32)

    def replenish(self) -> None:
        rotation_matrices = self.sample_buffer.extend(
            self.asset.sample_stable_pose(seed=self.rng, num_samples=self.replenish_size, z_rotation=self.z_rotation, **self.kwargs)
        )
        self.sample_buffer = np.concatenate([self.sample_buffer, rotation_matrices])

    def __next__(self):
        return self.sample(1)[0]
    

class OrientationGeneratorFaceTo(PoseGenerator):
    # TODO: it is not used because face to requires reading the scene graph
    def __init__(self, face_to_object_id: str, seed=None, replenish_size=DEFAULT_REPLISH_SIZE):
        super().__init__(seed, replenish_size)
        self.face_to_object_id = face_to_object_id
        self.sample_buffer: NDArray | list = np.empty((0, 4, 4), dtype=np.float32)
    
    def replenish(self) -> None:
        raise NotImplementedError


class PositionIterator3D(PoseGenerator):
    def __init__(self, seed=None, replenish_size=DEFAULT_REPLISH_SIZE):
        super().__init__(seed, replenish_size)
        self.container = None
        self.sample_buffer: NDArray | list = np.empty((0, 3), dtype=np.float32)

    def __iter__(self):
        return self

    def __call__(self, container):
        self.container = container
        return self

    def update(self, *args, **kwargs):
        pass

    def replenish(self) -> None:
        raise NotImplementedError

class PositionIteratorUniform3D(PositionIterator3D):
    def __next__(self):
        return self.sample(1)[0]
            
    def replenish(self) -> None:
        self.sample_buffer = np.concatenate([
            self.sample_buffer, 
            sample_volume_mesh(self.container.geometry, count=self.replenish_size, seed=self.rng)
        ])


class PositionIterator2D(PoseGenerator):
    """Overriding the PositionIterator2D class in scene_synthesizer."""
    def __init__(self, seed=None, replenish_size=DEFAULT_REPLISH_SIZE, xy_limit=None, erosion_distance:float=0.0):
        super().__init__(seed, replenish_size)
        self.polygon: Polygon = None
        self.support: SupportSurface = None
        self.sample_buffer: NDArray | list = np.empty((0, 2), dtype=np.float32)
        self.xy_limit = xy_limit
        self.erosion_distance = erosion_distance

    def __iter__(self):
        return self

    def __call__(self, support):
        self.polygon = support.polygon
        
        if (not self.polygon.is_empty) and self.xy_limit is not None:
            minvx, minvy, maxvx, maxvy = self.polygon.exterior.bounds
            low_x = self.xy_limit[0][0] * (maxvx - minvx) + minvx
            low_y = self.xy_limit[0][1] * (maxvy - minvy) + minvy
            high_x = self.xy_limit[1][0] * (maxvx - minvx) + minvx
            high_y = self.xy_limit[1][1] * (maxvy - minvy) + minvy
            xy_polygon = Polygon([
                (low_x, low_y),
                (high_x, low_y),
                (high_x, high_y),
                (low_x, high_y),
            ])
            cropped_polygon = self.polygon.intersection(xy_polygon)
            # TODO: deal with MultiPolygon, headache
            if isinstance(cropped_polygon, MultiPolygon):
                cropped_polygon = cropped_polygon.geoms[0]
            self.polygon = cropped_polygon
        
        if self.erosion_distance > 0:
            self.polygon = self.polygon.buffer(-self.erosion_distance)

        self.support = support
        self.sample_buffer = np.empty((0, 2), dtype=np.float32) # reset buffer if changed support
        return self

    def update(self, *args, **kwargs):
        pass

    def replenish(self) -> None:
        raise NotImplementedError
    
class PositionIteratorNone(PositionIterator2D):
    """None position iterator for empty polygon"""
    def replenish(self) -> None:
        # all sampled values are nan
        self.sample_buffer = np.concatenate([
            self.sample_buffer, 
            np.full((self.replenish_size, 2), np.nan, dtype=np.float32)
        ])

    def __call__(self, *args, **kwargs):
        # do nothing
        pass


class PositionIteratorUniform(PositionIterator2D):
    """Overriding the PositionIteratorUniform class in scene_synthesizer."""
    def __next__(self):
        return self.sample(1)[0]
            
    def replenish(self) -> None:
        self.sample_buffer = np.concatenate([
            self.sample_buffer, 
            sample_polygon(self.polygon, count=self.replenish_size, seed=self.rng)
        ])


class PositionIteratorDisk(PositionIterator2D):
    """Overriding the PositionIteratorDisk class in scene_synthesizer."""
    def __init__(self, r: float, center: NDArray, seed: int | None = None):
        """Sample positions on a 2D plane uniformly within a disk of radius r and center"""
        super().__init__(seed=seed)
        self.r = r
        self.center = center

    def replenish(self) -> None:
        alpha = np.pi * 2 * self.rng.uniform(0, 1, size=self.replenish_size)[np.newaxis, :]
        radius = self.r * np.sqrt(self.rng.uniform(0, 1, size=self.replenish_size))[np.newaxis, :]
        candidate = np.concatenate([radius * np.cos(alpha), radius * np.sin(alpha)], axis=0) + self.center
        candidate = candidate[check_within_polygon(self.polygon, candidate)]

        self.sample_buffer = np.concatenate([self.sample_buffer, candidate])


class PositionIteratorList(PositionIterator2D):
    """Overriding the PositionIteratorList class in scene_synthesizer."""
    def __init__(self, seed=None, replenish_size=DEFAULT_REPLISH_SIZE, positions: NDArray = None):
        super().__init__(seed, replenish_size)
        self.positions = positions # (n, 2)

    def replenish(self) -> None:
        to_add = self.positions[:]
        # exp expand
        while len(to_add) + len(self.sample_buffer) < self.replenish_size:
            to_add = np.concatenate([to_add, to_add[:]], axis=0)
        self.sample_buffer = np.concatenate([self.sample_buffer, to_add], axis=0)


class PositionIteratorGaussian(PositionIterator2D):
    """Overriding the PositionIteratorGaussian class in scene_synthesizer."""
    def __init__(self, seed=None, replenish_size=DEFAULT_REPLISH_SIZE, params: NDArray | None = None):
        super().__init__(seed, replenish_size)
        self.params = params

    def replenish(self) -> None:
        points = self.rng.normal(
            loc=np.array(self.params[:2])
            + np.array([self.polygon.centroid.x, self.polygon.centroid.y]),
            scale=self.params[2:],
            size=self.replenish_size
        )
        points = points[check_within_polygon(self.polygon, points)]
        self.sample_buffer = np.concatenate([self.sample_buffer, points])
    
    def __next__(self):
        return self.sample(1)[0]


class PositionIteratorGrid(PositionIterator2D):
    """Overriding the PositionIteratorGrid class in scene_synthesizer."""
    def __init__(
        self,
        step_x: float,
        step_y: float,
        noise_std_x: float = 0.0,
        noise_std_y: float = 0.0,
        direction: str = "x",
        stop_on_new_line: bool = False,
        seed: int | None = None,
        replenish_size: int = DEFAULT_REPLISH_SIZE,
    ):
        """
        Args:
            step_x (float): The step size in the x direction.
            step_y (float): The step size in the y direction.
            noise_std_x (float): The standard deviation of the noise in the x direction.
            noise_std_y (float): The standard deviation of the noise in the y direction.
            direction (str): The direction to sample in.
            stop_on_new_line (bool): Whether to stop on a new line.
            seed (int | None): The seed to use for the random number generator.
            replenish_size (int): The number of samples to replenish the buffer with.
        """
        super().__init__(seed=seed, replenish_size=replenish_size)
        self.step = np.array([step_x, step_y])
        self.noise_std_x = noise_std_x
        self.noise_std_y = noise_std_y
        self.direction = direction

        self.new_line = False
        self.stop_on_new_line = stop_on_new_line

        self.start_point = None
        self.end_point = None
        self.i = 0
        self.j = 0
        self.grid_points = None

    def replenish(self) -> None:
        if self.noise_std_x > 0 or self.noise_std_y > 0:
            points = self.grid_points + self.rng.normal(
                loc=points,
                scale=[self.noise_std_x, self.noise_std_y]
            )
        else:
            points = self.grid_points
        
        points = points[check_within_polygon(self.polygon, points)]
        self.sample_buffer = np.concatenate([self.sample_buffer, points])

    def __next__(self) -> NDArray:
        return self.sample(1)[0]

    def __call__(self, support):
        if support.polygon != self.polygon:
            self.polygon = support.polygon

            minx, miny, maxx, maxy = self.polygon.bounds
            self.start_point = np.array([minx, miny])
            self.end_point = np.array([maxx, maxy])
            self.i = 0
            self.j = 0

            self.new_line = False
            self.grid_points = None
            ix = np.arange(0, int((maxx-minx)/self.step[0])+1)
            iy = np.arange(0, int((maxy-miny)/self.step[1])+1)
            all_indices = np.array(np.meshgrid(ix, iy, indexing='ij')).reshape(2, -1).T
            self.grid_points = all_indices * self.step + np.array([minx, miny])

            return self

class PositionIterator2DCollection(PositionIterator2D):
    """A collection of position iterators"""
    def __init__(
        self, 
        seed: int | None = None, 
        position_iterators: list[PositionIterator2D] = [], 
        replenish_size: int = DEFAULT_REPLISH_SIZE
    ):
        """
        Args:
            seed (int | None): The seed to use for the random number generator.
            position_iterators (list[PositionIterator2D]): The position iterators to use.
            replenish_size (int): The number of samples to replenish the buffer with.
        """
        super().__init__(seed=seed, replenish_size=replenish_size)
        self.seed = seed
        self.load_position_iterators(position_iterators)
        self.sample_buffer: NDArray | list = np.empty((0, 2), dtype=np.float32)
        self.support_refs: NDArray = np.empty((0), dtype=object)

    def __iter__(self):
        return self

    def __call__(self, position_iterators):
        self.load_position_iterators(position_iterators)
        return self

    def update(self, *args, **kwargs):
        pass

    def replenish(self) -> None:
        rng = np.random.default_rng(self.seed)

        weights = np.array([pos_iter.polygon.area for pos_iter in self.position_iterators])
        weights = weights / weights.sum()
        counts = np.round(weights * self.replenish_size).astype(int)
        points = np.concatenate([
            pos_iter.sample(count) for pos_iter, count in zip(self.position_iterators, counts)
        ])
        support_refs = np.concatenate([
            np.full(count, pos_iter.support) for pos_iter, count in zip(self.position_iterators, counts)
        ])

        if len(self.position_iterators) > 1:
            indices = rng.permutation(len(points))
            points = points[indices]
            support_refs = support_refs[indices]

        self.sample_buffer = np.concatenate([self.sample_buffer, points])
        self.support_refs = np.concatenate([self.support_refs, support_refs])

    def load_position_iterators(self, position_iterators: list[PositionIterator2D]) -> None:
        """Load a list of position iterators.

        Args:
            position_iterators (list[PositionIterator2D]): The position iterators to load.
        """
        assert position_iterators is not None and len(position_iterators) > 0
        self.position_iterators = []
        for pos_iter in position_iterators:
            # don't nest
            if isinstance(pos_iter, PositionIterator2DCollection):
                raise ValueError("PositionIterator2DCollection cannot be nested")
            # filter out None
            if isinstance(pos_iter, PositionIteratorNone):
                continue
            if hasattr(pos_iter, "polygon") and pos_iter.polygon.area <= 1e-12:
                continue
            self.position_iterators.append(pos_iter)

    def sample(self, num_samples: int = 1) -> dict[str, Any]:
        while len(self.sample_buffer) < num_samples:
            self.replenish()
        
        samples = self.sample_buffer[:num_samples]
        self.sample_buffer = self.sample_buffer[num_samples:]
        support_refs = self.support_refs[:num_samples]
        self.support_refs = self.support_refs[num_samples:]
        return {"samples": samples, "support_refs": support_refs}
