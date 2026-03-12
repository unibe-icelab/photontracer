"""Python bindings for the PhotonTracer GPU ray tracing simulation.

PhotonTracer uses NVIDIA OptiX to trace rays through 3D scenes and compute
light scattering / transport quantities. The typical workflow is:

1. Build a geometry (:class:`MeshGeometry` or :class:`InstanceGeometry`).
2. Define a list of :class:`Material` objects describing how surfaces interact
   with light.
3. Choose a :class:`IRayGenerator` subclass that determines how source rays
   are spawned.
4. Create a :class:`Simulation`, assign the geometry, materials, ray generator
   and output types, then call :meth:`Simulation.run`.
5. Retrieve per-ray results with :meth:`Simulation.get_output_buffer`.

Example::

    import numpy as np
    from photontracer import (
        Simulation, Material, REFRACTIVE,
        ParallelRayGenerator, MeshGeometry, OutputType,
    )

    vertices = np.array([...], dtype=np.float32)   # shape (N, 3)
    faces    = np.array([...], dtype=np.uint32)     # shape (M, 3)

    sim = Simulation(gpu_id=0)
    sim.geometry      = MeshGeometry(vertices, faces)
    sim.materials     = [Material(REFRACTIVE, 1.33+0j)]
    sim.wavelength_um = 0.532
    sim.ray_generator = ParallelRayGenerator(
        number_of_rays=10_000,
        origin=[0, 0, -200],
        direction=[0, 0, 1],
        offset_radius=50,
    )
    sim.outputs = [OutputType.SCATTERING_COUNT, OutputType.LAST_DIRECTION]
    sim.run()

    counts = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
"""
from __future__ import annotations

import typing

import numpy
from numpy.typing import NDArray


__all__ = [
    "GeometryType",
    "MESH",
    "MESH_INSTANCED",
    "LengthUnit",
    "MICRO_METER",
    "MILLI_METER",
    "METER",
    "MaterialType",
    "DIFFUSE",
    "REFRACTIVE",
    "REFLECTIVE",
    "VOLUME_SCATTERING",
    "OutputType",
    "LAST_DIRECTION",
    "LAST_POSITION",
    "RAY_STATE",
    "LAST_MEDIUM_ID",
    "SCATTERING_COUNT",
    "NUMBER_OF_WARNINGS",
    "STOKES_VECTOR",
    "STOKES_VECTOR_IN",
    "OPTICAL_PATH_LENGTH",
    "SOURCE_DIRECTION",
    "SOURCE_POSITION",
    "SCATTERING_ANGLE",
    "Q_MINUS_AXIS_IN",
    "LOGS",
    "LOG_OFFSETS",
    "DIRECTION_HISTOGRAM_HEALPIX",
    "RefractiveIndex",
    "Diffuse",
    "Refractive",
    "Reflective",
    "VolumeScattering",
    "MaterialProperties",
    "Material",
    "IGeometry",
    "MeshGeometry",
    "InstanceGeometry",
    "IRayGenerator",
    "ParallelRayGenerator",
    "IsotropicRayGenerator",
    "CameraRayGenerator",
    "Simulation",
    "is_cuda_available",
]


Vector3 = typing.Union[
    NDArray[numpy.floating],
    NDArray[numpy.integer],
    tuple[float, float, float],
    tuple[int, int, int],
    list[float],
    list[int],
]
"""A 3-element vector accepted by ray generators and simulation properties.

Can be a NumPy array of shape ``(3,)`` (any numeric dtype — cast to float32
internally), a 3-tuple, or a 3-element list of numbers.
"""


class GeometryType:
    """Enum identifying the type of a geometry object.

    Members:

    * ``MESH`` – a single triangle mesh (:class:`MeshGeometry`).
    * ``MESH_INSTANCED`` – multiple mesh instances placed via transform matrices
      (:class:`InstanceGeometry`).
    """

    MESH: GeometryType
    MESH_INSTANCED: GeometryType
    __members__: dict[str, GeometryType]

    def __eq__(self, other: typing.Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __init__(self, value: int) -> None: ...

    def __int__(self) -> int: ...

    def __repr__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class LengthUnit:
    """Enum specifying the physical unit used by geometry coordinates.

    The simulation converts the geometry's unit to micrometres internally
    so that :attr:`Simulation.wavelength_um` is always expressed in μm.

    Members:

    * ``MICRO_METER`` – coordinates are in micrometres (μm).
    * ``MILLI_METER`` – coordinates are in millimetres (mm).
    * ``METER`` – coordinates are in metres (m).
    """

    MICRO_METER: LengthUnit
    MILLI_METER: LengthUnit
    METER: LengthUnit
    __members__: dict[str, LengthUnit]

    def __eq__(self, other: typing.Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __init__(self, value: int) -> None: ...

    def __int__(self) -> int: ...

    def __repr__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class MaterialType:
    """Enum specifying how a surface interacts with incident rays.

    Pass one of these values as the first argument to :class:`Material`.

    Members:

    * ``DIFFUSE`` – Lambertian diffuse reflection. Requires an ``albedo``
      parameter in ``[0, 1]``.
    * ``REFRACTIVE`` – Dielectric refraction and Fresnel reflection described by
      a complex refractive index *n + ik*. Requires a ``refractive_index``
      parameter.
    * ``REFLECTIVE`` – Specular (metallic) reflection with optional roughness.
      Requires a ``reflectivity`` in ``[0, 1]`` and optionally a ``fuzziness``
      in ``[0, 1)``.
    * ``VOLUME_SCATTERING`` – Participating medium described by a complex
      refractive index, a scattering coefficient, and a Henyey–Greenstein
      asymmetry parameter *g*.
    """

    DIFFUSE: MaterialType
    REFRACTIVE: MaterialType
    REFLECTIVE: MaterialType
    VOLUME_SCATTERING: MaterialType
    __members__: dict[str, MaterialType]

    def __eq__(self, other: typing.Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __init__(self, value: int) -> None: ...

    def __int__(self) -> int: ...

    def __repr__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class OutputType:
    """Enum selecting which per-ray quantity to store in an output buffer.

    Assign a list of these values to :attr:`Simulation.outputs` before calling
    :meth:`Simulation.run`, then read results with
    :meth:`Simulation.get_output_buffer`.

    Members:

    * ``LAST_DIRECTION`` – ``float32[N, 3]`` — unit direction vector at the
      final scattering/absorption event for each ray.
    * ``LAST_POSITION`` – ``float32[N, 3]`` — position at the final event.
    * ``RAY_STATE`` – ``int32[N]`` — final state code (e.g. absorbed,
      escaped).
    * ``LAST_MEDIUM_ID`` – ``int32[N]`` — index into the materials list of the
      medium the ray ended in.
    * ``SCATTERING_COUNT`` – ``uint32[N]`` — total number of scattering events
      experienced by each ray.
    * ``NUMBER_OF_WARNINGS`` – ``uint32[N]`` — diagnostic warning count per
      ray (e.g. exceeded max scattering count).
    * ``STOKES_VECTOR`` – ``float32[N, 4]`` — final Stokes vector
      *[I, Q, U, V]* for each ray.
    * ``STOKES_VECTOR_IN`` – ``float32[N, 4]`` — Stokes vector
      *[I, Q, U, V]* at the ray source.
    * ``OPTICAL_PATH_LENGTH`` – ``float64[N]`` — total optical path length
      (distance × refractive index) travelled by each ray.
    * ``SOURCE_DIRECTION`` – ``float32[N, 3]`` — initial ray direction at the
      source.
    * ``SOURCE_POSITION`` – ``float32[N, 3]`` — initial ray position at the
      source.
    * ``SCATTERING_ANGLE`` – ``float32[N]`` — angle (radians) between the
      source direction and the final direction.
    * ``Q_MINUS_AXIS_IN`` – ``float32[N, 3]`` — the Q− reference axis used
      for the source Stokes vector.
    * ``LOGS`` – ``char[N, LOG_BYTES_PER_RAY]`` — raw per-ray debug log
      strings (only available when debug logging is compiled in).
    * ``LOG_OFFSETS`` – ``uint32[N]`` — byte offsets into the ``LOGS`` buffer
      for each ray.
    * ``DIRECTION_HISTOGRAM_HEALPIX`` – ``uint32[npix]`` — histogram counting
      how many rays ended in each HEALPix pixel (requires
      :attr:`Simulation.direction_healpix_nside` > 0).
    """

    LAST_DIRECTION: OutputType
    LAST_POSITION: OutputType
    RAY_STATE: OutputType
    LAST_MEDIUM_ID: OutputType
    SCATTERING_COUNT: OutputType
    NUMBER_OF_WARNINGS: OutputType
    STOKES_VECTOR: OutputType
    STOKES_VECTOR_IN: OutputType
    OPTICAL_PATH_LENGTH: OutputType
    SOURCE_DIRECTION: OutputType
    SOURCE_POSITION: OutputType
    SCATTERING_ANGLE: OutputType
    Q_MINUS_AXIS_IN: OutputType
    LOGS: OutputType
    LOG_OFFSETS: OutputType
    DIRECTION_HISTOGRAM_HEALPIX: OutputType
    __members__: dict[str, OutputType]

    def __eq__(self, other: typing.Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __init__(self, value: int) -> None: ...

    def __int__(self) -> int: ...

    def __repr__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class IGeometry:
    """Abstract base class for all geometry types.

    Do not instantiate directly; use one of the concrete subclasses:
    :class:`MeshGeometry` or :class:`InstanceGeometry`.
    """

    def get_type(self) -> GeometryType:
        """Return the geometry type enum value for this object."""
        ...

    def free_device_memory(self) -> None:
        """Release the OptiX acceleration structure buffers on the GPU.

        Call this after the simulation is no longer needed to reclaim device
        memory.  The geometry object becomes unusable afterwards.
        """
        ...


class MeshGeometry(IGeometry):
    """A single closed or open triangle mesh.

    Args:
        vertices: Vertex positions.  Shape ``(V, 3)``, any floating-point dtype
            (cast to ``float32`` internally).  ``float64`` arrays from trimesh
            are accepted directly.
        indices: Triangle index table.  Either shape ``(T, 3)`` or a flat
            array of length ``3*T``, any integer dtype (cast to ``uint32``
            internally).  Each row/triple gives the three vertex indices of
            one triangle.

    Example::

        import numpy as np
        from photontracer import MeshGeometry

        verts = np.array([[-1,-1,0],[1,-1,0],[0,1,0]], dtype=np.float32)
        faces = np.array([[0,1,2]], dtype=np.uint32)
        geo = MeshGeometry(verts, faces)
    """

    def __init__(self, vertices: NDArray[numpy.floating],
                 indices: NDArray[numpy.integer]) -> None: ...


class InstanceGeometry(IGeometry):
    """Multiple instances of one or more sub-geometries placed via transforms.

    Use this geometry type to efficiently model scenes with many copies of a
    small set of particle shapes (e.g. ice crystals of several habits at
    arbitrary orientations).

    Args:
        sub_geometries: List of prototype geometries (e.g.
            :class:`MeshGeometry` objects).  Each entry corresponds to one
            particle type.
        instance_transforms: Affine 3×4 row-major transform matrices for every
            instance.  Shape ``(N, 3, 4)``, any floating-point dtype (cast to
            ``float32`` internally).
        particle_type_ids: Index into ``sub_geometries`` for each instance.
            Shape ``(N,)``, any integer dtype (cast to ``uint32`` internally).
        material_ids: Index into :attr:`Simulation.materials` for each
            instance.  Shape ``(N,)``, any integer dtype (cast to ``uint32``
            internally).

    Example::

        import numpy as np
        from photontracer import MeshGeometry, InstanceGeometry

        proto = MeshGeometry(vertices, faces)

        # Identity transform for two instances
        transforms = np.tile(np.eye(3, 4, dtype=np.float32), (2, 1, 1))
        type_ids   = np.array([0, 0], dtype=np.uint32)
        mat_ids    = np.array([0, 1], dtype=np.uint32)

        geo = InstanceGeometry([proto], transforms, type_ids, mat_ids)
    """

    def __init__(
        self,
        sub_geometries: typing.Sequence[IGeometry],
        instance_transforms: NDArray[numpy.floating],
        particle_type_ids: NDArray[numpy.integer],
        material_ids: NDArray[numpy.integer],
    ) -> None: ...


class RefractiveIndex:
    """Complex refractive index *n + ik* stored as two ``float32`` fields.

    The real part ``r`` is the phase refractive index *n* (≥ 1 for most
    optical media).  The imaginary part ``i`` is the extinction coefficient
    *k* (≥ 0; controls absorption).

    Attributes:
        r: Real part of the refractive index (*n*).
        i: Imaginary part of the refractive index (*k*, extinction
           coefficient).

    The :attr:`complex` property provides a convenient ``complex`` view that
    can be read and set using Python's native complex literals (e.g.
    ``1.33+0j``).

    Example::

        ri = RefractiveIndex(1.31, 0.0)            # non-absorbing ice
        ri = RefractiveIndex(complex_value=1.5+0.01j)
        ri.complex = 1.33+0j
    """

    r: float
    i: float

    @typing.overload
    def __init__(self) -> None:
        """Create a default RefractiveIndex with ``r=0`` and ``i=0``."""
        ...

    @typing.overload
    def __init__(self, r: float, i: float) -> None:
        """Create a RefractiveIndex from separate real and imaginary parts.

        Args:
            r: Real part (*n*).
            i: Imaginary part (*k*).
        """
        ...

    @typing.overload
    def __init__(self, complex_value: complex) -> None:
        """Create a RefractiveIndex from a Python complex number.

        Args:
            complex_value: Complex refractive index, e.g. ``1.31+0j``.
        """
        ...

    @property
    def complex(self) -> complex:
        """The refractive index as a Python ``complex`` number (*n + ik*)."""
        ...

    @complex.setter
    def complex(self, value: complex) -> None: ...

    def __repr__(self) -> str: ...


class Diffuse:
    """Properties for a Lambertian (diffuse) surface.

    Attributes:
        albedo: Fraction of incident light diffusely reflected, in ``[0, 1]``.
            ``0`` is perfectly black; ``1`` is a perfect Lambertian reflector.
    """

    albedo: float

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class Refractive:
    """Properties for a dielectric (refractive) surface.

    Fresnel reflection and transmission are computed from the complex
    refractive index and the current medium's refractive index.

    Attributes:
        refractive_index: Complex refractive index *n + ik* of the material.
    """

    refractive_index: RefractiveIndex

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class Reflective:
    """Properties for a metallic (specular) surface with optional roughness.

    Attributes:
        reflectivity: Fraction of incident light that is specularly reflected,
            in ``[0, 1]``.
        fuzziness: Surface roughness parameter in ``[0, 1)``.  ``0`` produces
            a perfect mirror; values approaching ``1`` scatter reflected rays
            over nearly a hemisphere.
    """

    reflectivity: float
    fuzziness: float

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class VolumeScattering:
    """Properties for a participating (volume scattering) medium.

    Rays entering a surface with this material type travel through the medium
    and scatter according to the Henyey–Greenstein phase function until they
    are absorbed or escape.

    Attributes:
        refractive_index: Complex refractive index of the medium.  The
            imaginary part *k* encodes absorption.
        scattering_coefficient: Scattering coefficient in units of inverse
            length (matching :attr:`Simulation.length_unit`).
        asymetry_parameter: Henyey–Greenstein asymmetry parameter *g*,
            in ``(-1, 1)``.  ``0`` is isotropic; ``>0`` forward-scattering;
            ``<0`` back-scattering.
    """

    refractive_index: RefractiveIndex
    scattering_coefficient: float
    asymetry_parameter: float

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class MaterialProperties:
    """Union of all material property sub-structs.

    Only the sub-struct corresponding to the :attr:`Material.type` of the
    owning :class:`Material` is meaningful at runtime.  The four properties
    ``diffuse``, ``refractive``, ``reflective``, and ``volume_scattering``
    share the same underlying GPU buffer.

    Attributes:
        diffuse: Diffuse properties (used when ``type == DIFFUSE``).
        refractive: Refractive properties (used when ``type == REFRACTIVE``).
        reflective: Reflective properties (used when ``type == REFLECTIVE``).
        volume_scattering: Volume scattering properties
            (used when ``type == VOLUME_SCATTERING``).
    """

    diffuse: Diffuse
    refractive: Refractive
    reflective: Reflective
    volume_scattering: VolumeScattering

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...


class Material:
    """Describes how a surface or volume interacts with traced rays.

    A material combines a :class:`MaterialType` with the parameters relevant
    to that type.  Use one of the convenience constructors below rather than
    manually populating :attr:`properties`.

    Attributes:
        type: The interaction model (:class:`MaterialType`).
        properties: Raw union of all property sub-structs
            (:class:`MaterialProperties`).

    Constructors:

    * ``Material()`` – default (uninitialized) material.
    * ``Material(DIFFUSE, albedo)`` – Lambertian reflector with the given
      albedo in ``[0, 1]``.
    * ``Material(REFLECTIVE, reflectivity)`` – perfect mirror with zero
      fuzziness.
    * ``Material(REFLECTIVE, reflectivity, fuzziness)`` – fuzzy mirror.
    * ``Material(REFRACTIVE, refractive_index)`` – dielectric surface; accepts
      a :class:`RefractiveIndex`, a ``complex``, or a ``float`` for the
      refractive index.
    * ``Material(VOLUME_SCATTERING, refractive_index, scattering_coefficient, g)``
      – participating medium.

    Example::

        from photontracer import Material, DIFFUSE, REFRACTIVE, VOLUME_SCATTERING

        m_diffuse  = Material(DIFFUSE, albedo=0.9)
        m_glass    = Material(REFRACTIVE, 1.5+0j)
        m_ice      = Material(REFRACTIVE, 1.31+1e-9j)
        m_cloud    = Material(VOLUME_SCATTERING, 1.33+0j,
                              scattering_coefficient=100.0, g=0.85)
    """

    type: MaterialType
    properties: MaterialProperties

    @typing.overload
    def __init__(self) -> None: ...

    @typing.overload
    def __init__(self, type: MaterialType, albedo: float) -> None:
        """Create a DIFFUSE material with the given albedo.

        Args:
            type: Must be ``DIFFUSE``.
            albedo: Reflectance in ``[0, 1]``.
        """
        ...

    @typing.overload
    def __init__(self, type: MaterialType, reflectivity: float,
                 fuzziness: float) -> None:
        """Create a REFLECTIVE material with reflectivity and fuzziness.

        Args:
            type: Must be ``REFLECTIVE``.
            reflectivity: Fraction of specularly reflected light in ``[0, 1]``.
            fuzziness: Surface roughness in ``[0, 1)``.
        """
        ...

    @typing.overload
    def __init__(self, type: MaterialType, reflectivity: float) -> None:
        """Create a REFLECTIVE material with zero fuzziness (perfect mirror).

        Args:
            type: Must be ``REFLECTIVE``.
            reflectivity: Fraction of specularly reflected light in ``[0, 1]``.
        """
        ...

    @typing.overload
    def __init__(self, type: MaterialType,
                 refractive_index: RefractiveIndex) -> None:
        """Create a REFRACTIVE material from a RefractiveIndex object.

        Args:
            type: Must be ``REFRACTIVE``.
            refractive_index: Complex refractive index.
        """
        ...

    @typing.overload
    def __init__(self, type: MaterialType,
                 refractive_index: complex) -> None:
        """Create a REFRACTIVE material from a Python complex number.

        Args:
            type: Must be ``REFRACTIVE``.
            refractive_index: Complex refractive index, e.g. ``1.5+0j``.
        """
        ...

    @typing.overload
    def __init__(
        self,
        type: MaterialType,
        refractive_index: complex,
        scattering_coefficient: float,
        asymetry_parameter: float,
    ) -> None:
        """Create a VOLUME_SCATTERING medium.

        Args:
            type: Must be ``VOLUME_SCATTERING``.
            refractive_index: Complex refractive index of the medium.
            scattering_coefficient: Scattering coefficient (inverse length
                units matching :attr:`Simulation.length_unit`).
            asymetry_parameter: Henyey–Greenstein *g* parameter in
                ``(-1, 1)``.
        """
        ...

    def __repr__(self) -> str: ...


class IRayGenerator:
    """Abstract base class for ray generators.

    A ray generator determines the number, origin, and initial direction of
    the photons traced by the simulation.  Use one of the concrete subclasses:

    * :class:`ParallelRayGenerator` – collimated beam (plane wave).
    * :class:`IsotropicRayGenerator` – rays emmitted uniformly from all directions outside of a sphere towards the scene.
    * :class:`CameraRayGenerator` – pinhole camera model for rendering.

    Assign a concrete instance to :attr:`Simulation.ray_generator` before
    calling :meth:`Simulation.run`.
    """
    pass


class ParallelRayGenerator(IRayGenerator):
    """A collimated (parallel) beam of rays — equivalent to a plane wave.

    All rays originate from a disk of radius ``offset_radius`` centred at
    ``origin`` and travel in the same ``direction``.  The direction vector is
    normalised automatically.

    Args:
        number_of_rays: Total number of rays to launch.
        origin: Centre of the source disk (:data:`Vector3`).
        direction: Propagation direction (:data:`Vector3`); normalised
            automatically.
        offset_radius: Radius of the circular source disk in geometry units.
            Pass ``0`` for a single-point source.

    Example::

        from photontracer import ParallelRayGenerator
        import numpy as np

        gen = ParallelRayGenerator(
            number_of_rays=100_000,
            origin=np.array([0, 0, -500], dtype=np.float32),
            direction=[0, 0, 1],
            offset_radius=50.0,
        )
    """

    def __init__(
        self,
        number_of_rays: int,
        origin: Vector3,
        direction: Vector3,
        offset_radius: float,
    ) -> None: ...

    @property
    def origin(self) -> Vector3:
        """Centre of the source disk."""
        ...

    @origin.setter
    def origin(self, value: Vector3) -> None: ...

    @property
    def direction(self) -> Vector3:
        """Unit propagation direction of the beam."""
        ...

    @direction.setter
    def direction(self, value: Vector3) -> None: ...

    @property
    def offset_radius(self) -> float:
        """Radius of the circular source disk in geometry units."""
        ...

    @offset_radius.setter
    def offset_radius(self, value: float) -> None: ...


class IsotropicRayGenerator(IRayGenerator):
    """An source that emits rays isotropically from outside of a sphere towards the scene.

    Rays are spawned from randomly sampled points on a sphere of radius
    ``source_radius`` around the ``center``, pointing toward the center.  
    ``offset_radius`` can be used to add a tangential offset to the ray origins.  
    The launch shape is ``(number_of_rays,)``, 
    and output buffers will be 1-D arrays of length ``number_of_rays``.

    Args:
        number_of_rays: Total number of rays to launch.
        center: Centre of the sphere (:data:`Vector3`).
        source_radius: Radius of the spherical source from which rays are emitted towards the center.
        offset_radius: Tangential offset radius in geometry units. Rays origins are offset in the tangential plane, 
        within a sphere with radius ``offset_radius`` around the intersection point.  
        Pass ``0`` so that rays are emmitted from the sphere surface and would intersect the center.

    Example::

        gen = IsotropicRayGenerator(
            number_of_rays=1_000_000,
            center=[0, 0, 0],
            source_radius=1.0,
            offset_radius=1.0,
        )
    """

    def __init__(
        self,
        number_of_rays: int,
        center: Vector3,
        source_radius: float,
        offset_radius: float,
    ) -> None: ...

    @property
    def center(self) -> Vector3:
        """Centre of the spherical source region."""
        ...

    @center.setter
    def center(self, value: Vector3) -> None: ...

    @property
    def source_radius(self) -> float:
        """Radius of the sphere from which rays are emitted towards the center."""
        ...

    @source_radius.setter
    def source_radius(self, value: float) -> None: ...

    @property
    def offset_radius(self) -> float:
        """Tangential offset radius in geometry units."""
        ...

    @offset_radius.setter
    def offset_radius(self, value: float) -> None: ...


class CameraRayGenerator(IRayGenerator):
    """A perspective pinhole camera that generates rays for image rendering.

    The launch shape is ``(samples_per_pixel, image_width, image_height)``,
    so the total number of device threads equals
    ``samples_per_pixel * image_width * image_height``.  Output buffer shapes
    will match this 3-D layout.

    All angle properties are in **degrees**.

    Example::

        cam = CameraRayGenerator()
        cam.image_width      = 800
        cam.image_height     = 600
        cam.samples_per_pixel = 16
        cam.vertical_fov     = 45.0
        cam.look_from        = [0, 0, -500]
        cam.look_at          = [0, 0, 0]
        cam.vertical_up      = [0, 1, 0]
        cam.focus_distance   = 500.0
    """

    def __init__(self) -> None: ...

    @property
    def image_width(self) -> int:
        """Output image width in pixels."""
        ...

    @image_width.setter
    def image_width(self, value: int) -> None: ...

    @property
    def image_height(self) -> int:
        """Output image height in pixels."""
        ...

    @image_height.setter
    def image_height(self, value: int) -> None: ...

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height).  Setting this updates the camera frame."""
        ...

    @aspect_ratio.setter
    def aspect_ratio(self, value: float) -> None: ...

    @property
    def samples_per_pixel(self) -> int:
        """Number of stochastic samples per pixel (anti-aliasing / Monte Carlo)."""
        ...

    @samples_per_pixel.setter
    def samples_per_pixel(self, value: int) -> None: ...

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view in degrees."""
        ...

    @vertical_fov.setter
    def vertical_fov(self, value: float) -> None: ...

    @property
    def look_from(self) -> Vector3:
        """Camera position in world space."""
        ...

    @look_from.setter
    def look_from(self, value: Vector3) -> None: ...

    @property
    def look_at(self) -> Vector3:
        """Point the camera is aimed at in world space."""
        ...

    @look_at.setter
    def look_at(self, value: Vector3) -> None: ...

    @property
    def vertical_up(self) -> Vector3:
        """World-space "up" vector used to orient the camera roll."""
        ...

    @vertical_up.setter
    def vertical_up(self, value: Vector3) -> None: ...

    @property
    def defocus_angle(self) -> float:
        """Half-angle of the lens aperture cone in degrees.  ``0`` disables depth-of-field."""
        ...

    @defocus_angle.setter
    def defocus_angle(self, value: float) -> None: ...

    @property
    def focus_distance(self) -> float:
        """Distance from the camera to the plane of perfect focus."""
        ...

    @focus_distance.setter
    def focus_distance(self, value: float) -> None: ...


class Simulation:
    """Main entry point for GPU ray tracing simulations.

    A ``Simulation`` wraps an NVIDIA OptiX device context and manages the full
    rendering pipeline: geometry acceleration structures, ray generation, CUDA
    kernel launch, and output buffer retrieval.

    Args:
        gpu_id: Zero-based index of the CUDA device to use (default ``0``).
        optix_logging_level: Verbosity of OptiX internal messages, ``0``
            (silent) to ``4`` (verbose).  Default ``1``.
        enable_validation_mode: Enable OptiX validation mode for additional
            runtime checks during development.  Significantly slower; leave
            ``False`` in production.

    Typical usage::

        sim = Simulation(gpu_id=0)

        sim.geometry      = MeshGeometry(vertices, faces)
        sim.materials     = [Material(REFRACTIVE, 1.33+0j)]
        sim.wavelength_um = 0.532          # green light
        sim.length_unit   = MICRO_METER
        sim.ray_generator = ParallelRayGenerator(100_000, [0,0,-200], [0,0,1], 50)
        sim.outputs       = [OutputType.SCATTERING_COUNT, OutputType.LAST_DIRECTION]
        sim.run()

        counts = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
    """

    def __init__(
        self,
        gpu_id: int = 0,
        optix_logging_level: int = 1,
        enable_validation_mode: bool = False,
    ) -> None: ...

    def run(self) -> None:
        """Build the pipeline and launch the ray tracing kernel.

        This method (re-)compiles the OptiX pipeline if any simulation
        parameter has changed since the last call, uploads all configuration
        to the device, launches the CUDA kernel, and waits for completion.
        Output buffers are then available via :meth:`get_output_buffer`.

        All of the following must be set before calling ``run()``:

        * :attr:`geometry`
        * :attr:`materials`
        * :attr:`ray_generator`
        * :attr:`outputs`
        """
        ...

    def free_device_memory(self) -> None:
        """Release all device-side buffers held by this simulation.

        Frees the OptiX pipeline, output buffers, and geometry acceleration
        structures.  :meth:`run` can be called again after this (it will
        reallocate everything from scratch).
        """
        ...

    def calculate_volume_fraction(self, box_min: Vector3, box_max: Vector3,
                                  number_of_samples: int) -> float:
        """Estimate the volume fraction of geometry inside an axis-aligned box.

        Uses Monte Carlo point sampling: ``number_of_samples`` random points
        are drawn uniformly inside the box and the fraction that fall inside
        the geometry is returned.

        Args:
            box_min: Minimum corner of the sampling box (:data:`Vector3`).
            box_max: Maximum corner of the sampling box (:data:`Vector3`).
            number_of_samples: Number of random sample points.  Higher values
                give a more accurate estimate at the cost of computation time.

        Returns:
            Estimated volume fraction in ``[0, 1]``.
        """
        ...

    def get_output_buffer(self, output_type: OutputType) -> numpy.ndarray:
        """Copy an output buffer from the GPU and return it as a NumPy array.

        Must be called **after** :meth:`run`.  The returned array is a fresh
        copy (not a view) of device memory.

        Args:
            output_type: The :class:`OutputType` to retrieve.  Must have been
                included in :attr:`outputs` before the last ``run()`` call.

        Returns:
            NumPy array with shape and dtype determined by the output type:

            ========================= =========== ==================
            OutputType                dtype       shape
            ========================= =========== ==================
            LAST_DIRECTION            float32     ``(N, 3)``
            LAST_POSITION             float32     ``(N, 3)``
            RAY_STATE                 int32       ``(N,)``
            LAST_MEDIUM_ID            int32       ``(N,)``
            SCATTERING_COUNT          uint32      ``(N,)``
            NUMBER_OF_WARNINGS        uint32      ``(N,)``
            STOKES_VECTOR             float32     ``(N, 4)``
            STOKES_VECTOR_IN          float32     ``(N, 4)``
            OPTICAL_PATH_LENGTH       float64     ``(N,)``
            SOURCE_DIRECTION          float32     ``(N, 3)``
            SOURCE_POSITION           float32     ``(N, 3)``
            SCATTERING_ANGLE          float32     ``(N,)``
            Q_MINUS_AXIS_IN           float32     ``(N, 3)``
            DIRECTION_HISTOGRAM_HEALPIX uint32    ``(npix,)``
            ========================= =========== ==================

            where *N* is the number of rays and *npix* is the number of
            HEALPix pixels for the configured ``direction_healpix_nside``.

        Raises:
            RuntimeError: If ``run()`` has not been called yet, or the
                requested output type was not enabled.
        """
        ...

    @property
    def geometry(self) -> IGeometry | None:
        """The root geometry used in the simulation.

        Assign a :class:`MeshGeometry` or
        :class:`InstanceGeometry` before calling :meth:`run`.
        If a MeshGeometry, which contains no material id information, is assigned,
        the material id defaults to 1 for the mesh. Wrapping the mesh in an InstanceGeometry 
        allows to set the material id for each instance.
        """
        ...

    @geometry.setter
    def geometry(self, value: IGeometry) -> None: ...

    @property
    def materials(self) -> list[Material]:
        """List of materials referenced by geometry faces / instances.

        Each geometry primitive or instance addresses a material by its
        zero-based index in this list.  Must contain at least the number of
        distinct material IDs used by the geometry.
        """
        ...

    @materials.setter
    def materials(self, value: list[Material]) -> None: ...

    @property
    def wavelength_um(self) -> float:
        """Simulation wavelength in micrometres (μm).

        Used to compute the scale factor that converts geometry coordinates
        (in :attr:`length_unit`) to μm for absorption and scattering calculations.
        Default ``0.5`` μm (green light).
        """
        ...

    @wavelength_um.setter
    def wavelength_um(self, value: float) -> None: ...

    @property
    def length_unit(self) -> LengthUnit:
        """Physical unit of the geometry coordinates.

        The simulation multiplies all lengths by the appropriate scale factor
        so that :attr:`wavelength_um` (always in μm) is consistent with the
        scattering / absorption calculations.  Default ``MICRO_METER``.
        """
        ...

    @length_unit.setter
    def length_unit(self, value: LengthUnit) -> None: ...

    @property
    def stokes_vector(self) -> tuple[float, float, float, float]:
        """Initial Stokes vector ``[I, Q, U, V]`` of all source rays.

        For unpolarised light use ``(1, 0, 0, 0)`` (default).
        For linearly polarised light along the Q+ direction use
        ``(1, 1, 0, 0)``.
        """
        ...

    @stokes_vector.setter
    def stokes_vector(self, value: typing.Sequence[float]) -> None: ...

    @property
    def q_minus_axis_seed(self) -> tuple[float, float, float]:
        """Reference direction for the Q− axis of the Stokes vector at the source.

        The Q− axis defines the reference frame for the polarisation state.
        This vector is perpendicular to the ray direction and determines the
        orientation of the ``Q`` and ``U`` Stokes components.
        """
        ...

    @q_minus_axis_seed.setter
    def q_minus_axis_seed(self, value: typing.Sequence[float]) -> None: ...

    @property
    def max_scattering_count(self) -> int:
        """Maximum number of scattering events per ray before it is terminated.

        Set to ``0`` (default) to allow unlimited scattering.  A finite value
        is useful to cap computation time.
        """
        ...

    @max_scattering_count.setter
    def max_scattering_count(self, value: int) -> None: ...

    @property
    def max_nested_geometry_levels(self) -> int:
        """Maximum depth of nested :class:`InstanceGeometry` hierarchies.

        Corresponds to the OptiX traversable depth.  Valid range is ``1``–``31``
        (default ``2``).  Increase when the geometry has more than two
        levels of nesting.
        """
        ...

    @max_nested_geometry_levels.setter
    def max_nested_geometry_levels(self, value: int) -> None: ...

    @property
    def max_sub_geometries(self) -> int:
        """Read-only hardware limit on the number of sub-geometries in an :class:`InstanceGeometry`."""
        ...

    @property
    def max_mesh_triangles(self) -> int:
        """Read-only hardware limit on the number of triangles per :class:`MeshGeometry`."""
        ...

    @property
    def seed(self) -> int:
        """Initial seed for the on-device random number generator.

        Changing the seed produces statistically independent simulation runs
        for variance estimation.  Default ``0``.
        """
        ...

    @seed.setter
    def seed(self, value: int) -> None: ...

    @property
    def use_complex_fresnel(self) -> bool:
        """Use the full complex Fresnel equations for refractive materials.

        When ``True``, the simulator applies complex-valued Fresnel
        coefficients, which correctly model absorption at interfaces (e.g. fresnel peak in water ice).  When ``False``, real-valued Fresnel
        equations are used. Mainly interesting for testing, as the complex valued calculation only adds a small overhead.
        Default ``False``.
        """
        ...

    @use_complex_fresnel.setter
    def use_complex_fresnel(self, value: bool) -> None: ...

    @property
    def direction_healpix_nside(self) -> int:
        """HEALPix ``NSIDE`` parameter for the direction histogram output.

        When > ``0``, rays that exit the scene without being absorbed are
        binned into a HEALPix sphere tessellation with this ``NSIDE``, giving
        ``12 * NSIDE ** 2`` pixels.  The histogram is available as
        ``OutputType.DIRECTION_HISTOGRAM_HEALPIX``.

        Set to ``0`` (default) to disable the histogram.
        """
        ...

    @direction_healpix_nside.setter
    def direction_healpix_nside(self, value: int) -> None: ...

    @property
    def ray_generator(self) -> IRayGenerator | None:
        """The ray generator that defines source rays.

        Assign a :class:`ParallelRayGenerator`, :class:`IsotropicRayGenerator`,
        or :class:`CameraRayGenerator` before calling :meth:`run`.
        """
        ...

    @ray_generator.setter
    def ray_generator(self, value: IRayGenerator) -> None: ...

    @property
    def outputs(self) -> list[OutputType]:
        """List of output quantities to compute during the simulation.

        Assign before calling :meth:`run`.  Only the listed
        :class:`OutputType` values will have their device buffers allocated
        and filled; requesting additional types after the fact requires
        another ``run()`` call.

        Example::

            sim.outputs = [
                OutputType.SCATTERING_COUNT,
                OutputType.LAST_DIRECTION,
                OutputType.STOKES_VECTOR,
            ]
        """
        ...

    @outputs.setter
    def outputs(self, value: typing.Sequence[OutputType]) -> None: ...


def is_cuda_available() -> bool:
    """Return ``True`` if at least one CUDA-capable GPU is detected.

    Queries ``cudaGetDeviceCount`` at runtime.  Use this to guard simulation
    code when running on machines that may not have a compatible GPU.

    Example::

        if not is_cuda_available():
            raise RuntimeError("No CUDA GPU found — cannot run simulation.")
    """
    ...


DIFFUSE: MaterialType
REFRACTIVE: MaterialType
REFLECTIVE: MaterialType
VOLUME_SCATTERING: MaterialType

MESH: GeometryType
MESH_INSTANCED: GeometryType

MICRO_METER: LengthUnit
MILLI_METER: LengthUnit
METER: LengthUnit

LAST_DIRECTION: OutputType
LAST_POSITION: OutputType
RAY_STATE: OutputType
LAST_MEDIUM_ID: OutputType
SCATTERING_COUNT: OutputType
NUMBER_OF_WARNINGS: OutputType
STOKES_VECTOR: OutputType
STOKES_VECTOR_IN: OutputType
OPTICAL_PATH_LENGTH: OutputType
SOURCE_DIRECTION: OutputType
SOURCE_POSITION: OutputType
SCATTERING_ANGLE: OutputType
Q_MINUS_AXIS_IN: OutputType
LOGS: OutputType
LOG_OFFSETS: OutputType
DIRECTION_HISTOGRAM_HEALPIX: OutputType
