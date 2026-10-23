"""
Python bindings for the PhotonTracer ray tracing simulation.

PhotonTracer traces rays through 3D scenes, on the GPU with NVIDIA OptiX or on
the CPU with Embree, and computes light scattering and transport quantities.
The typical workflow is:

1. Build a geometry (:class:`MeshGeometry` or :class:`InstanceGeometry`).
2. Define a list of :class:`Material` objects describing how surfaces and
   volumes interact with light.
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

    sim = Simulation()
    sim.geometry      = MeshGeometry(vertices, faces)
    sim.materials     = [Material(REFRACTIVE, 1+0j), Material(REFRACTIVE, 1.33+0j)]
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
import collections.abc
import numpy
import numpy.typing
import photontracer._typing
import typing
__all__: list[str] = ['Backend', 'CameraRayGenerator', 'DIFFUSE', 'DIRECTION_HISTOGRAM_HEALPIX', 'Diffuse', 'DiffuseRayGenerator', 'EMBREE', 'FIRST_POSITION', 'GeometryType', 'IGeometry', 'IRayGenerator', 'InstanceGeometry', 'IsotropicRayGenerator', 'LAST_DIRECTION', 'LAST_MEDIUM_ID', 'LAST_POSITION', 'LengthUnit', 'MESH', 'MESH_INSTANCED', 'METER', 'MICRO_METER', 'MILLI_METER', 'MRG32K3A', 'Material', 'MaterialProperties', 'MaterialType', 'MeshGeometry', 'NUMBER_OF_WARNINGS', 'OPTICAL_PATH_LENGTH', 'OPTIX', 'OutputType', 'PCG32', 'ParallelRayGenerator', 'Q_MINUS_AXIS_IN', 'RAY_STATE', 'REFLECTIVE', 'REFRACTIVE', 'RandomNumberGenerator', 'Reflective', 'Refractive', 'RefractiveIndex', 'SCATTERING_ANGLE', 'SCATTERING_COUNT', 'SOURCE_DIRECTION', 'SOURCE_POSITION', 'STOKES_VECTOR', 'STOKES_VECTOR_IN', 'Simulation', 'TraceEvent', 'VOLUME_SCATTERING', 'VolumeScattering', 'available_backends', 'is_cuda_available', 'is_verbose', 'set_verbose']
class GeometryType:
    """
    Type of a geometry, returned by IGeometry.get_type().
    
    Members:
    
      MESH : A single triangle mesh (MeshGeometry).
    
      MESH_INSTANCED : Instances of geometries placed by transforms (InstanceGeometry).
    """
    MESH: typing.ClassVar[GeometryType]
    MESH_INSTANCED: typing.ClassVar[GeometryType]
    __members__: typing.ClassVar[dict[str, GeometryType]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RandomNumberGenerator:
    """
    Random number generator of the ray tracing kernels.
    
    Assign one of these to :attr:`Simulation.random_number_generator`. Both give
    the same results statistically, but not ray by ray.
    
    
    Members:
    
      PCG32 : PCG32, the default. Cheap to seed, which matters most for scenes with little physics per ray.
    
      MRG32K3A : curand's MRG32k3a, the generator of the 1.0 releases, to reproduce their results ray by ray. OptiX only.
    """
    MRG32K3A: typing.ClassVar[RandomNumberGenerator]
    PCG32: typing.ClassVar[RandomNumberGenerator]
    __members__: typing.ClassVar[dict[str, RandomNumberGenerator]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Backend:
    """
    Where a Simulation traces its rays; pass one as backend to Simulation.
    
    Members:
    
      OPTIX : NVIDIA GPU with OptiX.
    
      EMBREE : CPU with Embree and TBB. Supports only the PCG32 generator.
    """
    EMBREE: typing.ClassVar[Backend]
    OPTIX: typing.ClassVar[Backend]
    __members__: typing.ClassVar[dict[str, Backend]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TraceEvent:
    """
    What happened to a traced ray in one step (event field of Simulation.get_trace()).
    
    Members:
    
      INTERACTION : Reflected, refracted or scattered; the ray goes on.
    
      ESCAPED : The ray left the scene.
    
      ABSORBED : The ray was absorbed.
    
      MAX_SCATTERING : The ray reached Simulation.max_scattering_count.
    
      ERROR : The ray ended on an error.
    """
    ABSORBED: typing.ClassVar[TraceEvent]
    ERROR: typing.ClassVar[TraceEvent]
    ESCAPED: typing.ClassVar[TraceEvent]
    INTERACTION: typing.ClassVar[TraceEvent]
    MAX_SCATTERING: typing.ClassVar[TraceEvent]
    __members__: typing.ClassVar[dict[str, TraceEvent]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LengthUnit:
    """
    Physical unit of the geometry coordinates.
    
    Assign one of these to :attr:`Simulation.length_unit`. Positions, distances
    and the optical path length are in this unit; it is needed to convert the
    imaginary part of a refractive index, given for :attr:`Simulation.wavelength_um`
    in μm, into an absorption coefficient per geometry unit.
    
    
    Members:
    
      MICRO_METER : Micrometres (μm).
    
      MILLI_METER : Millimetres (mm).
    
      METER : Metres (m).
    """
    METER: typing.ClassVar[LengthUnit]
    MICRO_METER: typing.ClassVar[LengthUnit]
    MILLI_METER: typing.ClassVar[LengthUnit]
    __members__: typing.ClassVar[dict[str, LengthUnit]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MaterialType:
    """
    How a surface or volume interacts with rays.
    
    Pass one of these as the first argument to :class:`Material`.
    
    
    Members:
    
      DIFFUSE : Lambertian reflection with an albedo in [0, 1]; the rest is absorbed.
    
      REFRACTIVE : Fresnel reflection and refraction at the surface of a medium with a complex refractive index n + ik.
    
      REFLECTIVE : Specular (metallic) reflection with a reflectivity in [0, 1] and an optional fuzziness in [0, 1).
    
      VOLUME_SCATTERING : A participating medium with a complex refractive index, a scattering coefficient and a Henyey-Greenstein asymmetry parameter g.
    """
    DIFFUSE: typing.ClassVar[MaterialType]
    REFLECTIVE: typing.ClassVar[MaterialType]
    REFRACTIVE: typing.ClassVar[MaterialType]
    VOLUME_SCATTERING: typing.ClassVar[MaterialType]
    __members__: typing.ClassVar[dict[str, MaterialType]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OutputType:
    """
    Per-ray quantity stored in an output buffer.
    
    Assign a list of these to :attr:`Simulation.outputs` before calling
    :meth:`Simulation.run`, then read the results with
    :meth:`Simulation.get_output_buffer`. The dtype and shape of each buffer are
    given below, where *N* stands for the ray dimensions (see
    :meth:`Simulation.get_output_buffer`). Positions and lengths are in
    geometry units (:attr:`Simulation.length_unit`).
    
    
    Members:
    
      LAST_DIRECTION : float32 (N, 3): unit direction at the end of the ray: the direction it left the scene in, or the one it travelled in when it was absorbed.
    
      LAST_POSITION : float32 (N, 3): position of the last interaction: where the ray was absorbed, or where it last interacted before it left the scene. A ray without any interaction keeps its source position.
    
      RAY_STATE : int32 (N,): how the ray ended: 0 left the scene, 1 absorbed, 2 stopped at Simulation.max_scattering_count, 3 error.
    
      LAST_MEDIUM_ID : int32 (N,): index in Simulation.materials of the medium the ray was in at the end; 0 is the medium outside of all geometry.
    
      SCATTERING_COUNT : uint32 (N,): number of interactions the ray went on from (reflections, refractions and volume scatterings). The absorption or escape that ended the ray is not counted.
    
      NUMBER_OF_WARNINGS : uint32 (N,): number of inconsistencies in the tracking of the media the ray is in: overlapping geometry, media nested deeper than can be tracked, or leaving a medium the ray was not in. Non-zero values point to problems with the geometry.
    
      STOKES_VECTOR : float32 (N, 4): Stokes vector [I, Q, U, V] at the end, with Q referring to the scattering plane spanned by the source direction and the last direction. If the two are (anti-)parallel, the vector stays in the frame of the Q- axis the ray carried.
    
      STOKES_VECTOR_IN : float32 (N, 4): Stokes vector [I, Q, U, V] of the ray at the source, rotated into the same scattering plane as STOKES_VECTOR, so the two can be compared directly.
    
      OPTICAL_PATH_LENGTH : float64 (N,): optical path length of the whole ray: the sum of the distances times the real part of the refractive index of the medium they were travelled in.
    
      SOURCE_DIRECTION : float32 (N, 3): unit direction of the ray at the source.
    
      SOURCE_POSITION : float32 (N, 3): position of the ray at the source.
    
      SCATTERING_ANGLE : float32 (N,): angle between the source direction and the last direction, in radians in [0, pi].
    
      Q_MINUS_AXIS_IN : float32 (N, 3): unit Q- reference axis of the Stokes vector at the source, perpendicular to the source direction (see Simulation.q_minus_axis_seed).
    
      DIRECTION_HISTOGRAM_HEALPIX : uint32 (npix,): number of rays that left the scene in each direction, binned in HEALPix pixels in RING ordering, with npix = 12 * Simulation.direction_healpix_nside ** 2. A CameraRayGenerator gives one histogram per image pixel, (image_height, image_width, npix). Needs direction_healpix_nside > 0.
    
      FIRST_POSITION : float32 (N, 3): position of the first interaction, at a surface or in a volume; NaN for rays that left the scene without one.
    """
    DIRECTION_HISTOGRAM_HEALPIX: typing.ClassVar[OutputType]
    FIRST_POSITION: typing.ClassVar[OutputType]
    LAST_DIRECTION: typing.ClassVar[OutputType]
    LAST_MEDIUM_ID: typing.ClassVar[OutputType]
    LAST_POSITION: typing.ClassVar[OutputType]
    NUMBER_OF_WARNINGS: typing.ClassVar[OutputType]
    OPTICAL_PATH_LENGTH: typing.ClassVar[OutputType]
    Q_MINUS_AXIS_IN: typing.ClassVar[OutputType]
    RAY_STATE: typing.ClassVar[OutputType]
    SCATTERING_ANGLE: typing.ClassVar[OutputType]
    SCATTERING_COUNT: typing.ClassVar[OutputType]
    SOURCE_DIRECTION: typing.ClassVar[OutputType]
    SOURCE_POSITION: typing.ClassVar[OutputType]
    STOKES_VECTOR: typing.ClassVar[OutputType]
    STOKES_VECTOR_IN: typing.ClassVar[OutputType]
    __members__: typing.ClassVar[dict[str, OutputType]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RefractiveIndex:
    """
    Complex refractive index *n + ik*.
    
    The real part ``r`` is the refractive index *n*, the imaginary part ``i`` the
    extinction coefficient *k* (≥ 0), which controls absorption.
    
    Example::
    
        ri = RefractiveIndex(1.31, 0.0)            # non-absorbing ice
        ri = RefractiveIndex(complex_value=1.5+0.01j)
        ri.complex = 1.33+0j
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Create a refractive index of 0.
        """
    @typing.overload
    def __init__(self, r: typing.SupportsFloat | typing.SupportsIndex, i: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a refractive index r + i·j from its real and imaginary parts.
        """
    @typing.overload
    def __init__(self, complex_value: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a refractive index from a complex number, e.g. 1.31+0j.
        """
    def __repr__(self) -> str:
        ...
    @property
    def complex(self) -> complex:
        """
        The refractive index as a complex number n + ik.
        """
    @complex.setter
    def complex(self, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def i(self) -> float:
        """
        Imaginary part k of the refractive index, the extinction coefficient.
        """
    @i.setter
    def i(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def r(self) -> float:
        """
        Real part n of the refractive index.
        """
    @r.setter
    def r(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Diffuse:
    """
    Properties of a Lambertian (diffuse) surface.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def albedo(self) -> float:
        """
        Fraction of the light reflected diffusely, in [0, 1]; the rest is absorbed.
        """
    @albedo.setter
    def albedo(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Refractive:
    """
    Properties of a refractive medium: Fresnel reflection and refraction at its surface.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def refractive_index(self) -> RefractiveIndex:
        """
        Complex refractive index n + ik of the medium; k causes absorption inside it.
        """
    @refractive_index.setter
    def refractive_index(self, arg0: RefractiveIndex) -> None:
        ...
class Reflective:
    """
    Properties of a metallic (specular) surface.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def fuzziness(self) -> float:
        """
        Roughness in [0, 1): 0 is a perfect mirror, values towards 1 spread the reflection over nearly a hemisphere.
        """
    @fuzziness.setter
    def fuzziness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def reflectivity(self) -> float:
        """
        Fraction of the light reflected, in [0, 1]; the rest is absorbed.
        """
    @reflectivity.setter
    def reflectivity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class VolumeScattering:
    """
    Properties of a participating (volume scattering) medium.
    
    Rays in this medium are scattered with the Henyey–Greenstein phase function
    or absorbed, at distances drawn from the extinction coefficient: the
    scattering coefficient plus the absorption coefficient from the imaginary
    part of the refractive index.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def asymetry_parameter(self) -> float:
        """
        Henyey-Greenstein asymmetry parameter g in (-1, 1): 0 isotropic, > 0 forward, < 0 backward scattering.
        """
    @asymetry_parameter.setter
    def asymetry_parameter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def refractive_index(self) -> RefractiveIndex:
        """
        Complex refractive index n + ik of the medium; k gives the absorption coefficient.
        """
    @refractive_index.setter
    def refractive_index(self, arg0: RefractiveIndex) -> None:
        ...
    @property
    def scattering_coefficient(self) -> float:
        """
        Scattering coefficient per geometry unit (Simulation.length_unit).
        """
    @scattering_coefficient.setter
    def scattering_coefficient(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MaterialProperties:
    """
    The parameters of a material, one set per :class:`MaterialType`.
    
    Only the set of the :attr:`Material.type` of the owning :class:`Material` is
    meaningful; the four share the same memory.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def diffuse(self) -> Diffuse:
        """
        Properties of a DIFFUSE material.
        """
    @diffuse.setter
    def diffuse(self, arg1: Diffuse) -> None:
        ...
    @property
    def reflective(self) -> Reflective:
        """
        Properties of a REFLECTIVE material.
        """
    @reflective.setter
    def reflective(self, arg1: Reflective) -> None:
        ...
    @property
    def refractive(self) -> Refractive:
        """
        Properties of a REFRACTIVE material.
        """
    @refractive.setter
    def refractive(self, arg1: Refractive) -> None:
        ...
    @property
    def volume_scattering(self) -> VolumeScattering:
        """
        Properties of a VOLUME_SCATTERING material.
        """
    @volume_scattering.setter
    def volume_scattering(self, arg1: VolumeScattering) -> None:
        ...
class Material:
    """
    How a surface or volume interacts with rays.
    
    A material combines a :class:`MaterialType` with the parameters of that type.
    Use one of the constructors rather than filling :attr:`properties` by hand.
    Material ``0`` of :attr:`Simulation.materials` is the medium outside of all
    geometry.
    
    Example::
    
        from photontracer import Material, DIFFUSE, REFRACTIVE, VOLUME_SCATTERING
    
        m_diffuse  = Material(DIFFUSE, albedo=0.9)
        m_glass    = Material(REFRACTIVE, 1.5+0j)
        m_ice      = Material(REFRACTIVE, 1.31+1e-9j)
        m_cloud    = Material(VOLUME_SCATTERING, 1.33+0j,
                              scattering_coefficient=100.0, g=0.85)
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Create a material without parameters.
        """
    @typing.overload
    def __init__(self, type: MaterialType, refractive_index: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a REFRACTIVE material from a complex refractive index, e.g. 1.5+0j.
        """
    @typing.overload
    def __init__(self, type: MaterialType, refractive_index: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, scattering_coefficient: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a VOLUME_SCATTERING medium.
        
        Args:
            type: Must be ``VOLUME_SCATTERING``.
            refractive_index: Complex refractive index of the medium.
            scattering_coefficient: Scattering coefficient per geometry unit
                (:attr:`Simulation.length_unit`).
            g: Henyey–Greenstein asymmetry parameter in ``(-1, 1)``.
        """
    @typing.overload
    def __init__(self, type: MaterialType, albedo: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a DIFFUSE material with an albedo in [0, 1].
        """
    @typing.overload
    def __init__(self, type: MaterialType, reflectivity: typing.SupportsFloat | typing.SupportsIndex, fuzziness: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a REFLECTIVE material with a reflectivity in [0, 1] and a fuzziness in [0, 1).
        """
    @typing.overload
    def __init__(self, type: MaterialType, reflectivity: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a REFLECTIVE material that is a perfect mirror (fuzziness 0).
        """
    @typing.overload
    def __init__(self, type: MaterialType, refractive_index: RefractiveIndex) -> None:
        """
        Create a REFRACTIVE material from a RefractiveIndex.
        """
    def __repr__(self) -> str:
        ...
    @property
    def properties(self) -> MaterialProperties:
        """
        The parameters of the material; only those of its type are used.
        """
    @properties.setter
    def properties(self, arg0: MaterialProperties) -> None:
        ...
    @property
    def type(self) -> MaterialType:
        """
        How the material interacts with rays.
        """
    @type.setter
    def type(self, arg0: MaterialType) -> None:
        ...
class IGeometry:
    """
    Base class of all geometries.
    
    Use one of the subclasses: :class:`MeshGeometry` or :class:`InstanceGeometry`.
    """
    def free_device_memory(self) -> None:
        """
        Free the acceleration structures of this geometry and its sub-geometries. The next Simulation.run() builds them again.
        """
    def get_type(self) -> GeometryType:
        """
        Return the type of the geometry.
        """
class MeshGeometry(IGeometry):
    """
    A single closed or open triangle mesh.
    
    A mesh has no material IDs of its own: the medium inside it is material
    ``1`` and the medium outside material ``0``. Wrap meshes in an
    :class:`InstanceGeometry` to give them other materials.
    
    Example::
    
        import numpy as np
        from photontracer import MeshGeometry
    
        verts = np.array([[-1,-1,0],[1,-1,0],[0,1,0]], dtype=np.float32)
        faces = np.array([[0,1,2]], dtype=np.uint32)
        geo = MeshGeometry(verts, faces)
    """
    def __init__(self, vertices: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], indices: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32], compact: bool = True) -> None:
        """
        Create a mesh from its vertices and triangles.
        
        Args:
            vertices: Vertex positions of shape ``(V, 3)``, cast to ``float32``.
            indices: Triangles as vertex indices, of shape ``(T, 3)`` or flat with
                length ``3 * T``, cast to ``uint32``.
            compact: Compact the acceleration structure after building it, which
                saves memory at a small cost in build time.
        """
class InstanceGeometry(IGeometry):
    """
    Instances of one or more sub-geometries placed by transforms.
    
    Use this geometry type to model scenes with many copies of a small set of
    particle shapes (e.g. ice crystals of several habits at arbitrary
    orientations) efficiently. Instance geometries can be nested up to
    :attr:`Simulation.max_nested_geometry_levels`.
    
    Example::
    
        import numpy as np
        from photontracer import MeshGeometry, InstanceGeometry
    
        proto = MeshGeometry(vertices, faces)
    
        # Identity transform for two instances
        transforms = np.tile(np.eye(3, 4, dtype=np.float32), (2, 1, 1))
        type_ids   = np.array([0, 0], dtype=np.uint32)
        mat_ids    = np.array([1, 2], dtype=np.uint32)
    
        geo = InstanceGeometry([proto], transforms, type_ids, mat_ids)
    """
    def __init__(self, sub_geometries: collections.abc.Sequence[IGeometry], instance_transforms: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], particle_type_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32], material_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32]) -> None:
        """
        Create instances of sub-geometries.
        
        Args:
            sub_geometries: Prototype geometries, one per particle type.
            instance_transforms: Affine 3×4 row-major transforms of the instances,
                of shape ``(N, 3, 4)``, cast to ``float32``.
            particle_type_ids: Index in ``sub_geometries`` of each instance, of
                shape ``(N,)``.
            material_ids: Index in :attr:`Simulation.materials` of the medium inside
                each instance, of shape ``(N,)``.
        """
class IRayGenerator:
    """
    Base class of the ray generators.
    
    A ray generator determines the number, origin, and initial direction of the
    rays traced by the simulation. Use one of the subclasses:
    
    * :class:`ParallelRayGenerator` – collimated beam (plane wave).
    * :class:`IsotropicRayGenerator` – rays from all directions, from outside of a
      sphere towards the scene.
    * :class:`DiffuseRayGenerator` – diffuse illumination of a surface:
      cosine-weighted directions around an axis.
    * :class:`CameraRayGenerator` – pinhole camera for rendering.
    
    Assign one to :attr:`Simulation.ray_generator` before calling
    :meth:`Simulation.run`.
    """
class ParallelRayGenerator(IRayGenerator):
    """
    A collimated (parallel) beam of rays, equivalent to a plane wave.
    
    All rays start on a disk of radius ``offset_radius`` centred at ``origin``
    and travel in the same ``direction``. The launch shape is
    ``(number_of_rays,)``.
    
    By default the disk is perpendicular to the beam, so its footprint on a
    tilted surface is stretched by ``1 / cos(angle)``. With ``disk_normal`` the
    disk can lie in another plane, e.g. parallel to the surface that is
    illuminated: the rays then start in that plane and still travel along
    ``direction``, and the footprint on the surface is a circle of the same radius
    at every angle of incidence.
    
    Example::
    
        from photontracer import ParallelRayGenerator
        import numpy as np
    
        gen = ParallelRayGenerator(
            number_of_rays=100_000,
            origin=np.array([0, 0, -500], dtype=np.float32),
            direction=[0, 0, 1],
            offset_radius=50.0,
        )
    
        # beam at 60 degrees from the z axis, emitted from a disk parallel to the xy-plane
        tilted = ParallelRayGenerator(
            number_of_rays=100_000,
            origin=[0, 0, -500],
            direction=[np.sin(np.pi / 3), 0, np.cos(np.pi / 3)],
            offset_radius=50.0,
            disk_normal=[0, 0, 1],
        )
    """
    def __init__(self, number_of_rays: typing.SupportsInt | typing.SupportsIndex, origin: photontracer._typing.Vector3, direction: photontracer._typing.Vector3, offset_radius: typing.SupportsFloat | typing.SupportsIndex, disk_normal: photontracer._typing.Vector3 | None = None) -> None:
        """
        Create a parallel beam.
        
        Args:
            number_of_rays: Number of rays to launch.
            origin: Centre of the source disk.
            direction: Direction of the beam; normalised automatically.
            offset_radius: Radius of the source disk in geometry units; ``0`` starts
                all rays at ``origin``.
            disk_normal: Normal of the plane of the source disk (normalised
                automatically), or ``None`` for a disk perpendicular to
                ``direction``. Must not be (nearly) perpendicular to ``direction``.
        """
    @property
    def direction(self) -> tuple[float, float, float]:
        """
        Unit direction of the beam; normalised when set.
        """
    @direction.setter
    def direction(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def disk_normal(self) -> tuple[float, float, float]:
        """
        Unit normal of the plane of the source disk; equal to direction for a disk perpendicular to the beam. Setting None makes the disk perpendicular to the beam again.
        """
    @disk_normal.setter
    def disk_normal(self, arg1: photontracer._typing.Vector3 | None) -> None:
        ...
    @property
    def offset_radius(self) -> float:
        """
        Radius of the source disk in geometry units.
        """
    @offset_radius.setter
    def offset_radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def origin(self) -> tuple[float, float, float]:
        """
        Centre of the source disk.
        """
    @origin.setter
    def origin(self, arg1: photontracer._typing.Vector3) -> None:
        ...
class IsotropicRayGenerator(IRayGenerator):
    """
    A source emitting rays isotropically from outside of a sphere towards the scene.
    
    Rays start at random points on a sphere of radius ``source_radius`` around
    ``center`` and point towards the center. ``offset_radius`` shifts the ray
    origins tangentially, so that the rays do not all meet in the center. The
    launch shape is ``(number_of_rays,)``.
    
    Example::
    
        gen = IsotropicRayGenerator(
            number_of_rays=1_000_000,
            center=[0, 0, 0],
            source_radius=1.0,
            offset_radius=1.0,
        )
    """
    def __init__(self, number_of_rays: typing.SupportsInt | typing.SupportsIndex, center: photontracer._typing.Vector3, source_radius: typing.SupportsFloat | typing.SupportsIndex, offset_radius: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create an isotropic source.
        
        Args:
            number_of_rays: Number of rays to launch.
            center: Centre of the sphere.
            source_radius: Radius of the sphere the rays start on.
            offset_radius: Radius in geometry units of the disk, tangential to the
                sphere, in which the ray origins are shifted; ``0`` makes every ray
                pass through ``center``.
        """
    @property
    def center(self) -> tuple[float, float, float]:
        """
        Centre of the sphere the rays start on.
        """
    @center.setter
    def center(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def offset_radius(self) -> float:
        """
        Radius in geometry units of the tangential shift of the ray origins.
        """
    @offset_radius.setter
    def offset_radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def source_radius(self) -> float:
        """
        Radius of the sphere the rays start on.
        """
    @source_radius.setter
    def source_radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class DiffuseRayGenerator(IRayGenerator):
    """
    A diffuse (Lambertian) source illuminating a surface from one hemisphere.
    
    Ray origins are uniformly distributed on a disk of radius ``offset_radius``
    centred at ``origin`` and perpendicular to ``direction``. Ray directions are
    cosine-weighted around ``direction``, i.e. the density is proportional to the
    cosine of the angle to ``direction``, which is the flux of an isotropic
    radiance field onto a surface normal to ``direction``. All rays point into
    the hemisphere around ``direction``. The launch shape is
    ``(number_of_rays,)``.
    
    Example::
    
        gen = DiffuseRayGenerator(
            number_of_rays=1_000_000,
            origin=[0, 0, 100],
            direction=[0, 0, -1],
            offset_radius=50.0,
        )
    """
    def __init__(self, number_of_rays: typing.SupportsInt | typing.SupportsIndex, origin: photontracer._typing.Vector3, direction: photontracer._typing.Vector3, offset_radius: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Create a diffuse source.
        
        Args:
            number_of_rays: Number of rays to launch.
            origin: Centre of the source disk.
            direction: Axis of the hemisphere the rays are emitted into, e.g.
                ``[0, 0, -1]`` for light falling down onto a surface in the xy-plane;
                normalised automatically.
            offset_radius: Radius of the source disk in geometry units.
        """
    @property
    def direction(self) -> tuple[float, float, float]:
        """
        Unit axis of the hemisphere the rays are emitted into; normalised when set.
        """
    @direction.setter
    def direction(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def offset_radius(self) -> float:
        """
        Radius of the source disk in geometry units.
        """
    @offset_radius.setter
    def offset_radius(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def origin(self) -> tuple[float, float, float]:
        """
        Centre of the source disk.
        """
    @origin.setter
    def origin(self, arg1: photontracer._typing.Vector3) -> None:
        ...
class CameraRayGenerator(IRayGenerator):
    """
    A perspective pinhole camera that generates rays for rendering images.
    
    The launch shape is ``(samples_per_pixel, image_width, image_height)``, so
    the output buffers have the shape ``(image_height, image_width,
    samples_per_pixel, ...)``, without the samples dimension if
    ``samples_per_pixel`` is ``1``. Angles are in degrees.
    
    Example::
    
        cam = CameraRayGenerator()
        cam.image_width       = 800
        cam.image_height      = 600
        cam.samples_per_pixel = 16
        cam.vertical_fov      = 45.0
        cam.look_from         = [0, 0, -500]
        cam.look_at           = [0, 0, 0]
        cam.vertical_up       = [0, 1, 0]
        cam.focus_distance    = 500.0
    """
    def __init__(self) -> None:
        ...
    @property
    def aspect_ratio(self) -> float:
        """
        Aspect ratio width / height.
        """
    @aspect_ratio.setter
    def aspect_ratio(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def defocus_angle(self) -> float:
        """
        Opening angle of the cone of rays through each pixel in degrees, for depth of field; 0 disables it.
        """
    @defocus_angle.setter
    def defocus_angle(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def focus_distance(self) -> float:
        """
        Distance from the camera to the plane in focus.
        """
    @focus_distance.setter
    def focus_distance(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def image_height(self) -> int:
        """
        Image height in pixels.
        """
    @image_height.setter
    def image_height(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def image_width(self) -> int:
        """
        Image width in pixels.
        """
    @image_width.setter
    def image_width(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def look_at(self) -> tuple[float, float, float]:
        """
        Point the camera is aimed at.
        """
    @look_at.setter
    def look_at(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def look_from(self) -> tuple[float, float, float]:
        """
        Position of the camera.
        """
    @look_from.setter
    def look_from(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def samples_per_pixel(self) -> int:
        """
        Number of rays per pixel.
        """
    @samples_per_pixel.setter
    def samples_per_pixel(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def vertical_fov(self) -> float:
        """
        Vertical field of view in degrees.
        """
    @vertical_fov.setter
    def vertical_fov(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vertical_up(self) -> tuple[float, float, float]:
        """
        Up direction of the camera, which sets its roll.
        """
    @vertical_up.setter
    def vertical_up(self, arg1: photontracer._typing.Vector3) -> None:
        ...
class Simulation:
    """
    A ray tracing simulation.
    
    A ``Simulation`` holds a ray tracing backend and everything needed for a
    launch: the geometry, the materials, the ray generator and the outputs.
    
    Example::
    
        sim = Simulation()
    
        sim.geometry      = MeshGeometry(vertices, faces)
        sim.materials     = [Material(REFRACTIVE, 1+0j), Material(REFRACTIVE, 1.33+0j)]
        sim.wavelength_um = 0.532          # green light
        sim.length_unit   = MICRO_METER
        sim.ray_generator = ParallelRayGenerator(100_000, [0,0,-200], [0,0,1], 50)
        sim.outputs       = [OutputType.SCATTERING_COUNT, OutputType.LAST_DIRECTION]
        sim.run()
    
        counts = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
    """
    def __init__(self, gpu_id: typing.SupportsInt | typing.SupportsIndex = 0, optix_logging_level: typing.SupportsInt | typing.SupportsIndex = 1, enable_validation_mode: bool = False, backend: Backend | None = None, cpu_threads: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Create a simulation on one of the backends of this build.
        
        Args:
            gpu_id: Index of the CUDA device to use. OptiX only.
            optix_logging_level: Verbosity of the OptiX messages, ``0`` (silent) to
                ``4``. OptiX only.
            enable_validation_mode: Enable the OptiX validation mode with additional
                runtime checks. Much slower, for development only. OptiX only.
            backend: :class:`Backend` to trace with. By default the first of
                :func:`available_backends`, which is OptiX if the build has it.
            cpu_threads: Number of threads of the Embree backend; ``0`` uses all
                cores.
        """
    def calculate_volume_fraction(self, box_min: photontracer._typing.Vector3, box_max: photontracer._typing.Vector3, number_of_samples: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fraction of an axis-aligned box that is inside the geometry.
        
        ``number_of_samples`` random points are drawn uniformly in the box, and the
        fraction of them inside the geometry is returned.
        
        Args:
            box_min: Minimum corner of the box.
            box_max: Maximum corner of the box.
            number_of_samples: Number of random points.
        
        Returns:
            The estimated volume fraction in ``[0, 1]``.
        """
    def free_device_memory(self) -> None:
        """
        Release the device buffers held by the simulation.
        
        Frees the pipeline, the output buffers and the acceleration structures of
        the geometry. :meth:`run` can be called again afterwards and builds them anew.
        """
    def get_output_buffer(self, output_type: OutputType) -> numpy.ndarray:
        """
        Copy an output buffer of the last run to a new NumPy array.
        
        The dtype and the components of each buffer are listed in
        :class:`OutputType`. The leading dimensions are those of the launch, without
        dimensions of size 1: ``(number_of_rays,)`` for the parallel, isotropic and
        diffuse ray generators, ``(image_height, image_width, samples_per_pixel)``
        for a :class:`CameraRayGenerator`. A vector quantity adds a last dimension of
        3 or 4 components.
        
        Args:
            output_type: The buffer to copy. It must have been in :attr:`outputs`
                for the last :meth:`run`.
        
        Raises:
            RuntimeError: If :meth:`run` has not been called yet, or the output
                type was not enabled.
        """
    def get_trace(self) -> numpy.ndarray:
        """
        Return the steps of the traced rays of the last run, ordered by ray and step.
        
        A structured array with the fields ``ray``, ``step``, ``event`` (a
        :class:`TraceEvent`), ``medium_in``, ``medium_out``, ``origin_in``,
        ``direction_in``, ``origin_out``, ``direction_out`` (each ``float32[3]``),
        ``stokes`` (``float32[4]``) and ``optical_path_length``. A step goes from a
        start to the next interaction: ``origin_in`` and ``direction_in`` are the ray
        before it, ``origin_out`` (just off the surface that was hit) and
        ``direction_out`` after it. The Stokes vector and the optical path length are
        those after the step. A ray with ``SCATTERING_COUNT`` scatterings has that
        many steps plus the last one.
        
        Raises:
            RuntimeError: If no rays were traced in the last run.
        """
    def run(self) -> None:
        """
        Trace the rays.
        
        Builds what changed since the last run (acceleration structures, pipeline,
        output buffers), launches the rays and waits for them. The results are then
        available from :meth:`get_output_buffer` and :meth:`get_trace`.
        
        :attr:`geometry`, :attr:`materials`, :attr:`wavelength_um`,
        :attr:`ray_generator` and :attr:`outputs` must be set before.
        """
    @property
    def direction_healpix_nside(self) -> int:
        """
        HEALPix ``NSIDE`` of the histogram of escape directions.
        
        When > ``0`` (a power of 2), the directions of the rays that leave the scene
        are counted in ``12 * NSIDE ** 2`` HEALPix pixels, available as
        ``OutputType.DIRECTION_HISTOGRAM_HEALPIX``. ``0`` (default) disables the
        histogram.
        
        Raises:
            ValueError: If the value is not ``0`` or a power of 2.
        """
    @direction_healpix_nside.setter
    def direction_healpix_nside(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def geometry(self) -> IGeometry:
        """
        The root geometry of the scene.
        
        Assign a :class:`MeshGeometry` or :class:`InstanceGeometry` before calling
        :meth:`run`.
        """
    @geometry.setter
    def geometry(self, arg1: IGeometry) -> None:
        ...
    @property
    def length_unit(self) -> LengthUnit:
        """
        Unit of the geometry coordinates, default MICRO_METER.
        """
    @length_unit.setter
    def length_unit(self, arg1: LengthUnit) -> None:
        ...
    @property
    def materials(self) -> list[Material]:
        """
        The materials of the scene, at most 16.
        
        Geometries refer to materials by their index in this list; material ``0`` is
        the medium outside of all geometry. A :class:`MeshGeometry` uses material
        ``1`` for its inside, so it needs at least two materials.
        """
    @materials.setter
    def materials(self, arg1: collections.abc.Sequence[Material]) -> None:
        ...
    @property
    def max_mesh_triangles(self) -> int:
        """
        Maximum number of triangles of a MeshGeometry on this backend.
        """
    @property
    def max_nested_geometry_levels(self) -> int:
        """
        Maximum depth of nested :class:`InstanceGeometry` hierarchies, default ``5``.
        
        Must be between ``1`` and the limit of the backend. Increase it for
        geometries nested more deeply.
        
        Raises:
            ValueError: If the value is outside of that range.
        """
    @max_nested_geometry_levels.setter
    def max_nested_geometry_levels(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_scattering_count(self) -> int:
        """
        Maximum number of interactions per ray.
        
        A ray that reaches it is stopped with ``RAY_STATE`` 2. ``0`` means no limit;
        the default is ``4294967295``.
        """
    @max_scattering_count.setter
    def max_scattering_count(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_sub_geometries(self) -> int:
        """
        Maximum number of sub-geometries of an InstanceGeometry on this backend.
        """
    @property
    def max_trace_steps(self) -> int:
        """
        Steps recorded per traced ray (default 1000); later steps are not stored.
        """
    @max_trace_steps.setter
    def max_trace_steps(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def outputs(self) -> list[OutputType]:
        """
        The output buffers to fill in the next run.
        
        Only the listed :class:`OutputType` buffers are allocated and filled.
        
        Example::
        
            sim.outputs = [
                OutputType.SCATTERING_COUNT,
                OutputType.LAST_DIRECTION,
                OutputType.STOKES_VECTOR,
            ]
        """
    @outputs.setter
    def outputs(self, arg1: collections.abc.Sequence[OutputType]) -> None:
        ...
    @property
    def q_minus_axis_seed(self) -> tuple[float, float, float]:
        """
        Direction from which the Q− axis of the source Stokes vector is made.
        
        The Q− axis of each source ray is this direction projected onto the plane
        perpendicular to the ray. With the default ``(nan, nan, nan)``, or if the
        direction is parallel to the ray, the x axis is used instead (the y axis for
        rays along x). The axis of each ray is available as
        ``OutputType.Q_MINUS_AXIS_IN``.
        """
    @q_minus_axis_seed.setter
    def q_minus_axis_seed(self, arg1: photontracer._typing.Vector3) -> None:
        ...
    @property
    def random_number_generator(self) -> RandomNumberGenerator:
        """
        Random number generator of the ray tracing kernels.
        
        ``PCG32`` (default) or ``MRG32K3A``, the curand generator of the 1.0
        releases. Changing it rebuilds the pipeline on the next :meth:`run`.
        
        Raises:
            ValueError: If the generator is not available on the backend.
        """
    @random_number_generator.setter
    def random_number_generator(self, arg1: RandomNumberGenerator) -> None:
        ...
    @property
    def ray_generator(self) -> IRayGenerator:
        """
        The source of the rays: a ParallelRayGenerator, IsotropicRayGenerator, DiffuseRayGenerator or CameraRayGenerator.
        """
    @ray_generator.setter
    def ray_generator(self, arg1: IRayGenerator) -> None:
        ...
    @property
    def seed(self) -> int:
        """
        Seed of the random number generator, default ``42``.
        
        The same seed gives the same rays when run again with the same build and
        :attr:`random_number_generator`; other seeds give statistically independent
        runs.
        """
    @seed.setter
    def seed(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def stokes_vector(self) -> tuple[float, float, float, float]:
        """
        Stokes vector ``[I, Q, U, V]`` of the source rays, default ``(1, 0, 0, 0)``.
        
        ``I`` must be ``1``, and ``Q² + U² + V² ≤ 1``. ``nan`` for ``Q`` or ``U``
        gives every ray a random linear polarization, ``nan`` for ``V`` a random
        circular one. ``(1, 0, 0, 0)`` is unpolarized light, ``(1, 1, 0, 0)`` light
        polarized along the Q+ axis.
        
        Raises:
            ValueError: If the vector is not a valid Stokes vector.
        """
    @stokes_vector.setter
    def stokes_vector(self, arg1: photontracer._typing.Vector4) -> None:
        ...
    @property
    def trace_rays(self) -> list[int]:
        """
        Indices of the rays whose steps are recorded (at most 256).
        
        An index is the position of the ray in the flattened output buffers. Seeds
        are deterministic, so a ray found in a normal run can be traced in the next
        one. Sorted and without duplicates when read. See :meth:`get_trace`.
        """
    @trace_rays.setter
    def trace_rays(self, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    @property
    def use_complex_fresnel(self) -> bool:
        """
        Use the complex Fresnel equations for refractive materials, default ``True``.
        
        The complex coefficients model absorption at interfaces correctly (e.g. the
        Fresnel peak in water ice) at a small cost. ``False`` uses the real-valued
        equations, mainly for testing.
        """
    @use_complex_fresnel.setter
    def use_complex_fresnel(self, arg1: bool) -> None:
        ...
    @property
    def wavelength_um(self) -> float:
        """
        Wavelength in micrometres (μm); must be set before :meth:`run`.
        
        Together with :attr:`length_unit` it converts the imaginary part of the
        refractive indices into absorption coefficients.
        """
    @wavelength_um.setter
    def wavelength_um(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def available_backends() -> list[Backend]:
    """
    The raytracing backends of this build, the default one first.
    """
def is_cuda_available() -> bool:
    """
    Whether at least one CUDA device is available.
    """
def is_verbose() -> bool:
    """
    Whether progress messages are printed.
    """
def set_verbose(verbose: bool) -> None:
    """
    Print progress messages (device in use, compile and launch times). Off by default.
    """
DIFFUSE: MaterialType
DIRECTION_HISTOGRAM_HEALPIX: OutputType
EMBREE: Backend
FIRST_POSITION: OutputType
LAST_DIRECTION: OutputType
LAST_MEDIUM_ID: OutputType
LAST_POSITION: OutputType
MESH: GeometryType
MESH_INSTANCED: GeometryType
METER: LengthUnit
MICRO_METER: LengthUnit
MILLI_METER: LengthUnit
MRG32K3A: RandomNumberGenerator
NUMBER_OF_WARNINGS: OutputType
OPTICAL_PATH_LENGTH: OutputType
OPTIX: Backend
PCG32: RandomNumberGenerator
Q_MINUS_AXIS_IN: OutputType
RAY_STATE: OutputType
REFLECTIVE: MaterialType
REFRACTIVE: MaterialType
SCATTERING_ANGLE: OutputType
SCATTERING_COUNT: OutputType
SOURCE_DIRECTION: OutputType
SOURCE_POSITION: OutputType
STOKES_VECTOR: OutputType
STOKES_VECTOR_IN: OutputType
VOLUME_SCATTERING: MaterialType
