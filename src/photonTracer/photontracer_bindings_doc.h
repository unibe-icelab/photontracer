// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

// The longer docstrings of the Python bindings. They are the documentation of the Python API:
// help() shows them, and photontracer_bindings.pyi is generated from them (scripts/generate_stubs.py).

namespace doc
{

constexpr const char *module = R"doc(Python bindings for the PhotonTracer ray tracing simulation.

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
)doc";

constexpr const char *LengthUnit = R"doc(Physical unit of the geometry coordinates.

Assign one of these to :attr:`Simulation.length_unit`. Positions, distances
and the optical path length are in this unit; it is needed to convert the
imaginary part of a refractive index, given for :attr:`Simulation.wavelength_um`
in μm, into an absorption coefficient per geometry unit.
)doc";

constexpr const char *MaterialType = R"doc(How a surface or volume interacts with rays.

Pass one of these as the first argument to :class:`Material`.
)doc";

constexpr const char *RandomNumberGenerator = R"doc(Random number generator of the ray tracing kernels.

Assign one of these to :attr:`Simulation.random_number_generator`. Both give
the same results statistically, but not ray by ray.
)doc";

constexpr const char *OutputType = R"doc(Per-ray quantity stored in an output buffer.

Assign a list of these to :attr:`Simulation.outputs` before calling
:meth:`Simulation.run`, then read the results with
:meth:`Simulation.get_output_buffer`. The dtype and shape of each buffer are
given below, where *N* stands for the ray dimensions (see
:meth:`Simulation.get_output_buffer`). Positions and lengths are in
geometry units (:attr:`Simulation.length_unit`).
)doc";

constexpr const char *OutputType_LAST_DIRECTION =
    "float32 (N, 3): unit direction at the end of the ray: the direction it left the scene in, or the one it "
    "travelled in when it was absorbed.";
constexpr const char *OutputType_LAST_POSITION =
    "float32 (N, 3): position of the last interaction: where the ray was absorbed, or where it last interacted "
    "before it left the scene. A ray without any interaction keeps its source position.";
constexpr const char *OutputType_RAY_STATE =
    "int32 (N,): how the ray ended: 0 left the scene, 1 absorbed, 2 stopped at Simulation.max_scattering_count, "
    "3 error.";
constexpr const char *OutputType_LAST_MEDIUM_ID =
    "int32 (N,): index in Simulation.materials of the medium the ray was in at the end; 0 is the medium outside "
    "of all geometry.";
constexpr const char *OutputType_SCATTERING_COUNT =
    "uint32 (N,): number of interactions the ray went on from (reflections, refractions and volume "
    "scatterings). The absorption or escape that ended the ray is not counted.";
constexpr const char *OutputType_NUMBER_OF_WARNINGS =
    "uint32 (N,): number of inconsistencies in the tracking of the media the ray is in: overlapping geometry, "
    "media nested deeper than can be tracked, or leaving a medium the ray was not in. Non-zero values point to "
    "problems with the geometry.";
constexpr const char *OutputType_STOKES_VECTOR =
    "float32 (N, 4): Stokes vector [I, Q, U, V] at the end, with Q referring to the scattering plane spanned "
    "by the source direction and the last direction. If the two are (anti-)parallel, the vector stays in the "
    "frame of the Q- axis the ray carried.";
constexpr const char *OutputType_STOKES_VECTOR_IN =
    "float32 (N, 4): Stokes vector [I, Q, U, V] of the ray at the source, rotated into the same scattering "
    "plane as STOKES_VECTOR, so the two can be compared directly.";
constexpr const char *OutputType_OPTICAL_PATH_LENGTH =
    "float64 (N,): optical path length of the whole ray: the sum of the distances times the real part of the "
    "refractive index of the medium they were travelled in.";
constexpr const char *OutputType_SOURCE_DIRECTION = "float32 (N, 3): unit direction of the ray at the source.";
constexpr const char *OutputType_SOURCE_POSITION = "float32 (N, 3): position of the ray at the source.";
constexpr const char *OutputType_SCATTERING_ANGLE =
    "float32 (N,): angle between the source direction and the last direction, in radians in [0, pi].";
constexpr const char *OutputType_Q_MINUS_AXIS_IN =
    "float32 (N, 3): unit Q- reference axis of the Stokes vector at the source, perpendicular to the source "
    "direction (see Simulation.q_minus_axis_seed).";
constexpr const char *OutputType_DIRECTION_HISTOGRAM_HEALPIX =
    "uint32 (npix,): number of rays that left the scene in each direction, binned in HEALPix pixels in RING "
    "ordering, with npix = 12 * Simulation.direction_healpix_nside ** 2. A CameraRayGenerator gives one "
    "histogram per image pixel, (image_height, image_width, npix). Needs direction_healpix_nside > 0.";
constexpr const char *OutputType_FIRST_POSITION =
    "float32 (N, 3): position of the first interaction, at a surface or in a volume; NaN for rays that left "
    "the scene without one.";

constexpr const char *MeshGeometry = R"doc(A single closed or open triangle mesh.

A mesh has no material IDs of its own: the medium inside it is material
``1`` and the medium outside material ``0``. Wrap meshes in an
:class:`InstanceGeometry` to give them other materials.

Example::

    import numpy as np
    from photontracer import MeshGeometry

    verts = np.array([[-1,-1,0],[1,-1,0],[0,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2]], dtype=np.uint32)
    geo = MeshGeometry(verts, faces)
)doc";

constexpr const char *MeshGeometry_init = R"doc(Create a mesh from its vertices and triangles.

Args:
    vertices: Vertex positions of shape ``(V, 3)``, cast to ``float32``.
    indices: Triangles as vertex indices, of shape ``(T, 3)`` or flat with
        length ``3 * T``, cast to ``uint32``.
    compact: Compact the acceleration structure after building it, which
        saves memory at a small cost in build time.
)doc";

constexpr const char *InstanceGeometry = R"doc(Instances of one or more sub-geometries placed by transforms.

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
)doc";

constexpr const char *InstanceGeometry_init = R"doc(Create instances of sub-geometries.

Args:
    sub_geometries: Prototype geometries, one per particle type.
    instance_transforms: Affine 3×4 row-major transforms of the instances,
        of shape ``(N, 3, 4)``, cast to ``float32``.
    particle_type_ids: Index in ``sub_geometries`` of each instance, of
        shape ``(N,)``.
    material_ids: Index in :attr:`Simulation.materials` of the medium inside
        each instance, of shape ``(N,)``.
)doc";

constexpr const char *RefractiveIndex = R"doc(Complex refractive index *n + ik*.

The real part ``r`` is the refractive index *n*, the imaginary part ``i`` the
extinction coefficient *k* (≥ 0), which controls absorption.

Example::

    ri = RefractiveIndex(1.31, 0.0)            # non-absorbing ice
    ri = RefractiveIndex(complex_value=1.5+0.01j)
    ri.complex = 1.33+0j
)doc";

constexpr const char *VolumeScattering = R"doc(Properties of a participating (volume scattering) medium.

Rays in this medium are scattered with the Henyey–Greenstein phase function
or absorbed, at distances drawn from the extinction coefficient: the
scattering coefficient plus the absorption coefficient from the imaginary
part of the refractive index.
)doc";

constexpr const char *MaterialProperties = R"doc(The parameters of a material, one set per :class:`MaterialType`.

Only the set of the :attr:`Material.type` of the owning :class:`Material` is
meaningful; the four share the same memory.
)doc";

constexpr const char *Material = R"doc(How a surface or volume interacts with rays.

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
)doc";

constexpr const char *Material_init_volume_scattering = R"doc(Create a VOLUME_SCATTERING medium.

Args:
    type: Must be ``VOLUME_SCATTERING``.
    refractive_index: Complex refractive index of the medium.
    scattering_coefficient: Scattering coefficient per geometry unit
        (:attr:`Simulation.length_unit`).
    g: Henyey–Greenstein asymmetry parameter in ``(-1, 1)``.
)doc";

constexpr const char *IGeometry = R"doc(Base class of all geometries.

Use one of the subclasses: :class:`MeshGeometry` or :class:`InstanceGeometry`.
)doc";

constexpr const char *IRayGenerator = R"doc(Base class of the ray generators.

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
)doc";

constexpr const char *ParallelRayGenerator = R"doc(A collimated (parallel) beam of rays, equivalent to a plane wave.

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
)doc";

constexpr const char *ParallelRayGenerator_init = R"doc(Create a parallel beam.

Args:
    number_of_rays: Number of rays to launch.
    origin: Centre of the source disk.
    direction: Direction of the beam; normalised automatically.
    offset_radius: Radius of the source disk in geometry units; ``0`` starts
        all rays at ``origin``.
    disk_normal: Normal of the plane of the source disk (normalised
        automatically), or ``None`` for a disk perpendicular to
        ``direction``. Must not be (nearly) perpendicular to ``direction``.
)doc";

constexpr const char *IsotropicRayGenerator = R"doc(A source emitting rays isotropically from outside of a sphere towards the scene.

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
)doc";

constexpr const char *IsotropicRayGenerator_init = R"doc(Create an isotropic source.

Args:
    number_of_rays: Number of rays to launch.
    center: Centre of the sphere.
    source_radius: Radius of the sphere the rays start on.
    offset_radius: Radius in geometry units of the disk, tangential to the
        sphere, in which the ray origins are shifted; ``0`` makes every ray
        pass through ``center``.
)doc";

constexpr const char *DiffuseRayGenerator = R"doc(A diffuse (Lambertian) source illuminating a surface from one hemisphere.

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
)doc";

constexpr const char *DiffuseRayGenerator_init = R"doc(Create a diffuse source.

Args:
    number_of_rays: Number of rays to launch.
    origin: Centre of the source disk.
    direction: Axis of the hemisphere the rays are emitted into, e.g.
        ``[0, 0, -1]`` for light falling down onto a surface in the xy-plane;
        normalised automatically.
    offset_radius: Radius of the source disk in geometry units.
)doc";

constexpr const char *CameraRayGenerator = R"doc(A perspective pinhole camera that generates rays for rendering images.

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
)doc";

constexpr const char *Simulation = R"doc(A ray tracing simulation.

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
)doc";

constexpr const char *Simulation_init = R"doc(Create a simulation on one of the backends of this build.

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
)doc";

constexpr const char *Simulation_run = R"doc(Trace the rays.

Builds what changed since the last run (acceleration structures, pipeline,
output buffers), launches the rays and waits for them. The results are then
available from :meth:`get_output_buffer` and :meth:`get_trace`.

:attr:`geometry`, :attr:`materials`, :attr:`wavelength_um`,
:attr:`ray_generator` and :attr:`outputs` must be set before.
)doc";

constexpr const char *Simulation_free_device_memory = R"doc(Release the device buffers held by the simulation.

Frees the pipeline, the output buffers and the acceleration structures of
the geometry. :meth:`run` can be called again afterwards and builds them anew.
)doc";

constexpr const char *Simulation_calculate_volume_fraction = R"doc(Estimate the fraction of an axis-aligned box that is inside the geometry.

``number_of_samples`` random points are drawn uniformly in the box, and the
fraction of them inside the geometry is returned.

Args:
    box_min: Minimum corner of the box.
    box_max: Maximum corner of the box.
    number_of_samples: Number of random points.

Returns:
    The estimated volume fraction in ``[0, 1]``.
)doc";

constexpr const char *Simulation_get_output_buffer = R"doc(Copy an output buffer of the last run to a new NumPy array.

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
)doc";

constexpr const char *Simulation_trace_rays = R"doc(Indices of the rays whose steps are recorded (at most 256).

An index is the position of the ray in the flattened output buffers. Seeds
are deterministic, so a ray found in a normal run can be traced in the next
one. Sorted and without duplicates when read. See :meth:`get_trace`.
)doc";

constexpr const char *Simulation_get_trace = R"doc(Return the steps of the traced rays of the last run, ordered by ray and step.

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
)doc";

constexpr const char *Simulation_geometry = R"doc(The root geometry of the scene.

Assign a :class:`MeshGeometry` or :class:`InstanceGeometry` before calling
:meth:`run`.
)doc";

constexpr const char *Simulation_materials = R"doc(The materials of the scene, at most 16.

Geometries refer to materials by their index in this list; material ``0`` is
the medium outside of all geometry. A :class:`MeshGeometry` uses material
``1`` for its inside, so it needs at least two materials.
)doc";

constexpr const char *Simulation_wavelength_um = R"doc(Wavelength in micrometres (μm); must be set before :meth:`run`.

Together with :attr:`length_unit` it converts the imaginary part of the
refractive indices into absorption coefficients.
)doc";

constexpr const char *Simulation_stokes_vector = R"doc(Stokes vector ``[I, Q, U, V]`` of the source rays, default ``(1, 0, 0, 0)``.

``I`` must be ``1``, and ``Q² + U² + V² ≤ 1``. ``nan`` for ``Q`` or ``U``
gives every ray a random linear polarization, ``nan`` for ``V`` a random
circular one. ``(1, 0, 0, 0)`` is unpolarized light, ``(1, 1, 0, 0)`` light
polarized along the Q+ axis.
)doc";

constexpr const char *Simulation_q_minus_axis_seed = R"doc(Direction from which the Q− axis of the source Stokes vector is made.

The Q− axis of each source ray is this direction projected onto the plane
perpendicular to the ray. With the default ``(nan, nan, nan)``, or if the
direction is parallel to the ray, the x axis is used instead (the y axis for
rays along x). The axis of each ray is available as
``OutputType.Q_MINUS_AXIS_IN``.
)doc";

constexpr const char *Simulation_max_scattering_count = R"doc(Maximum number of interactions per ray.

A ray that reaches it is stopped with ``RAY_STATE`` 2. ``0`` means no limit;
the default is ``4294967295``.
)doc";

constexpr const char *Simulation_max_nested_geometry_levels = R"doc(Maximum depth of nested :class:`InstanceGeometry` hierarchies, default ``5``.

Must be between ``1`` and the limit of the backend. Increase it for
geometries nested more deeply.
)doc";

constexpr const char *Simulation_seed = R"doc(Seed of the random number generator, default ``42``.

The same seed gives the same rays when run again with the same build and
:attr:`random_number_generator`; other seeds give statistically independent
runs.
)doc";

constexpr const char *Simulation_random_number_generator = R"doc(Random number generator of the ray tracing kernels.

``PCG32`` (default) or ``MRG32K3A``, the curand generator of the 1.0
releases. Changing it rebuilds the pipeline on the next :meth:`run`.

Raises:
    ValueError: If the generator is not available on the backend.
)doc";

constexpr const char *Simulation_use_complex_fresnel = R"doc(Use the complex Fresnel equations for refractive materials, default ``True``.

The complex coefficients model absorption at interfaces correctly (e.g. the
Fresnel peak in water ice) at a small cost. ``False`` uses the real-valued
equations, mainly for testing.
)doc";

constexpr const char *Simulation_direction_healpix_nside = R"doc(HEALPix ``NSIDE`` of the histogram of escape directions.

When > ``0`` (a power of 2), the directions of the rays that leave the scene
are counted in ``12 * NSIDE ** 2`` HEALPix pixels, available as
``OutputType.DIRECTION_HISTOGRAM_HEALPIX``. ``0`` (default) disables the
histogram.
)doc";

constexpr const char *Simulation_outputs = R"doc(The output buffers to fill in the next run.

Only the listed :class:`OutputType` buffers are allocated and filled.

Example::

    sim.outputs = [
        OutputType.SCATTERING_COUNT,
        OutputType.LAST_DIRECTION,
        OutputType.STOKES_VECTOR,
    ]
)doc";

} // namespace doc
