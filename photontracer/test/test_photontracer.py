import numpy as np
import trimesh
import pytest
from photontracer import available_backends, Backend, LengthUnit, Material, MaterialType, MeshGeometry, InstanceGeometry, Simulation, ParallelRayGenerator, IsotropicRayGenerator, OutputType, RandomNumberGenerator

def test_reflection_angle(backend):
    sim = Simulation(backend=backend)
    angle_deg = 45
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    origin = -k
    ray_generator = ParallelRayGenerator(
        number_of_rays=1000,
        origin=origin, direction=k, 
        offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))

    sim.wavelength_um = 1
    sim.stokes_vector = [1, 0, 0, 0]
    sim.geometry = MeshGeometry(slab.vertices, slab.faces)
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (count == 1) & (k_out[:, 2] > 0)
    k_out_up = k_out[escape_up]
    stokes_up = stokes[escape_up]

    # check if all reflected rays have the same angle with the normal
    normal = np.array([0, 0, 1], dtype=float)

    k_out_should = k - 2 * np.dot(k, normal) * normal
    k_out_should /= np.linalg.norm(k_out_should)

    assert np.allclose(
        k_out_up, k_out_should), "Reflected directions do not match expected reflection"
    
    print(stokes_up.mean(axis=0))
    assert stokes_up.mean(axis=0)[1] < 0, "Stokes vector Q should be positive on average"


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("subdivisions", [0, 6])
def test_dense_mesh_reflection(backend, subdivisions, compact):
    # A finely subdivided slab must behave like the 12-triangle one, with or without compaction.
    sim = Simulation(backend=backend)
    k = np.array([-np.sin(np.pi / 4), 0, -np.cos(np.pi / 4)])
    slab = trimesh.creation.box(extents=[10, 10, 1],
                                transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    for _ in range(subdivisions):
        slab = slab.subdivide()

    sim.wavelength_um = 1
    sim.stokes_vector = [1, 0, 0, 0]
    sim.geometry = MeshGeometry(slab.vertices, slab.faces, compact=compact)
    sim.materials = [Material(MaterialType.REFRACTIVE, 1+0j), Material(MaterialType.REFRACTIVE, 1.5+0j)]
    sim.ray_generator = ParallelRayGenerator(number_of_rays=10000, origin=-k, direction=k, offset_radius=3)
    sim.outputs = [OutputType.SCATTERING_COUNT, OutputType.LAST_DIRECTION]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
    reflected = (count == 1) & (k_out[:, 2] > 0)
    assert reflected.any()
    assert np.allclose(k_out[reflected], [k[0], 0, -k[2]], atol=1e-5)


def _box_simulation(backend, materials, geometry=None):
    sim = Simulation(backend=backend)
    box = trimesh.creation.box(extents=[2, 2, 2])
    sim.geometry = geometry or MeshGeometry(box.vertices, box.faces)
    sim.wavelength_um = 1.0
    sim.materials = materials
    sim.ray_generator = ParallelRayGenerator(number_of_rays=100, origin=[0, 0, 5], direction=[0, 0, -1], offset_radius=0.5)
    sim.outputs = [OutputType.SCATTERING_COUNT]
    return sim


def _materials(count):
    """Vacuum followed by glass materials."""
    return [Material(MaterialType.REFRACTIVE, 1 + 0j)] + [Material(MaterialType.REFRACTIVE, 1.5 + 0j)] * (count - 1)


def _instances(material_id):
    sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
    return InstanceGeometry([MeshGeometry(sphere.vertices, sphere.faces)],
                            np.eye(4, dtype=np.float32)[None, :3, :], [0], [material_id])


def test_too_many_materials_are_rejected(backend):
    sim = Simulation(backend=backend)
    with pytest.raises(ValueError, match="materials"):
        sim.materials = _materials(17)


def test_mesh_needs_two_materials(backend):
    with pytest.raises(ValueError, match="material"):
        _box_simulation(backend, _materials(1)).run()


@pytest.mark.parametrize("material_id, count", [(2, 2), (5, 2), (16, 16)])
def test_instance_material_id_out_of_range_is_rejected(backend, material_id, count):
    with pytest.raises(ValueError, match="material"):
        _box_simulation(backend, _materials(count), _instances(material_id)).run()


def test_last_material_id_is_supported(backend):
    sim = _box_simulation(backend, _materials(16), _instances(15))
    sim.run()
    assert sim.get_output_buffer(OutputType.SCATTERING_COUNT)[0] == 2


def test_geometry_shared_between_simulations(backend):
    box = trimesh.creation.box(extents=[2, 2, 2])
    geometry = MeshGeometry(box.vertices, box.faces)
    a = _box_simulation(backend, _materials(2), geometry)
    b = _box_simulation(backend, _materials(2), geometry)

    counts = []
    for sim in (a, b, a, b):
        sim.run()
        counts.append(sim.get_output_buffer(OutputType.SCATTERING_COUNT).copy())
    del b
    a.run()
    counts.append(a.get_output_buffer(OutputType.SCATTERING_COUNT).copy())

    for result in counts[1:]:
        assert (result == counts[0]).all()


def test_simulation_survives_destruction_of_another(backend):
    first = _box_simulation(backend, _materials(2))
    first.run()
    second = _box_simulation(backend, _materials(2))
    second.run()
    del second
    before = first.get_output_buffer(OutputType.SCATTERING_COUNT).copy()
    first.run()
    assert (first.get_output_buffer(OutputType.SCATTERING_COUNT) == before).all()


# Statistical checks against exact results. They only pass if the random numbers are uniform
# and uncorrelated. Five standard errors keep them stable.

GENERATORS = [RandomNumberGenerator.PCG32, RandomNumberGenerator.MRG32K3A]


def _slab_beam_simulation(backend, material, rays, generator, seed=42):
    """A beam falling straight onto a large slab whose top surface is z = 0."""
    sim = Simulation(backend=backend)
    slab = trimesh.creation.box(extents=[10, 10, 1],
                                transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    sim.geometry = MeshGeometry(slab.vertices, slab.faces)
    sim.wavelength_um = 1.0
    sim.materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), material]
    sim.seed = seed
    _set_generator(sim, generator)
    sim.ray_generator = ParallelRayGenerator(number_of_rays=rays, origin=[0, 0, 5], direction=[0, 0, -1], offset_radius=3)
    sim.outputs = [OutputType.SCATTERING_COUNT, OutputType.LAST_DIRECTION, OutputType.RAY_STATE]
    sim.run()
    return (sim.get_output_buffer(OutputType.SCATTERING_COUNT), sim.get_output_buffer(OutputType.LAST_DIRECTION),
            sim.get_output_buffer(OutputType.RAY_STATE))


def _set_generator(sim, generator):
    try:
        sim.random_number_generator = generator
    except ValueError:
        pytest.skip(f"{generator} is not available on this backend")


def _assert_fraction(count, total, expected, what):
    sigma = np.sqrt(expected * (1 - expected) / total)
    assert abs(count / total - expected) < 5 * sigma, f"{what}: {count / total:.5f}, expected {expected:.5f} +- {sigma:.5f}"


def test_pcg32_is_the_default_random_number_generator(backend):
    assert Simulation(backend=backend).random_number_generator == RandomNumberGenerator.PCG32


@pytest.mark.parametrize("generator", GENERATORS)
def test_normal_incidence_reflectance_matches_fresnel(backend, generator):
    n = 400000
    counts, directions, _ = _slab_beam_simulation(backend, Material(MaterialType.REFRACTIVE, 1.5 + 0j), n, generator)
    reflected_at_top = (counts == 1) & (directions[:, 2] > 0)
    _assert_fraction(int(reflected_at_top.sum()), n, ((1.5 - 1) / (1.5 + 1)) ** 2, "reflectance")


@pytest.mark.parametrize("generator", GENERATORS)
def test_diffuse_surface_absorbs_one_minus_albedo(backend, generator):
    n = 400000
    counts, directions, state = _slab_beam_simulation(backend, Material(MaterialType.DIFFUSE, 0.3), n, generator)
    _assert_fraction(int((state == 1).sum()), n, 0.7, "absorbed fraction")

    # a Lambertian surface sends light out with a mean cosine of 2/3
    scattered = state == 0
    mean_cosine = directions[scattered, 2].mean()
    assert abs(mean_cosine - 2 / 3) < 5 * np.sqrt(1 / 18 / scattered.sum())


@pytest.mark.parametrize("generator", GENERATORS)
def test_isotropic_source_is_uniform_on_the_sphere(backend, generator):
    n = 300000
    sim = Simulation(backend=backend)
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    sim.geometry = MeshGeometry(sphere.vertices, sphere.faces)
    sim.wavelength_um = 1.0
    sim.materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.3 + 0j)]
    _set_generator(sim, generator)
    sim.ray_generator = IsotropicRayGenerator(number_of_rays=n, center=(0, 0, 0), source_radius=10, offset_radius=1)
    sim.outputs = [OutputType.SOURCE_DIRECTION]
    sim.run()
    directions = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)

    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-5)
    assert (np.abs(directions.mean(axis=0)) < 5 / np.sqrt(3 * n)).all()  # each component has variance 1/3
    assert np.allclose((directions ** 2).mean(axis=0), 1 / 3, atol=5 * np.sqrt(4 / 45 / n))


@pytest.mark.parametrize("generator", GENERATORS)
def test_same_seed_gives_the_same_rays_and_another_seed_different_ones(backend, generator):
    glass = Material(MaterialType.REFRACTIVE, 1.5 + 1e-3j)
    first = _slab_beam_simulation(backend, glass, 20000, generator, seed=5)
    again = _slab_beam_simulation(backend, glass, 20000, generator, seed=5)
    other = _slab_beam_simulation(backend, glass, 20000, generator, seed=6)

    assert (first[1] == again[1]).all()
    assert not (first[1] == other[1]).all()


@pytest.mark.parametrize("generator", GENERATORS)
def test_pipelines_pass_optix_validation_with_each_generator(backend, generator):
    if backend != Backend.OPTIX:
        pytest.skip("validation mode is an OptiX feature")
    # Validation mode checks every payload access against the declared payload size
    sim = Simulation(backend=backend, enable_validation_mode=True)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    sim.geometry = MeshGeometry(sphere.vertices, sphere.faces)
    sim.materials = _materials(2)
    sim.wavelength_um = 1.0
    sim.random_number_generator = generator
    sim.ray_generator = ParallelRayGenerator(number_of_rays=1000, origin=[0, 0, 5], direction=[0, 0, -1], offset_radius=1)
    sim.outputs = [OutputType.SCATTERING_COUNT]
    sim.run()
    assert sim.calculate_volume_fraction([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 1000) == 1.0


def test_the_generator_can_be_changed_between_runs(backend):
    if backend != Backend.OPTIX:
        pytest.skip("only OptiX has a second generator")
    glass = Material(MaterialType.REFRACTIVE, 1.5 + 1e-3j)
    sim = _box_simulation(backend, _materials(2))
    sim.materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), glass]
    sim.ray_generator = ParallelRayGenerator(number_of_rays=20000, origin=[0, 0, 5], direction=[0, 0, -1], offset_radius=0.9)
    sim.outputs = [OutputType.LAST_DIRECTION]

    results = []
    for generator in (RandomNumberGenerator.PCG32, RandomNumberGenerator.MRG32K3A, RandomNumberGenerator.PCG32):
        sim.random_number_generator = generator
        sim.run()
        results.append(sim.get_output_buffer(OutputType.LAST_DIRECTION).copy())

    assert not (results[0] == results[1]).all()  # another generator draws other numbers
    assert (results[0] == results[2]).all()      # switching back reproduces the first run


def test_circular_polarization_at_normal(backend):
    sim = Simulation(backend=backend)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    number_of_rays = 1000
    ray_generator = ParallelRayGenerator(
        number_of_rays = number_of_rays, origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.wavelength_um = 1
    sim.max_scattering_count = 1
    sim.stokes_vector = [1, 0, 0, -1]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (count == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(count))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [
                       1, 0, 0, 1], atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [
                       1, 0, 0, -1], atol=1e-2), "Stokes vector not as expected on average"


def test_linear_polarization_at_normal(backend):
    sim = Simulation(backend=backend)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    ray_generator = ParallelRayGenerator(
        number_of_rays=1000, origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)


    sim.wavelength_um = 1
    sim.max_scattering_count = 1
    sim.stokes_vector = [1, 1, 0, 0]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (count == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(count))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [
                       1, 1, 0, 0], atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [
                       1, 1, 0, 0], atol=1e-2), "Stokes vector not as expected on average"


def test_linear_u_polarization_at_normal(backend):
    sim = Simulation(backend=backend)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    ray_generator = ParallelRayGenerator(
        number_of_rays=1000, origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.wavelength_um = 1
    sim.max_scattering_count = 1
    s = [1, 0, 1, 0]
    sim.stokes_vector = s
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (count == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(count))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [1, 0, -1, 0],
                       atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [1, 0, 1, 0],
                       atol=1e-2), "Stokes vector not as expected on average"



def test_ref_brewster_p_pol(backend):
    sim = Simulation(backend=backend)
    # brewster angle for air to glass
    brewster_angle = np.arctan(1.5)

    k = np.array([-np.sin(brewster_angle), 0, -
                 np.cos(brewster_angle)], dtype=float)

    origin = -k
    ray_generator = ParallelRayGenerator(
        number_of_rays=1000, origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.q_minus_axis_seed = [0, 1, 0]
    sim.wavelength_um = 1
    sim.stokes_vector = [1, 1, 0, 0]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (count == 1) & (k_out[:, 2] > 0)
    
    assert escape_up.sum() == 0, "No rays should be reflected at brewster angle"

def test_layered_absorption(backend):
    # create a mesh representation af a sphere
    slab_1 = trimesh.creation.box(extents=(8, 8, 5), transform=trimesh.transformations.translation_matrix((0, 0, 2.5)))
    slab_2 = trimesh.creation.box(extents=(9, 9, 3), transform=trimesh.transformations.translation_matrix((0, 0, 1.5)))
    slab_3 = trimesh.creation.box(extents=(10, 10, 1), transform=trimesh.transformations.translation_matrix((0, 0, 0.5)))

    scene = trimesh.Scene([slab_1, slab_2, slab_3])

    slab_1_mesh = MeshGeometry(slab_1.vertices, slab_1.faces)  # water-like
    slab_2_mesh = MeshGeometry(slab_2.vertices, slab_2.faces)  # water-like
    slab_3_mesh = MeshGeometry(slab_3.vertices, slab_3.faces)  # water-like

    transforms = np.eye(4)[np.newaxis,:3].repeat(3, axis=0).astype(np.float32)
    transforms[0,2,3] = -(2.5+2.5)
    transforms[1,2,3] = -(1.5+2.5)
    transforms[2,2,3] = -(0.5+2.5)
    material_ids = [2, 1, 3]
    particle_ids = [0, 1, 2]

    sim = Simulation(backend=backend)
    sim.geometry = InstanceGeometry([slab_1_mesh, slab_2_mesh, slab_3_mesh], transforms, particle_ids, material_ids)

    sim.wavelength_um = 0.55  # green light

    sim.materials = [
        Material(MaterialType.REFRACTIVE, 1+0j),  # vacuum
        Material(MaterialType.REFRACTIVE, 1.33+1e-5j), # water
        Material(MaterialType.REFRACTIVE, 1.5+0j), # glass
        Material(MaterialType.REFRACTIVE, 1.5+1e-4j) # dark glass
    ]

    number_of_rays = 1_000_000
    sim.length_unit = LengthUnit.MILLI_METER
    sim.ray_generator = ParallelRayGenerator(
        number_of_rays=number_of_rays, origin=[0, 0, 1000], direction=[0, 0, -1], offset_radius=0)
    sim.outputs = [OutputType.RAY_STATE, OutputType.LAST_MEDIUM_ID, OutputType.SCATTERING_ANGLE, OutputType.LAST_POSITION, OutputType.SCATTERING_COUNT
]

    r_pts = []
    t_pts = []
    a_2s = []
    a_3s = []
    a_4s = []
    import random

    for _ in range(10):
        sim.seed = random.randint(0, 2**31 - 1)
        sim.run()

        scattering_angles = sim.get_output_buffer(OutputType.SCATTERING_ANGLE)
        last_medium_ids = sim.get_output_buffer(OutputType.LAST_MEDIUM_ID)
        ray_states = sim.get_output_buffer(OutputType.RAY_STATE)
        last_position = sim.get_output_buffer(OutputType.LAST_POSITION)
        count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
        print(count.mean(), count.max())

        escaped = ray_states == 0
        absorbed = ray_states == 1
        escape_up = escaped & np.isclose(scattering_angles, np.pi)
        escape_down = escaped & np.isclose(scattering_angles, 0)

        absorbed_medium = last_medium_ids[absorbed]
        absorption_positions = last_position[absorbed]
        r_pt = np.sum(escape_up) / number_of_rays
        t_pt = np.sum(escape_down) / number_of_rays

        a_2 = np.sum(absorbed_medium == 1 & (absorption_positions[:,2] > -2.5)) / number_of_rays
        a_3 = np.sum(absorbed_medium == 3) / number_of_rays
        a_4 = np.sum(absorbed_medium == 3 & (absorption_positions[:,2] < -2.5)) / number_of_rays

        r_pts.append(r_pt)
        t_pts.append(t_pt)
        a_2s.append(a_2)
        a_3s.append(a_3)
        a_4s.append(a_4)

    r_mean = np.mean(r_pts)
    t_mean = np.mean(t_pts)
    a_2_mean = np.mean(a_2s)
    a_3_mean = np.mean(a_3s)
    a_4_mean = np.mean(a_4s)

    # expected values from TMM calculation
    r_expected = 0.04560143096001716
    a_2_expected = 0.1960392553688567
    a_3_expected = 0.6835196015048082
    a_4_expected = 0.016259660787520183
    t_expected = 0.05858005137879813

    assert np.isclose(r_mean, r_expected, atol=1e-3)
    assert np.isclose(t_mean, t_expected, atol=1e-3)
    assert np.isclose(a_2_mean, a_2_expected, atol=1e-3)
    assert np.isclose(a_3_mean, a_3_expected, atol=1e-3)
    assert np.isclose(a_4_mean, a_4_expected, atol=1e-3)

def test_a_build_has_a_backend_and_creates_it_by_default():
    assert available_backends()
    sim = Simulation()
    assert sim.random_number_generator == RandomNumberGenerator.PCG32


def test_embree_only_supports_pcg32():
    if Backend.EMBREE not in available_backends():
        pytest.skip("this build has no Embree backend")
    sim = Simulation(backend=Backend.EMBREE)
    with pytest.raises(ValueError, match="random number generator"):
        sim.random_number_generator = RandomNumberGenerator.MRG32K3A


def test_a_backend_that_is_not_built_is_rejected():
    missing = [b for b in (Backend.OPTIX, Backend.EMBREE) if b not in available_backends()]
    if not missing:
        pytest.skip("this build has every backend")
    with pytest.raises(RuntimeError, match="does not contain"):
        Simulation(backend=missing[0])


def _glass_ball():
    ball = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    return MeshGeometry(ball.vertices, ball.faces)


def _transform(scale=1.0, shift=(0, 0, 0)):
    matrix = np.eye(4, dtype=np.float32)[:3]
    matrix[:, :3] *= scale
    matrix[:, 3] = shift
    return matrix


def _beam_on_instances(backend, geometry, materials, origin=(0, 0, 10), rays=1000):
    sim = Simulation(backend=backend)
    sim.geometry = geometry
    sim.materials = materials
    sim.wavelength_um = 1.0
    sim.ray_generator = ParallelRayGenerator(number_of_rays=rays, origin=origin, direction=[0, 0, -1], offset_radius=0.05)
    sim.outputs = [OutputType.RAY_STATE, OutputType.LAST_POSITION, OutputType.SCATTERING_COUNT]
    sim.run()
    return {o: sim.get_output_buffer(o) for o in sim.outputs}


def test_instance_is_placed_by_its_transform(backend):
    # An absorbing ball of radius 2 at x = 3 stops a beam at its top
    absorber = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.DIFFUSE, 0.0)]
    geometry = InstanceGeometry([_glass_ball()], _transform(2.0, (3, 0, 0))[None], [0], [1])
    result = _beam_on_instances(backend, geometry, absorber, origin=(3, 0, 10))
    assert (result[OutputType.RAY_STATE] == 1).all()
    position = result[OutputType.LAST_POSITION]
    assert np.allclose(position[:, 0], 3, atol=0.2)
    assert (position[:, 2] > 1.95).all() and (position[:, 2] < 2.0001).all()

    # beside the ball the beam passes
    result = _beam_on_instances(backend, geometry, absorber, origin=(0, 0, 10))
    assert (result[OutputType.RAY_STATE] == 0).all()
    assert (result[OutputType.SCATTERING_COUNT] == 0).all()


def test_each_instance_uses_its_own_material(backend):
    materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 0j),
                 Material(MaterialType.DIFFUSE, 0.0)]
    transforms = np.stack([_transform(1.0, (0, 0, 0)), _transform(1.0, (5, 0, 0))])
    geometry = InstanceGeometry([_glass_ball()], transforms, [0, 0], [1, 2])
    glass = _beam_on_instances(backend, geometry, materials, origin=(0, 0, 10))
    black = _beam_on_instances(backend, geometry, materials, origin=(5, 0, 10))
    assert (glass[OutputType.RAY_STATE] == 0).all() and (glass[OutputType.SCATTERING_COUNT] >= 1).all()
    assert (black[OutputType.RAY_STATE] == 1).all()


# The material of a hit in nested instances is the ID of the innermost instance, the one that holds
# the mesh. The OptiX documentation does not specify this for nested instance acceleration
# structures; the Embree backend copies the behaviour, and these tests fail if either changes.

@pytest.mark.parametrize("inner, outer", [(1, 2), (2, 1)])
def test_nested_instances_use_the_innermost_material(backend, inner, outer):
    materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 0j),
                 Material(MaterialType.DIFFUSE, 0.0)]
    inside = InstanceGeometry([_glass_ball()], _transform()[None], [0], [inner])
    geometry = InstanceGeometry([inside], _transform(1.0, (0, 0, 1))[None], [0], [outer])
    result = _beam_on_instances(backend, geometry, materials, origin=(0, 0, 10))
    absorbed = (result[OutputType.RAY_STATE] == 1).all()
    assert absorbed == (inner == 2)


@pytest.mark.parametrize("absorbing_level", [0, 1, 2])
def test_only_the_innermost_of_three_levels_decides_the_material(backend, absorbing_level):
    # Level 0 is the innermost. Only a black innermost instance absorbs; the same ID on a
    # level above it, while the innermost is glass, must not.
    materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 0j),
                 Material(MaterialType.DIFFUSE, 0.0)]
    ids = [2 if level == absorbing_level else 1 for level in range(3)]
    geometry = _glass_ball()
    for level, shift in enumerate([(0, 0, 0), (0, 0, 1), (0, 0, 1)]):
        geometry = InstanceGeometry([geometry], _transform(1.0, shift)[None], [0], [ids[level]])
    result = _beam_on_instances(backend, geometry, materials, origin=(0, 0, 10))
    assert (result[OutputType.RAY_STATE] == 1).all() == (absorbing_level == 0)


def test_one_inner_instance_used_by_two_outer_instances_with_different_ids(backend):
    # The outer IDs differ, the inner geometry is shared and has the ID of the glass
    materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 0j),
                 Material(MaterialType.DIFFUSE, 0.0)]
    inner = InstanceGeometry([_glass_ball()], _transform()[None], [0], [1])
    transforms = np.stack([_transform(1.0, (0, 0, 0)), _transform(1.0, (5, 0, 0))])
    geometry = InstanceGeometry([inner], transforms, [0, 0], [2, 2])
    for x in (0, 5):
        result = _beam_on_instances(backend, geometry, materials, origin=(x, 0, 10))
        assert (result[OutputType.RAY_STATE] == 0).all()
        assert (result[OutputType.SCATTERING_COUNT] >= 1).all()


def test_instances_can_share_a_mesh_and_a_geometry_can_be_reused(backend):
    ball = _glass_ball()
    materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, 1.5 + 0j)]
    transforms = np.stack([_transform(1.0, (4 * i, 0, 0)) for i in range(50)])
    geometry = InstanceGeometry([ball], transforms, [0] * 50, [1] * 50)
    for _ in range(2):
        hit = _beam_on_instances(backend, geometry, materials, origin=(12, 0, 10))
        gap = _beam_on_instances(backend, geometry, materials, origin=(14, 0, 10))
        assert (hit[OutputType.SCATTERING_COUNT] >= 1).all()
        assert (gap[OutputType.SCATTERING_COUNT] == 0).all()
