import numpy as np
import trimesh
from photontracer import LengthUnit, Material, MaterialType, MeshGeometry, InstanceGeometry, Simulation, ParallelRayGenerator, OutputType

def test_reflection_angle():
    sim = Simulation(gpu_id=0)
    angle_deg = 45
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    origin = -k
    ray_generator = ParallelRayGenerator(
        origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))

    sim.number_of_rays = 1000
    sim.wavelength_um = 1
    sim.stokes_vector = [1, 0, 0, 0]
    sim.geometry = MeshGeometry(slab.vertices, slab.faces)
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    depth = sim.get_output_buffer(OutputType.DEPTH)


    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (depth == 1) & (k_out[:, 2] > 0)
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


def test_circular_polarization_at_normal():
    sim = Simulation(gpu_id=0)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    ray_generator = ParallelRayGenerator(
        origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.number_of_rays = 1000
    sim.wavelength_um = 1
    sim.max_depth = 1
    sim.stokes_vector = [1, 0, 0, -1]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    depth = sim.get_output_buffer(OutputType.DEPTH)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (depth == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(depth))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [
                       1, 0, 0, 1], atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [
                       1, 0, 0, -1], atol=1e-2), "Stokes vector not as expected on average"


def test_linear_polarization_at_normal():
    sim = Simulation(gpu_id=0)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    ray_generator = ParallelRayGenerator(
        origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
               OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]

    sim.number_of_rays = 1000
    sim.wavelength_um = 1
    sim.max_depth = 1
    sim.stokes_vector = [1, 1, 0, 0]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    depth = sim.get_output_buffer(OutputType.DEPTH)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (depth == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(depth))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [
                       1, 1, 0, 0], atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [
                       1, 1, 0, 0], atol=1e-2), "Stokes vector not as expected on average"


def test_linear_u_polarization_at_normal():
    sim = Simulation(gpu_id=0)
    angle_deg = 0
    angle_rad = np.radians(angle_deg)
    k = np.array([-np.sin(angle_rad), 0, -np.cos(angle_rad)], dtype=float)
    print(f"{k=}")
    origin = -k
    ray_generator = ParallelRayGenerator(
        origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.number_of_rays = 1000
    sim.wavelength_um = 1
    sim.max_depth = 1
    s = [1, 0, 1, 0]
    sim.stokes_vector = s
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    stokes = sim.get_output_buffer(OutputType.STOKES_VECTOR)
    depth = sim.get_output_buffer(OutputType.DEPTH)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (depth == 1) & (k_out[:, 2] > 0)
    print(escape_up.sum() / len(depth))

    stokes_up = stokes[escape_up]
    stokes_down = stokes[~escape_up]

    print(stokes_up.mean(axis=0))
    assert np.allclose(stokes_up.mean(axis=0), [1, 0, -1, 0],
                       atol=1e-2), "Stokes vector not as expected on average"

    print(stokes_down.mean(axis=0))
    assert np.allclose(stokes_down.mean(axis=0), [1, 0, 1, 0],
                       atol=1e-2), "Stokes vector not as expected on average"



def test_ref_brewster_p_pol():
    sim = Simulation(gpu_id=0)
    # brewster angle for air to glass
    brewster_angle = np.arctan(1.5)

    k = np.array([-np.sin(brewster_angle), 0, -
                 np.cos(brewster_angle)], dtype=float)

    origin = -k
    ray_generator = ParallelRayGenerator(
        origin=origin, direction=k, offset_radius=0)
    # create slab at z=0
    slab = trimesh.creation.box(extents=[
                                10, 10, 1], transform=trimesh.transformations.translation_matrix([0, 0, -0.5]))
    geometry = MeshGeometry(slab.vertices, slab.faces)

    sim.number_of_rays = 1000
    sim.q_minus_axis_seed = [0, 1, 0]
    sim.wavelength_um = 1
    sim.stokes_vector = [1, 1, 0, 0]
    sim.geometry = geometry
    vacuum = Material(MaterialType.REFRACTIVE, 1+0j)
    glass = Material(MaterialType.REFRACTIVE, 1.5+0j)
    sim.materials = [vacuum, glass]
    sim.ray_generator = ray_generator
    sim.outputs = [OutputType.SOURCE_DIRECTION, OutputType.DEPTH,
                   OutputType.LAST_DIRECTION, OutputType.STOKES_VECTOR]
    sim.run()

    k_out = sim.get_output_buffer(OutputType.LAST_DIRECTION)
    k_in = sim.get_output_buffer(OutputType.SOURCE_DIRECTION)
    depth = sim.get_output_buffer(OutputType.DEPTH)

    # check if k_in if equal to k for all rays
    k_norm = k / np.linalg.norm(k)
    assert np.allclose(k_in, k_norm), "Input directions do not match"

    escape_up = (depth == 1) & (k_out[:, 2] > 0)
    
    assert escape_up.sum() == 0, "No rays should be reflected at brewster angle"

def test_layered_absorption():
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

    sim = Simulation(gpu_id=0)
    sim.geometry = InstanceGeometry([slab_1_mesh, slab_2_mesh, slab_3_mesh], transforms, particle_ids, material_ids)

    sim.wavelength_um = 0.55  # green light

    sim.materials = [
        Material(MaterialType.REFRACTIVE, 1+0j),  # vacuum
        Material(MaterialType.REFRACTIVE, 1.33+1e-5j), # water
        Material(MaterialType.REFRACTIVE, 1.5+0j), # glass
        Material(MaterialType.REFRACTIVE, 1.5+1e-4j) # dark glass
    ]

    sim.length_unit = LengthUnit.MILLI_METER
    sim.number_of_rays = 1_000_000
    sim.ray_generator = ParallelRayGenerator(
        origin=[0, 0, 1000], direction=[0, 0, -1], offset_radius=0)
    sim.outputs = [OutputType.RAY_STATE, OutputType.LAST_MEDIUM_ID, OutputType.SCATTERING_ANGLE, OutputType.LAST_POSITION, OutputType.DEPTH
]

    r_pts = []
    t_pts = []
    a_1s = []
    a_3s = []
    a_5s = []
    import random

    for _ in range(10):
        sim.seed = random.randint(0, 2**31 - 1)
        sim.run()

        scattering_angles = sim.get_output_buffer(OutputType.SCATTERING_ANGLE)
        last_medium_ids = sim.get_output_buffer(OutputType.LAST_MEDIUM_ID)
        ray_states = sim.get_output_buffer(OutputType.RAY_STATE)
        last_position = sim.get_output_buffer(OutputType.LAST_POSITION)
        depth = sim.get_output_buffer(OutputType.DEPTH)
        print(depth.mean(), depth.max())

        escaped = ray_states == 0
        absorbed = ray_states == 1
        escape_up = escaped & np.isclose(scattering_angles, np.pi)
        escape_down = escaped & np.isclose(scattering_angles, 0)

        absorbed_medium = last_medium_ids[absorbed]
        absorption_positions = last_position[absorbed]
        r_pt = np.sum(escape_up) / sim.number_of_rays
        t_pt = np.sum(escape_down) / sim.number_of_rays

        a_1 = np.sum(absorbed_medium == 1 & (absorption_positions[:,2] > -2.5)) / sim.number_of_rays
        a_3 = np.sum(absorbed_medium == 3) / sim.number_of_rays
        a_5 = np.sum(absorbed_medium == 3 & (absorption_positions[:,2] < -2.5)) / sim.number_of_rays

        r_pts.append(r_pt)
        t_pts.append(t_pt)
        a_1s.append(a_1)
        a_3s.append(a_3)
        a_5s.append(a_5)

    r_mean = np.mean(r_pts)
    t_mean = np.mean(t_pts)
    a_2_mean = np.mean(a_1s)
    a_3_mean = np.mean(a_3s)
    a_4_mean = np.mean(a_5s)

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