// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include "backend_factory.h"
#include "log.h"
#include "simulation.h"
#include "geometries/mesh_geometry.h"
#include "geometries/instance_geometry.h"
#include "photontracer.h" // For enums such as GeometryType, MaterialType, Material, etc.
#include "photontracer_bindings_doc.h"

namespace py = pybind11;

namespace PYBIND11_NAMESPACE
{
namespace detail
{
/// Converts a float vector from a NumPy array, tuple or list of N numbers, and to a tuple
template <typename Vector, size_t N>
struct float_vector_caster
{
    static_assert(sizeof(Vector) == N * sizeof(float), "the components are copied as packed floats");

    bool load(handle src, bool)
    {
        float components[N];
        if (isinstance<array>(src))
        {
            auto values = array_t<float, array::c_style | array::forcecast>::ensure(src);
            if (!values)
            {
                return false;
            }
            if (values.size() != static_cast<ssize_t>(N))
            {
                throw value_error("expected " + std::to_string(N) + " values, got an array of " + std::to_string(values.size()));
            }
            std::copy(values.data(), values.data() + N, components);
        }
        else if (isinstance<tuple>(src) || isinstance<list>(src))
        {
            const auto values = reinterpret_borrow<sequence>(src);
            if (values.size() != N)
            {
                throw value_error("expected " + std::to_string(N) + " values, got " + std::to_string(values.size()));
            }
            for (size_t i = 0; i < N; ++i)
            {
                make_caster<float> component;
                if (!component.load(values[i], true))
                {
                    return false;
                }
                components[i] = cast_op<float>(component);
            }
        }
        else
        {
            return false;
        }
        std::memcpy(&value, components, sizeof(Vector));
        return true;
    }

    static handle cast(const Vector &src, return_value_policy, handle)
    {
        const float *components = reinterpret_cast<const float *>(&src);
        tuple result(N);
        for (size_t i = 0; i < N; ++i)
        {
            result[i] = float_(components[i]);
        }
        return result.release();
    }

    Vector value;
};

// The input types are defined in photontracer/_typing.py
template <>
struct type_caster<float3> : float_vector_caster<float3, 3>
{
    static constexpr auto name = io_name("photontracer._typing.Vector3", "tuple[float, float, float]");
    template <typename T>
    using cast_op_type = movable_cast_op_type<T>;
    operator float3 *() { return &value; }
    operator float3 &() { return value; }
    operator float3 &&() && { return std::move(value); }
};

template <>
struct type_caster<float4> : float_vector_caster<float4, 4>
{
    static constexpr auto name = io_name("photontracer._typing.Vector4", "tuple[float, float, float, float]");
    template <typename T>
    using cast_op_type = movable_cast_op_type<T>;
    operator float4 *() { return &value; }
    operator float4 &() { return value; }
    operator float4 &&() && { return std::move(value); }
};
} // namespace detail
} // namespace PYBIND11_NAMESPACE

PYBIND11_MODULE(photontracer_bindings, m)
{
    m.doc() = doc::module;

    // Expose enums.
    py::enum_<GeometryType>(m, "GeometryType", "Type of a geometry, returned by IGeometry.get_type().")
        .value("MESH", MESH, "A single triangle mesh (MeshGeometry).")
        .value("MESH_INSTANCED", MESH_INSTANCED, "Instances of geometries placed by transforms (InstanceGeometry).")
        .export_values();

    py::enum_<RandomNumberGenerator>(m, "RandomNumberGenerator", doc::RandomNumberGenerator)
        .value("PCG32", RandomNumberGenerator::PCG32,
               "PCG32, the default. Cheap to seed, which matters most for scenes with little physics per ray.")
        .value("MRG32K3A", RandomNumberGenerator::MRG32K3A,
               "curand's MRG32k3a, the generator of the 1.0 releases, to reproduce their results ray by ray. OptiX only.")
        .export_values();

    py::enum_<BackendType>(m, "Backend", "Where a Simulation traces its rays; pass one as backend to Simulation.")
        .value("OPTIX", BackendType::OPTIX, "NVIDIA GPU with OptiX.")
        .value("EMBREE", BackendType::EMBREE, "CPU with Embree and TBB. Supports only the PCG32 generator.")
        .export_values();

    py::enum_<TraceEvent>(m, "TraceEvent", "What happened to a traced ray in one step (event field of Simulation.get_trace()).")
        .value("INTERACTION", TRACE_INTERACTION, "Reflected, refracted or scattered; the ray goes on.")
        .value("ESCAPED", TRACE_ESCAPED, "The ray left the scene.")
        .value("ABSORBED", TRACE_ABSORBED, "The ray was absorbed.")
        .value("MAX_SCATTERING", TRACE_MAX_SCATTERING, "The ray reached Simulation.max_scattering_count.")
        .value("ERROR", TRACE_ERROR, "The ray ended on an error.");

    py::enum_<LengthUnit>(m, "LengthUnit", doc::LengthUnit)
        .value("MICRO_METER", LengthUnit::MICRO_METER, "Micrometres (μm).")
        .value("MILLI_METER", LengthUnit::MILLI_METER, "Millimetres (mm).")
        .value("METER", LengthUnit::METER, "Metres (m).")
        .export_values();

    py::enum_<MaterialType>(m, "MaterialType", doc::MaterialType)
        .value("DIFFUSE", DIFFUSE, "Lambertian reflection with an albedo in [0, 1]; the rest is absorbed.")
        .value("REFRACTIVE", REFRACTIVE,
               "Fresnel reflection and refraction at the surface of a medium with a complex refractive index n + ik.")
        .value("REFLECTIVE", REFLECTIVE,
               "Specular (metallic) reflection with a reflectivity in [0, 1] and an optional fuzziness in [0, 1).")
        .value("VOLUME_SCATTERING", VOLUME_SCATTERING,
               "A participating medium with a complex refractive index, a scattering coefficient and a "
               "Henyey-Greenstein asymmetry parameter g.")
        .export_values();

    py::enum_<OutputType>(m, "OutputType", doc::OutputType)
        .value("LAST_DIRECTION", OutputType::LAST_DIRECTION, doc::OutputType_LAST_DIRECTION)
        .value("LAST_POSITION", OutputType::LAST_POSITION, doc::OutputType_LAST_POSITION)
        .value("RAY_STATE", OutputType::RAY_STATE, doc::OutputType_RAY_STATE)
        .value("LAST_MEDIUM_ID", OutputType::LAST_MEDIUM_ID, doc::OutputType_LAST_MEDIUM_ID)
        .value("SCATTERING_COUNT", OutputType::SCATTERING_COUNT, doc::OutputType_SCATTERING_COUNT)
        .value("NUMBER_OF_WARNINGS", OutputType::NUMBER_OF_WARNINGS, doc::OutputType_NUMBER_OF_WARNINGS)
        .value("STOKES_VECTOR", OutputType::STOKES_VECTOR, doc::OutputType_STOKES_VECTOR)
        .value("STOKES_VECTOR_IN", OutputType::STOKES_VECTOR_IN, doc::OutputType_STOKES_VECTOR_IN)
        .value("OPTICAL_PATH_LENGTH", OutputType::OPTICAL_PATH_LENGTH, doc::OutputType_OPTICAL_PATH_LENGTH)
        .value("SOURCE_DIRECTION", OutputType::SOURCE_DIRECTION, doc::OutputType_SOURCE_DIRECTION)
        .value("SOURCE_POSITION", OutputType::SOURCE_POSITION, doc::OutputType_SOURCE_POSITION)
        .value("SCATTERING_ANGLE", OutputType::SCATTERING_ANGLE, doc::OutputType_SCATTERING_ANGLE)
        .value("Q_MINUS_AXIS_IN", OutputType::Q_MINUS_AXIS_IN, doc::OutputType_Q_MINUS_AXIS_IN)
        .value("DIRECTION_HISTOGRAM_HEALPIX", OutputType::DIRECTION_HISTOGRAM_HEALPIX, doc::OutputType_DIRECTION_HISTOGRAM_HEALPIX)
        .value("FIRST_POSITION", OutputType::FIRST_POSITION, doc::OutputType_FIRST_POSITION)
        .export_values();

    // Bind RefractiveIndex with support for Python complex numbers.
    py::class_<RefractiveIndex>(m, "RefractiveIndex", doc::RefractiveIndex)
        .def(py::init<>(), "Create a refractive index of 0.")
        .def(py::init<float, float>(), py::arg("r"), py::arg("i"), "Create a refractive index r + i·j from its real and imaginary parts.")
        // Add constructor that takes a Python complex number
        .def(py::init([](const std::complex<double> &c)
                      {
            RefractiveIndex ri;
            ri.r = static_cast<float>(c.real());
            ri.i = static_cast<float>(c.imag());
            return ri; }),
             py::arg("complex_value"), "Create a refractive index from a complex number, e.g. 1.31+0j.")
        .def_readwrite("r", &RefractiveIndex::r, "Real part n of the refractive index.")
        .def_readwrite("i", &RefractiveIndex::i, "Imaginary part k of the refractive index, the extinction coefficient.")
        // Add property to get/set as complex number
        .def_property("complex",
                      // Getter: return as Python complex
                      [](const RefractiveIndex &ri)
                      { return std::complex<float>(ri.r, ri.i); },
                      // Setter: accept Python complex
                      [](RefractiveIndex &ri, const std::complex<float> &c)
                      {
                ri.r = c.real();
                ri.i = c.imag(); },
                      "The refractive index as a complex number n + ik.")
        .def("__repr__", [](const RefractiveIndex &ri)
             { return "<RefractiveIndex (real=" + std::to_string(ri.r) +
                      ", imag=" + std::to_string(ri.i) + ")>"; });

    // Bind the Diffuse variant inside MaterialProperties.
    py::class_<MaterialProperties::Diffuse>(m, "Diffuse", "Properties of a Lambertian (diffuse) surface.")
        .def(py::init<>())
        .def_readwrite("albedo", &MaterialProperties::Diffuse::albedo,
                       "Fraction of the light reflected diffusely, in [0, 1]; the rest is absorbed.")
        .def("__repr__", [](const MaterialProperties::Diffuse &l)
             { return "<Diffuse albedo=" + std::to_string(l.albedo) + ">"; });

    // Bind the Refractive variant inside MaterialProperties.
    py::class_<MaterialProperties::Refractive>(m, "Refractive",
                                               "Properties of a refractive medium: Fresnel reflection and refraction at its surface.")
        .def(py::init<>())
        .def_readwrite("refractive_index", &MaterialProperties::Refractive::refractiveIndex,
                       "Complex refractive index n + ik of the medium; k causes absorption inside it.")
        .def("__repr__", [](const MaterialProperties::Refractive &r)
             { return "<Refractive refractive_index=" + std::to_string(r.refractiveIndex.r) +
                      " + " + std::to_string(r.refractiveIndex.i) + "i>"; });

    // Bind the Reflective variant inside MaterialProperties.
    py::class_<MaterialProperties::Reflective>(m, "Reflective", "Properties of a metallic (specular) surface.")
        .def(py::init<>())
        .def_readwrite("reflectivity", &MaterialProperties::Reflective::reflectivity,
                       "Fraction of the light reflected, in [0, 1]; the rest is absorbed.")
        .def_readwrite("fuzziness", &MaterialProperties::Reflective::fuzziness,
                       "Roughness in [0, 1): 0 is a perfect mirror, values towards 1 spread the reflection over nearly a hemisphere.")
        .def("__repr__", [](const MaterialProperties::Reflective &r)
             { return "<Reflective reflectivity=" + std::to_string(r.reflectivity) +
                      ", fuzziness=" + std::to_string(r.fuzziness) + ">"; });

    // Bind the VolumeScattering variant inside MaterialProperties.
    py::class_<MaterialProperties::VolumeScattering>(m, "VolumeScattering", doc::VolumeScattering)
        .def(py::init<>())
        .def_readwrite("refractive_index", &MaterialProperties::VolumeScattering::refractiveIndex,
                       "Complex refractive index n + ik of the medium; k gives the absorption coefficient.")
        .def_readwrite("scattering_coefficient", &MaterialProperties::VolumeScattering::scatteringCoefficient,
                       "Scattering coefficient per geometry unit (Simulation.length_unit).")
        .def_readwrite("asymetry_parameter", &MaterialProperties::VolumeScattering::asymetryParameter,
                       "Henyey-Greenstein asymmetry parameter g in (-1, 1): 0 isotropic, > 0 forward, < 0 backward scattering.")
        .def("__repr__", [](const MaterialProperties::VolumeScattering &v)
             { return "<VolumeScattering refractive_index=" + std::to_string(v.refractiveIndex.r) +
                      " + " + std::to_string(v.refractiveIndex.i) + "i, scattering_coefficient=" +
                      std::to_string(v.scatteringCoefficient) + ", asymetry_parameter=" + std::to_string(v.asymetryParameter) + ">"; });

    // Bind MaterialProperties as a whole, providing property getters/setters for each variant.
    py::class_<MaterialProperties>(m, "MaterialProperties", doc::MaterialProperties)
        .def(py::init([]()
                      {
            MaterialProperties mp = {};
            return mp; }))
        .def_property("diffuse",
                      // Getter: return a reference to the Diffuse struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Diffuse &
                      { return mp.diffuse; },
                      // Setter: set the diffuse field.
                      [](MaterialProperties &mp, const MaterialProperties::Diffuse &l)
                      { mp.diffuse = l; }, "Properties of a DIFFUSE material.")
        .def_property("refractive",
                      // Getter: return a reference to the Refractive struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Refractive &
                      { return mp.refractive; },
                      // Setter: set the refractive field.
                      [](MaterialProperties &mp, const MaterialProperties::Refractive &r)
                      { mp.refractive = r; }, "Properties of a REFRACTIVE material.")
        .def_property("reflective",
                      // Getter: return a reference to the Reflective struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Reflective &
                      { return mp.reflective; },
                      // Setter: set the reflective field.
                      [](MaterialProperties &mp, const MaterialProperties::Reflective &r)
                      { mp.reflective = r; }, "Properties of a REFLECTIVE material.")
        .def_property("volume_scattering",
                      // Getter: return a reference to the VolumeScattering struct.
                      [](MaterialProperties &mp) -> MaterialProperties::VolumeScattering &
                      { return mp.volumeScattering; },
                      // Setter: set the volumeScattering field.
                      [](MaterialProperties &mp, const MaterialProperties::VolumeScattering &v)
                      { mp.volumeScattering = v; }, "Properties of a VOLUME_SCATTERING material.")
        .def("__repr__", [](const MaterialProperties &)
             { return "<MaterialProperties>"; });

    // Bind Material.
    py::class_<Material>(m, "Material", doc::Material)
        .def(py::init<>(), "Create a material without parameters.")
        // Constructor for REFRACTIVE material with complex refractive index
        .def(py::init([](MaterialType type, const std::complex<float> &refractive_index)
                      {
            if (type != REFRACTIVE) {
                throw py::value_error("Complex refractive index can only be used with REFRACTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.refractive.refractiveIndex.r = refractive_index.real();
            mat.properties.refractive.refractiveIndex.i = refractive_index.imag();
            return mat; }),
             py::arg("type"), py::arg("refractive_index"),
             "Create a REFRACTIVE material from a complex refractive index, e.g. 1.5+0j.")
        // Constructor for VOLUME_SCATTERING material with complex refractive index, scattering coefficient, and asymetryParameter
        .def(py::init([](MaterialType type, const std::complex<float> &refractive_index, float scattering_coefficient, float asymetryParameter)
                      {
            if (type != VOLUME_SCATTERING) {
                throw py::value_error("Parameters can only be used with VOLUME_SCATTERING material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.volumeScattering.refractiveIndex.r = refractive_index.real();
            mat.properties.volumeScattering.refractiveIndex.i = refractive_index.imag();
            mat.properties.volumeScattering.scatteringCoefficient = scattering_coefficient;
            mat.properties.volumeScattering.asymetryParameter = asymetryParameter;
            return mat; }),
             py::arg("type"), py::arg("refractive_index"), py::arg("scattering_coefficient"), py::arg("g"),
             doc::Material_init_volume_scattering)
        // Constructor for DIFFUSE material with albedo
        .def(py::init([](MaterialType type, float albedo)
                      {
            if (type != DIFFUSE) {
                throw py::value_error("Albedo parameter can only be used with DIFFUSE material type");
            }
            if (albedo < 0.0f || albedo > 1.0f) {
                throw py::value_error("Albedo must be between 0.0 and 1.0, got " + std::to_string(albedo));
            }
            Material mat;
            mat.type = type;
            mat.properties.diffuse.albedo = albedo;
            return mat; }),
             py::arg("type"), py::arg("albedo"), "Create a DIFFUSE material with an albedo in [0, 1].")
        // Constructor for REFLECTIVE material with reflectivity and fuzziness
        .def(py::init([](MaterialType type, float reflectivity, float fuzziness)
                      {
            if (type != REFLECTIVE) {
                throw py::value_error("Reflectivity and fuzziness can only be used with REFLECTIVE material type");
            }
            if (reflectivity < 0.0f || reflectivity > 1.0f) {
                throw py::value_error("Reflectivity must be between in [0.0, 1.0], got " + std::to_string(reflectivity));
            }
            if (fuzziness < 0.0f || fuzziness >= 1.0f) {
                throw py::value_error("Fuzziness must be between in [0.0, 1.0), got " + std::to_string(fuzziness));
            }
            Material mat;
            mat.type = type;
            mat.properties.reflective.reflectivity = reflectivity;
            mat.properties.reflective.fuzziness = fuzziness;
            return mat; }),
             py::arg("type"), py::arg("reflectivity"), py::arg("fuzziness"),
             "Create a REFLECTIVE material with a reflectivity in [0, 1] and a fuzziness in [0, 1).")
        // Constructor for REFLECTIVE material with reflectivity only (fuzziness=0)
        .def(py::init([](MaterialType type, float reflectivity)
                      {
            if (type != REFLECTIVE) {
                throw py::value_error("Reflectivity can only be used with REFLECTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.reflective.reflectivity = reflectivity;
            mat.properties.reflective.fuzziness = 0.0f; // Default fuzziness to 0
            return mat; }),
             py::arg("type"), py::arg("reflectivity"), "Create a REFLECTIVE material that is a perfect mirror (fuzziness 0).")
        // Constructor for REFRACTIVE material with RefractiveIndex object
        .def(py::init([](MaterialType type, const RefractiveIndex &ri)
                      {
            if (type != REFRACTIVE) {
                throw py::value_error("RefractiveIndex can only be used with REFRACTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.refractive.refractiveIndex = ri;
            return mat; }),
             py::arg("type"), py::arg("refractive_index"), "Create a REFRACTIVE material from a RefractiveIndex.")
        .def_readwrite("type", &Material::type, "How the material interacts with rays.")
        .def_readwrite("properties", &Material::properties, "The parameters of the material; only those of its type are used.")
        .def("__repr__", [](const Material &mat)
             {
            std::string type_str = (mat.type == DIFFUSE) ? "DIFFUSE" : "REFRACTIVE";
            if (mat.type == DIFFUSE) {
                return "<Material(DIFFUSE, albedo=" + std::to_string(mat.properties.diffuse.albedo) + ")>";
            } else if (mat.type == REFLECTIVE) {
                return "<Material(REFLECTIVE, reflectivity=" + std::to_string(mat.properties.reflective.reflectivity) + 
                       ", fuzziness=" + std::to_string(mat.properties.reflective.fuzziness) + ")>";
            } else if (mat.type == VOLUME_SCATTERING) {
                return "<Material(VOLUME_SCATTERING, n=" + std::to_string(mat.properties.volumeScattering.refractiveIndex.r) + 
                       "+" + std::to_string(mat.properties.volumeScattering.refractiveIndex.i) + "j, scattering_coefficient=" +
                       std::to_string(mat.properties.volumeScattering.scatteringCoefficient) + ", asymetry_parameter=" +
                       std::to_string(mat.properties.volumeScattering.asymetryParameter) + ")>";
            } else {
                return "<Material(REFRACTIVE, n=" + std::to_string(mat.properties.refractive.refractiveIndex.r) + 
                       "+" + std::to_string(mat.properties.refractive.refractiveIndex.i) + "j)>";
            } });

    // Use std::shared_ptr as the holder type to match Simulation::setGeometry.
    py::class_<IGeometry, std::shared_ptr<IGeometry>>(m, "IGeometry", doc::IGeometry)
        .def("get_type", &IGeometry::getType, "Return the type of the geometry.")
        .def("free_device_memory", &IGeometry::freeDeviceMemory,
             "Free the acceleration structures of this geometry and its sub-geometries. The next Simulation.run() builds them again.");

    py::class_<MeshGeometry, IGeometry, std::shared_ptr<MeshGeometry>>(m, "MeshGeometry", doc::MeshGeometry)
        .def(py::init([](py::array_t<float> vertices, py::array_t<unsigned int> indices, bool compact)
                      {
        // Validate vertices array
        if (vertices.ndim() != 2) {
            throw py::value_error("vertices must be a 2D array, got " + 
                                std::to_string(vertices.ndim()) + " dimensions");
        }
        if (vertices.shape(1) != 3) {
            throw py::value_error("vertices must have exactly 3 columns (x, y, z), got " + 
                                std::to_string(vertices.shape(1)) + " columns");
        }
        if (vertices.shape(0) == 0) {
            throw py::value_error("vertices array cannot be empty");
        }
        
        // Validate indices array
        if (indices.ndim() == 1) {
            // 1D array: flattened indices
            if (indices.shape(0) % 3 != 0) {
                throw py::value_error("1D indices array length must be divisible by 3 (triangles), got " + 
                                    std::to_string(indices.shape(0)) + " indices");
            }
        } else if (indices.ndim() == 2) {
            // 2D array: Nx3 indices
            if (indices.shape(1) != 3) {
                throw py::value_error("2D indices array must have exactly 3 columns (triangles), got " + 
                                    std::to_string(indices.shape(1)) + " columns");
            }
        } else {
            throw py::value_error("indices must be a 1D or 2D array, got " + 
                                std::to_string(indices.ndim()) + " dimensions");
        }
        
        // Convert numpy [N,3] array to vector<float3>
        auto vertices_array = vertices.unchecked<2>();
        std::vector<float3> vertex_vec;
        vertex_vec.reserve(vertices_array.shape(0));
        for (py::ssize_t i = 0; i < vertices_array.shape(0); ++i) {
            float3 v;
            v.x = vertices_array(i, 0);
            v.y = vertices_array(i, 1);
            v.z = vertices_array(i, 2);
            vertex_vec.push_back(v);
        }
        
        // Convert indices to vector<unsigned int>
        std::vector<unsigned int> index_vec;
        if (indices.ndim() == 1) {
            // Flattened 1D array
            auto indices_array = indices.unchecked<1>();
            index_vec.reserve(indices_array.shape(0));
            for (py::ssize_t i = 0; i < indices_array.shape(0); ++i) {
                index_vec.push_back(indices_array(i));
            }
        } else {
            // 2D Nx3 array
            auto indices_array = indices.unchecked<2>();
            index_vec.reserve(indices_array.shape(0) * 3);
            for (py::ssize_t i = 0; i < indices_array.shape(0); ++i) {
                for (py::ssize_t j = 0; j < 3; ++j) {
                    index_vec.push_back(indices_array(i, j));
                }
            }
        }
        
        return std::make_shared<MeshGeometry>(std::move(vertex_vec), std::move(index_vec), compact); }),
             py::arg("vertices"), py::arg("indices"), py::arg("compact") = true, doc::MeshGeometry_init);

    // For InstanceGeometry with numpy arrays
    py::class_<InstanceGeometry, IGeometry, std::shared_ptr<InstanceGeometry>>(m, "InstanceGeometry", doc::InstanceGeometry)
        .def(
            py::init([](std::vector<std::shared_ptr<IGeometry>> sub_geometries,
                        py::array_t<float> instance_transforms,
                        py::array_t<unsigned int> particle_type_ids,
                        py::array_t<unsigned int> material_ids)
                     {
                // Validate sub_geometries
                if (sub_geometries.empty()) {
                    throw py::value_error("sub_geometries cannot be empty");
                }
                
                // Validate instance_transforms array [N,4,3]
                if (instance_transforms.ndim() != 3) {
                    throw py::value_error("instance_transforms must be a 3D array, got " + 
                                        std::to_string(instance_transforms.ndim()) + " dimensions");
                }
                if (instance_transforms.shape(1) != 3) {
                    throw py::value_error("instance_transforms must have 3 rows per transform matrix, got " + 
                                        std::to_string(instance_transforms.shape(1)) + " rows");
                }
                if (instance_transforms.shape(2) != 4) {
                    throw py::value_error("instance_transforms must have 4 columns per transform matrix, got " + 
                                        std::to_string(instance_transforms.shape(2)) + " columns");
                }
                if (instance_transforms.shape(0) == 0) {
                    throw py::value_error("instance_transforms array cannot be empty");
                }
                
                // Validate particle_type_ids array [N]
                if (particle_type_ids.ndim() != 1) {
                    throw py::value_error("particle_type_ids must be a 1D array, got " + 
                                        std::to_string(particle_type_ids.ndim()) + " dimensions");
                }
                if (particle_type_ids.shape(0) != instance_transforms.shape(0)) {
                    throw py::value_error("particle_type_ids length (" + 
                                        std::to_string(particle_type_ids.shape(0)) + 
                                        ") must match number of instances (" + 
                                        std::to_string(instance_transforms.shape(0)) + ")");
                }
                
                // Validate material_ids array [N]
                if (material_ids.ndim() != 1) {
                    throw py::value_error("material_ids must be a 1D array, got " + 
                                        std::to_string(material_ids.ndim()) + " dimensions");
                }
                if (material_ids.shape(0) != instance_transforms.shape(0)) {
                    throw py::value_error("material_ids length (" + 
                                        std::to_string(material_ids.shape(0)) + 
                                        ") must match number of instances (" + 
                                        std::to_string(instance_transforms.shape(0)) + ")");
                }
                
                // Convert numpy [N,3,4] array to vector<float>
                auto transforms_array = instance_transforms.unchecked<3>();
                std::vector<float> transform_vec;
                transform_vec.reserve(transforms_array.shape(0) * 12); // 3,4 = 12 floats per transform
                for (py::ssize_t i = 0; i < transforms_array.shape(0); ++i) {
                    for (py::ssize_t j = 0; j < 3; ++j) {
                        for (py::ssize_t k = 0; k < 4; ++k) {
                            transform_vec.push_back(transforms_array(i, j, k));
                        }
                    }
                }
                
                // Convert numpy [N] arrays to vector<unsigned int>
                auto type_ids_array = particle_type_ids.unchecked<1>();
                std::vector<unsigned int> type_ids_vec;
                type_ids_vec.reserve(type_ids_array.shape(0));
                for (py::ssize_t i = 0; i < type_ids_array.shape(0); ++i) {
                    type_ids_vec.push_back(type_ids_array(i));
                }

                auto material_ids_array = material_ids.unchecked<1>();
                std::vector<unsigned int> material_ids_vec;
                material_ids_vec.reserve(material_ids_array.shape(0));
                for (py::ssize_t i = 0; i < material_ids_array.shape(0); ++i) {
                    material_ids_vec.push_back(material_ids_array(i));
                }
                
                return std::make_shared<InstanceGeometry>(std::move(sub_geometries),
                                                          std::move(transform_vec),
                                                          std::move(type_ids_vec),
                                                          std::move(material_ids_vec)); }),
            py::arg("sub_geometries"),
            py::arg("instance_transforms"),
            py::arg("particle_type_ids"),
            py::arg("material_ids"),
            doc::InstanceGeometry_init);

    py::class_<IRayGenerator, std::shared_ptr<IRayGenerator>>(m, "IRayGenerator", doc::IRayGenerator);

    py::class_<ParallelRayGenerator, IRayGenerator, std::shared_ptr<ParallelRayGenerator>>(m, "ParallelRayGenerator", doc::ParallelRayGenerator)
        .def(py::init([](int number_of_rays, float3 origin, float3 direction, float offset_radius, std::optional<float3> disk_normal)
                      {
                          ParallelRayGenerator generator(number_of_rays, origin, direction, offset_radius);
                          if (disk_normal)
                          {
                              generator.setDiskNormal(*disk_normal);
                          }
                          return generator; }),
             py::arg("number_of_rays"), py::arg("origin"), py::arg("direction"), py::arg("offset_radius"),
             py::arg("disk_normal") = py::none(), doc::ParallelRayGenerator_init)
        .def_property("origin", &ParallelRayGenerator::getOrigin, &ParallelRayGenerator::setOrigin, "Centre of the source disk.")
        .def_property("direction", &ParallelRayGenerator::getDirection, &ParallelRayGenerator::setDirection,
                      "Unit direction of the beam; normalised when set.")
        .def_property("disk_normal", &ParallelRayGenerator::getDiskNormal, [](ParallelRayGenerator &self, std::optional<float3> disk_normal)
                      {
                          if (disk_normal)
                          {
                              self.setDiskNormal(*disk_normal);
                          }
                          else
                          {
                              self.resetDiskNormal();
                          } },
                      "Unit normal of the plane of the source disk; equal to direction for a disk perpendicular to the beam. "
                      "Setting None makes the disk perpendicular to the beam again.")
        .def_property("offset_radius", &ParallelRayGenerator::getOffsetRadius, &ParallelRayGenerator::setOffsetRadius,
                      "Radius of the source disk in geometry units.");

    py::class_<IsotropicRayGenerator, IRayGenerator, std::shared_ptr<IsotropicRayGenerator>>(m, "IsotropicRayGenerator", doc::IsotropicRayGenerator)
        .def(py::init<int, float3, float, float>(),
             py::arg("number_of_rays"), py::arg("center"), py::arg("source_radius"), py::arg("offset_radius"),
             doc::IsotropicRayGenerator_init)
        .def_property("center", &IsotropicRayGenerator::getCenter, &IsotropicRayGenerator::setCenter, "Centre of the sphere the rays start on.")
        .def_property("source_radius", &IsotropicRayGenerator::getSourceRadius, &IsotropicRayGenerator::setSourceRadius,
                      "Radius of the sphere the rays start on.")
        .def_property("offset_radius", &IsotropicRayGenerator::getOffsetRadius, &IsotropicRayGenerator::setOffsetRadius,
                      "Radius in geometry units of the tangential shift of the ray origins.");

    py::class_<DiffuseRayGenerator, IRayGenerator, std::shared_ptr<DiffuseRayGenerator>>(m, "DiffuseRayGenerator", doc::DiffuseRayGenerator)
        .def(py::init<int, float3, float3, float>(),
             py::arg("number_of_rays"), py::arg("origin"), py::arg("direction"), py::arg("offset_radius"),
             doc::DiffuseRayGenerator_init)
        .def_property("origin", &DiffuseRayGenerator::getOrigin, &DiffuseRayGenerator::setOrigin, "Centre of the source disk.")
        .def_property("direction", &DiffuseRayGenerator::getDirection, &DiffuseRayGenerator::setDirection,
                      "Unit axis of the hemisphere the rays are emitted into; normalised when set.")
        .def_property("offset_radius", &DiffuseRayGenerator::getOffsetRadius, &DiffuseRayGenerator::setOffsetRadius,
                      "Radius of the source disk in geometry units.");

    py::class_<CameraRayGenerator, IRayGenerator, std::shared_ptr<CameraRayGenerator>>(m, "CameraRayGenerator", doc::CameraRayGenerator)
        .def(py::init<>())
        .def_property("image_width", &CameraRayGenerator::getImageWidth, &CameraRayGenerator::setImageWidth, "Image width in pixels.")
        .def_property("image_height", &CameraRayGenerator::getImageHeight, &CameraRayGenerator::setImageHeight, "Image height in pixels.")
        .def_property("aspect_ratio", &CameraRayGenerator::getAspectRatio, &CameraRayGenerator::setAspectRatio, "Aspect ratio width / height.")
        .def_property("samples_per_pixel", &CameraRayGenerator::getSamplesPerPixel, &CameraRayGenerator::setSamplesPerPixel,
                      "Number of rays per pixel.")
        .def_property("vertical_fov", &CameraRayGenerator::getVerticalFov, &CameraRayGenerator::setVerticalFov,
                      "Vertical field of view in degrees.")
        .def_property("look_from", &CameraRayGenerator::getLookFrom, &CameraRayGenerator::setLookFrom, "Position of the camera.")
        .def_property("look_at", &CameraRayGenerator::getLookAt, &CameraRayGenerator::setLookAt, "Point the camera is aimed at.")
        .def_property("vertical_up", &CameraRayGenerator::getVerticalUp, &CameraRayGenerator::setVerticalUp,
                      "Up direction of the camera, which sets its roll.")
        .def_property("defocus_angle", &CameraRayGenerator::getDefocusAngle, &CameraRayGenerator::setDefocusAngle,
                      "Opening angle of the cone of rays through each pixel in degrees, for depth of field; 0 disables it.")
        .def_property("focus_distance", &CameraRayGenerator::getFocusDistance, &CameraRayGenerator::setFocusDistance,
                      "Distance from the camera to the plane in focus.");

    py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation", doc::Simulation)
        .def(py::init([](int gpuId, int optixLoggingLevel, bool enableValidationMode, std::optional<BackendType> backend, uint32_t cpuThreads)
                      {
                          const std::vector<BackendType> available = availableBackends();
                          if (available.empty())
                          {
                              throw std::runtime_error("This build of photontracer contains no raytracing backend");
                          }
                          BackendOptions options;
                          options.gpuId = gpuId;
                          options.optixLoggingLevel = optixLoggingLevel;
                          options.enableValidationMode = enableValidationMode;
                          options.cpuThreads = cpuThreads;
                          return std::make_unique<Simulation>(makeBackend(backend.value_or(available.front()), options)); }),
             py::arg("gpu_id") = 0, py::arg("optix_logging_level") = 1, py::arg("enable_validation_mode") = false,
             py::arg("backend") = py::none(), py::arg("cpu_threads") = 0, doc::Simulation_init)
        .def("run", [](Simulation &sim)
             {
                 if (verboseOutput())
                 {
                     // flushed, so that Jupyter notebooks show that the simulation is running
                     py::print("Starting simulation", py::arg("flush") = true);
                 }
                 sim.run(); },
             doc::Simulation_run)
        .def("free_device_memory", &Simulation::freeDeviceMemory, doc::Simulation_free_device_memory)
        .def("calculate_volume_fraction", &Simulation::calculateVolumeFraction,
             py::arg("box_min"),
             py::arg("box_max"),
             py::arg("number_of_samples"),
             doc::Simulation_calculate_volume_fraction)
        .def("get_output_buffer", [](Simulation &sim, OutputType type) -> py::array
             {
            auto *deviceBuffers = sim.getDeviceOutputBuffers();
            if (!deviceBuffers)
            {
                throw std::runtime_error("Output buffers not initialized. Call run() first.");
            }

            auto rayOutput = sim.getRayOutput();
            if (!rayOutput)
            {
                throw std::runtime_error("Outputs not configured. Call configure_outputs() first.");
            }

            const auto &descriptor = rayOutput->getDescriptor(type);
            const BufferLayout &layout = deviceBuffers->getBufferLayout(type);
            if (!layout.devicePtr)
            {
                throw std::runtime_error("Output buffer for the specified type is not available.");
            }

            const uint3 shape = layout.shape;

            auto baseDimensions = [&]() -> std::vector<py::ssize_t>
            {
                std::vector<py::ssize_t> dims;
                auto pushDim = [&](unsigned int value, bool force)
                {
                    if (value > 1 || force)
                    {
                        dims.push_back(static_cast<py::ssize_t>(value));
                    }
                };
                pushDim(shape.z, false);
                pushDim(shape.y, false);
                pushDim(shape.x, dims.empty());
                if (dims.empty())
                {
                    dims.push_back(1);
                }
                return dims;
            };

            auto toShapeContainer = [](const std::vector<py::ssize_t> &dims) -> py::array::ShapeContainer
            {
                return py::array::ShapeContainer(dims.begin(), dims.end());
            };

            auto shapeFor = [&](py::ssize_t components) -> py::array::ShapeContainer
            {
                auto dims = baseDimensions();
                if (components > 1)
                {
                    dims.push_back(components);
                }
                return toShapeContainer(dims);
            };

            auto copyFromDevice = [&](void *hostPtr)
            {
                try
                {
                    deviceBuffers->copyToHost(type, hostPtr);
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error(std::string("Failed to copy output buffer '") + descriptor.name +
                                             "' from device memory: " + e.what());
                }
            };

            switch (descriptor.elementType)
            {
            case BufferDescriptor::ElementType::Float3:
            {
                py::array_t<float> result(shapeFor(3));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Float4:
            {
                py::array_t<float> result(shapeFor(4));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Float:
            {
                py::array_t<float> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Double:
            {
                py::array_t<double> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Int32:
            {
                py::array_t<int32_t> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::UInt32:
            {
                py::array_t<uint32_t> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            default:
                throw std::runtime_error("Unsupported buffer element type.");
            } }, py::arg("output_type"), doc::Simulation_get_output_buffer)

        .def_property("trace_rays", &Simulation::getTraceRays, &Simulation::setTraceRays, doc::Simulation_trace_rays)
        .def_property("max_trace_steps", &Simulation::getMaxTraceSteps, &Simulation::setMaxTraceSteps,
                      "Steps recorded per traced ray (default 1000); later steps are not stored.")
        .def("get_trace", [](Simulation &sim) -> py::array
             {
            const std::vector<TraceRecord> steps = sim.getTrace();
            py::list fields;
            fields.append(py::make_tuple("ray", "<u4"));
            fields.append(py::make_tuple("step", "<u4"));
            fields.append(py::make_tuple("event", "<u4"));
            fields.append(py::make_tuple("medium_in", "<u4"));
            fields.append(py::make_tuple("medium_out", "<u4"));
            fields.append(py::make_tuple("origin_in", "<f4", 3));
            fields.append(py::make_tuple("direction_in", "<f4", 3));
            fields.append(py::make_tuple("origin_out", "<f4", 3));
            fields.append(py::make_tuple("direction_out", "<f4", 3));
            fields.append(py::make_tuple("stokes", "<f4", 4));
            fields.append(py::make_tuple("optical_path_length", "<f4"));
            py::dtype dtype = py::dtype::from_args(fields);
            if (static_cast<size_t>(dtype.itemsize()) != sizeof(TraceRecord))
            {
                throw std::runtime_error("The trace record layout does not match its numpy type");
            }
            py::array result(dtype, static_cast<py::ssize_t>(steps.size()));
            if (!steps.empty())
            {
                std::memcpy(result.mutable_data(), steps.data(), steps.size() * sizeof(TraceRecord));
            }
            return result; },
             doc::Simulation_get_trace)
        .def_property("geometry", &Simulation::getGeometry, &Simulation::setGeometry, doc::Simulation_geometry)
        .def_property("materials", &Simulation::getMaterials, &Simulation::setMaterials, doc::Simulation_materials)
        .def_property("wavelength_um", &Simulation::getWavelengthUm, &Simulation::setWavelengthUm, doc::Simulation_wavelength_um)
        .def_property("length_unit", &Simulation::getLengthUnit, &Simulation::setLengthUnit,
                      "Unit of the geometry coordinates, default MICRO_METER.")
        .def_property("stokes_vector", &Simulation::getStokesVector, &Simulation::setStokesVector, doc::Simulation_stokes_vector)
        .def_property("q_minus_axis_seed", &Simulation::getQMinusAxisSeed, &Simulation::setQMinusAxisSeed, doc::Simulation_q_minus_axis_seed)
        .def_property("max_scattering_count", &Simulation::getMaxScatteringCount, &Simulation::setMaxScatteringCount,
                      doc::Simulation_max_scattering_count)
        .def_property("max_nested_geometry_levels", &Simulation::getMaxNestedGeometryLevels, &Simulation::setMaxNestedGeometryLevels,
                      doc::Simulation_max_nested_geometry_levels)
        .def_property_readonly("max_sub_geometries", &Simulation::getMaxSubGeometries,
                               "Maximum number of sub-geometries of an InstanceGeometry on this backend.")
        .def_property_readonly("max_mesh_triangles", &Simulation::getMaxMeshTriangles,
                               "Maximum number of triangles of a MeshGeometry on this backend.")
        .def_property("seed", &Simulation::getInitSeed, &Simulation::setInitSeed, doc::Simulation_seed)
        .def_property("random_number_generator", &Simulation::getRandomNumberGenerator, &Simulation::setRandomNumberGenerator,
                      doc::Simulation_random_number_generator)
        .def_property("use_complex_fresnel", &Simulation::getUseComplexFresnel, &Simulation::setUseComplexFresnel,
                      doc::Simulation_use_complex_fresnel)
        .def_property("direction_healpix_nside", &Simulation::getDirectionHealpixNside, &Simulation::setDirectionHealpixNside,
                      doc::Simulation_direction_healpix_nside)
        .def_property("ray_generator", &Simulation::getRayGenerator, &Simulation::setRayGenerator,
                      "The source of the rays: a ParallelRayGenerator, IsotropicRayGenerator, DiffuseRayGenerator or CameraRayGenerator.")
        .def_property("outputs", [](const Simulation &sim) -> std::vector<OutputType>
                      {
            if (!sim.getRayOutput()) {
                throw std::runtime_error("Outputs not configured. Call configure_outputs() first.");
            }

            const auto &descriptors = sim.getRayOutput()->getDescriptors();
            std::vector<OutputType> outputTypes;
            for (const auto &desc : descriptors) {
                if (desc.enabled) {
                    outputTypes.push_back(desc.type);
                }
            }
            return outputTypes; }, [](Simulation &sim, const std::vector<OutputType> &outputTypes)
                      {
                          auto rayTracingOutput = std::make_shared<RayTracingOutput>();
                          for (const auto &type : outputTypes)
                          {
                              rayTracingOutput->enableOutput(type, true); // Enable the buffer
                          }
                          sim.setRayOutput(rayTracingOutput); // Set the configured output buffers in the simulation
                      },
                      doc::Simulation_outputs);

    m.def("is_cuda_available", &isCudaAvailable, "Whether at least one CUDA device is available.");
    m.def("set_verbose", [](bool verbose) { verboseOutput() = verbose; },
          py::arg("verbose"), "Print progress messages (device in use, compile and launch times). Off by default.");
    m.def("is_verbose", []() { return verboseOutput().load(); }, "Whether progress messages are printed.");
    m.def("available_backends", &availableBackends, "The raytracing backends of this build, the default one first.");
}