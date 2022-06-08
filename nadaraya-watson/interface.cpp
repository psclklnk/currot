#include <nadaraya_watson.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace na = nadaraya_watson;

PYBIND11_MODULE(nadaraya_watson, m) {

    py::class_<na::NadarayaWatson<double>>(m, "NadarayaWatson")
            .def(py::init<py::buffer, py::buffer, uint32_t>(), py::arg("sx"), py::arg("sy"),
                 py::arg("n_threads") = 1)
            .def("predict", &na::NadarayaWatson<double>::predict, py::arg("query"),
                 py::arg("bandwidth"), py::arg("n_max") = 200, py::arg("radius_scale") = 3.,
                 py::return_value_policy::move);

}
