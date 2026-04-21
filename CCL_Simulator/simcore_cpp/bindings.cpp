#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sim.hpp"

namespace py = pybind11;

PYBIND11_MODULE(simcore_cpp, m) {
    m.doc() = "C++ backend for simcore";

    using namespace simcore;

    py::class_<PolicyEntry>(m, "PolicyEntry")
        .def(py::init<>())
        .def_readwrite("chunk_id", &PolicyEntry::chunk_id)
        .def_readwrite("src", &PolicyEntry::src)
        .def_readwrite("dst", &PolicyEntry::dst)
        .def_readwrite("qpid", &PolicyEntry::qpid)
        .def_readwrite("rate", &PolicyEntry::rate)
        .def_readwrite("use_max_rate", &PolicyEntry::use_max_rate)
        .def_readwrite("chunk_size_bytes", &PolicyEntry::chunk_size_bytes)
        .def_readwrite("path", &PolicyEntry::path)
        .def_readwrite("time", &PolicyEntry::time)
        .def_readwrite("dependency", &PolicyEntry::dependency);

    py::class_<Sim>(m, "Sim")
        .def(py::init<int, int>(), py::arg("packet_size_bytes") = 1500, py::arg("header_size_bytes") = 0)
        .def("add_gpu", &Sim::add_gpu, py::arg("node_id"), py::arg("num_qps"), py::arg("quantum_packets"), py::arg("tx_proc_delay"), py::arg("gpu_store_delay"))
        .def("add_switch", &Sim::add_switch, py::arg("node_id"), py::arg("num_qps"), py::arg("quantum_packets"), py::arg("tx_proc_delay"), py::arg("sw_proc_delay"))
        .def("add_link", &Sim::add_link, py::arg("u"), py::arg("v"), py::arg("link_rate_bps"), py::arg("prop_delay"))
        .def("load_policy", &Sim::load_policy)
        .def("start", &Sim::start)
        .def("run", &Sim::run, py::arg("until") = -1.0)
        .def_property_readonly("tx_complete_time", [](Sim& sim) {
            std::map<std::tuple<std::string, std::string, std::string>, double> res;
            for (auto const& [key, val] : sim.tx_complete_time) {
                res[std::make_tuple(std::get<0>(key), std::get<1>(key), std::get<2>(key))] = val;
            }
            return res;
        });
}
