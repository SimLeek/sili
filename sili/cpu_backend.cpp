//CPU ONLY CODE
// All the GPU code is in GLSL and should run entirely on the GPU,
// so there is no optimization that can be done in C++

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lib/sparse_linear.cpp"
#include "lib/sparse_conv2d.cpp"
#include "lib/sparse_conv2d_over_on.cpp"
#include "lib/utils.cpp"

namespace py = pybind11;

PYBIND11_MODULE(backend, m)
{
    // sidlso = sparse input, dense layer, sparse output
    // there will be a lot of these and I want them to be auto-selected based on input type
    // for now though, sidlso is the only one we care about.

    //linear
    m.def("linear_sidlso_olist", &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward_olist", &sparse_linear_vectorized_backward_wrapper);
    m.def("linear_sidlso_scipy", &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward_scipy", &sparse_linear_vectorized_backward_wrapper);

    //conv2d
    m.def("linear_sidlso", &sparse_linear_vectorized_forward_wrapper);
    m.def("linear_sidlso_backward", &sparse_linear_vectorized_backward_wrapper);
}