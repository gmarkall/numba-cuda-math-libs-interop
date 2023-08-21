import cppimport
import numpy as np
import os
import sys
import tempfile
from functools import cached_property

fft_decl = """\
using FFT = decltype(Size<{size}>()
          + Precision<{precision}>()
          + Type<fft_type::{fft_type}>()
          + Direction<fft_direction::{direction}>()
          + ElementsPerThread<{elements_per_thread}>()
          + FFTsPerBlock<{ffts_per_block}>()
          + SM<{sm}>()
          + {operator_expression}());
"""

storage_size_template = """\
unsigned get_storage_size() {{
{fft_decl}
return FFT::storage_size;
}}
"""


class FFT:
    def __init__(self, *, size, precision, fft_type, direction,
                 elements_per_thread, ffts_per_block, sm, operator_expression):
        self.size = size
        self.precision = precision
        self.fft_type = fft_type
        self.direction = direction
        self.elements_per_thread = elements_per_thread
        self.ffts_per_block = ffts_per_block
        self.sm = sm
        self.operator_expression = operator_expression

        if fft_type == 'c2c':
            self.value_type = np.complex64

    @cached_property
    def fft_decl(self):
        c_precision = {np.float32: 'float'}[self.precision]
        return fft_decl.format(
            size=self.size,
            precision=c_precision,
            fft_type=self.fft_type,
            direction=self.direction,
            elements_per_thread=self.elements_per_thread,
            ffts_per_block=self.ffts_per_block,
            sm=self.sm,
            operator_expression=self.operator_expression
        )

    @cached_property
    def storage_size(self):
        code = storage_size_template.format(fft_decl=self.fft_decl)
        print(code)
        # return instant.inline(code)()


test_code = """\
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) {
    return x * x;
}

PYBIND11_MODULE(somecode, m) {
    m.def("square", &square);
}
/*
<%
setup_pybind11(cfg)
%>
*/
"""


def cpp_jit_module(code):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'somecode.cpp')
        with open(filename, 'w') as f:
            f.write(code)
        sys.path.append(tmpdir)
        mod = cppimport.imp_from_filepath(filename, fullname='somecode')
        sys.path.remove(tmpdir)
    return mod


if __name__ == '__main__':
    breakpoint()
    mod = cpp_jit_module(test_code)
    print(mod.square(9))
