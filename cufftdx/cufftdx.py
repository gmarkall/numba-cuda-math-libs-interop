import numpy as np
from instant import inline

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

code_template = """\
void* make_workspace() {
  {fft_decl}
  return nullptr;
}
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

    def make_workspace(self):
        c_precision = {np.float32: 'float'}[self.precision]
        decl = fft_decl.format(
            size=self.size,
            precision=c_precision,
            fft_type=self.fft_type,
            direction=self.direction,
            elements_per_thread=self.elements_per_thread,
            ffts_per_block=self.ffts_per_block,
            sm=self.sm,
            operator_expression=self.operator_expression
        )

        code = code_template.format(fft_decl

        print(decl)
        # FIXME: Need to runtime compile the make_workspace function and
        # invoke (or provide a compiled library implementation of all
        # variants?)
