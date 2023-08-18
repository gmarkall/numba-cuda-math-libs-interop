import numpy as np


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
            # FIXME: Need to runtime compile the make_workspace function and
            # invoke (or provide a compiled library implementation of all
            # variants?)
