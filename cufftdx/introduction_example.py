from numba import cuda
import cufftdx
import numpy as np


def get_block_fft_kernel(fft):
    complex_type = fft.value_type
    storage_size = fft.storage_size
    ffts_per_block = fft.ffts_per_block
    stride = fft.stride
    elements_per_thread = fft.elements_per_thread
    fft_execute = fft.execute
    fft_size = fft.size

    @cuda.jit
    def block_fft_kernel(data):
        # Local array for thread
        thread_data = cuda.local.array(storage_size, dtype=complex_type)

        # Local batch id of this FFI in CUDA block, in range
        # [0; ffts_per_block]
        local_fft_id = cuda.threadIdx.y

        # Global batch id of this FFT in CUDA grid is equal to number of
        # batches per CUDA block (ffts_per_block) times CUDA block id, plus
        # local batch id.
        global_fft_id = (cuda.blockIdx.x * ffts_per_block) + local_fft_id

        # Load data from global memory to registers
        offset = fft_size * global_fft_id
        index = offset + cuda.threadIdx.x

        for i in range(elements_per_thread):
            if (i * stride + cuda.threadIdx.x) < fft_size:
                thread_data[i] = data[index]
                index += stride

        # Execute FFT
        shared_memory = cuda.shared.array(0, dtype=complex_type)
        fft_execute(thread_data, shared_memory)

        # Save results
        index = offset + cuda.threadIdx.x
        for i in range(elements_per_thread):
            if (i * stride + cuda.threadIdx.x) < fft_size:
                data[index] = thread_data[i]
                index += stride

    return block_fft_kernel


def get_current_sm():
    """Returns the SM for the current device"""
    cc = cuda.get_current_device().compute_capability
    return cc[0] * 100 + cc[1] * 10


# In this example a one-dimensional complex-to-complex transform is performed
# by a CUDA block.
#
# One block is run, it calculates two 128-point C2C float precision FFTs.
# Data is generated on host, copied to device buffer, and then results are
# copied back to host.
def introduction_example():
    # FFT definition
    #
    # size, precision, type, and direction are defined by kwargs.
    # 'block' specifies that the FFT will be executed at the block level.
    # Shared memory is required for co-operation between threads.
    #
    # Additionally:
    # * ffts_per_block defines how many FFTs (batches) are executed in a single
    #   CUDA block,
    # * elements_per_thread defines how FFT calculations are mapped into a CUDA
    #   block, i.e. how many thread are required, and
    # * sm defines the targeted CUDA architecture.
    fft = cufftdx.FFT(size=128, precision=np.float32, fft_type='c2c',
                      direction='forward', elements_per_thread=8,
                      ffts_per_block=2, sm=get_current_sm(),
                      operator_expression='block')

    # Allocate managed memory for input/output
    size = fft.ffts_per_block * fft.size
    data = cuda.managed_array(size, dtype=fft.value_type)

    # Generate data
    # XXX: This generates twice as much data as the example, but is the example
    # only generating half as much as needed if there are 2 FFTs per block?
    for i in range(size):
        data[i] = i - (i * 1j)

    # Generate workspace omitted as not required for this example

    print("Input [1st FFT]:")
    print(data)

    # Increase max shared memory if needed
    # FIXME: Implement

    # Invoke kernel with fft.block_dim threads in CUDA block
    block_fft_kernel = get_block_fft_kernel(fft)
    block_fft_kernel[1, fft.block_dim, 0, fft.shared_memory_size](data)
    cuda.synchronize()

    print("Output [1st FFT]:")
    print(data)


if __name__ == '__main__':
    introduction_example()
