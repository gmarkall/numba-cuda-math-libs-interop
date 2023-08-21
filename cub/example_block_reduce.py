# Copyright (c) 2011, Duane Merrill.  All rights reserved.
# Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Simple demonstration of cub::BlockReduce

from numba import cuda, types
from numba.core.extending import intrinsic, overload
from numba.core.typing import signature
from numba.cuda import nvvmutils
import argparse
import numpy as np


def specialize_block_reduce(dtype, block_threads, algorithm):
    """
    Specialize BlockReduce type for our thread block

    :param dtype: Data type
    :param block_threads: Number of threads per block
    :param algorithm: Reduction algorithm to use
    """
    # XXX: Implement - cuda.declare_device(...), etc.


def specialize_load_direct_striped(dtype, block_threads):
    """
    Specialize LoadDirectStriped for our thread block and dtype

    :param dtype: Data type to load
    :param block_threads: Number of threads in a block
    """
    # XXX: Implement


# Numba doesn't seem to implement cycle count on device so here's a quick
# extension to do so

def clock64():
    pass


@intrinsic
def _clock64(typingctx):
    sig = signature(types.uint64)

    def codegen(context, builder, sig, args):
        return nvvmutils.call_sreg(builder, 'clock64')

    return sig, codegen


@overload(clock64, target='cuda')
def ol_clock64():
    def impl():
        return _clock64()
    return impl


# Kernels

def get_block_reduce_kernel(dtype, block_threads, items_per_thread, algorithm):
    block_reduce_type = specialize_block_reduce(dtype, block_threads,
                                                algorithm)
    block_reduce_sum = block_reduce_type.sum

    temp_storage_dtype = block_reduce_type.temp_storage_dtype
    temp_storage_size = block_reduce_type.temp_storage_size

    load_direct_striped = specialize_load_direct_striped(dtype, block_threads)

    @cuda.jit
    def block_reduce_kernel(d_in, d_out, d_elapsed):
        """
        Simple kernel for performing a block-wide reduction.

        :param d_in: Tile of input
        :param d_out: Tile of output
        :param d_elapsed: Elapsed cycle count of block reduction
        """

        tid = cuda.grid(1)

        # Shared memory
        temp_storage = cuda.shared.array(temp_storage_size, temp_storage_dtype)

        # Per-thread tile data
        data = cuda.local.array(items_per_thread, dtype=dtype)
        load_direct_striped(tid, d_in, data)

        # Start cycle timer
        start = clock64()

        # Compute sum
        aggregate = block_reduce_sum(data, temp_storage)

        # Stop cycle timer
        stop = clock64()

        # Store aggregate and elapsed clocks
        if (tid == 0):
            d_elapsed[0] = stop - start
            d_out[0] = aggregate


# ---------------------------------------------------------------------
# Host utilities
# ---------------------------------------------------------------------

class GpuTimer:
    def __init__(self):
        self.start = cuda.event()
        self.stop = cuda.event()

    def start(self):
        self.start.record()

    def stop(self):
        self.stop.record()

    def get_elapsed_millis(self):
        self.stop.synchronize()
        return cuda.event_elapsed_time(self.start, self.stop)


# Initialize reduction problem (and solution).
# Returns the aggregate
def initialize(h_in, num_items):
    inclusive = 0

    for i in range(num_items):
        h_in[i] = i % 17
        inclusive += h_in[i]

    return inclusive


# Test thread block reduction
def test(args, block_threads, items_per_thread, algorithm):
    tile_size = block_threads * items_per_thread
    dtype = np.int32

    # Allocate host arrays
    h_in = np.zeros(tile_size, dtype=dtype)

    # Initialize problem and reference output on host
    h_aggregate = initialize(h_in, tile_size)

    # Initialize device arrays
    d_out = cuda.device_array(1, dtype=dtype)
    d_elapsed = cuda.device_array(1, dtype=np.uint64)

    # Display input problem data
    if args.verbose:
        print("Input data:")
        print(h_in)

    # Get a specialized instance of the kernel
    block_reduce_kernel = get_block_reduce_kernel(dtype, block_threads,
                                                  items_per_thread, algorithm)

    # Kernel props - note that "cooperative" is just an unfortunate naming
    # choice in the Numba API - this just returns the max active blocks
    max_sm_occupancy = \
        block_reduce_kernel.max_cooperative_grid_blocks(block_threads)

    # Copy problem to device
    d_in = cuda.to_device(h_in)

    print(f"BlockReduce Algorithm {algorithm} on {tile_size} items ("
          f"{args.timing_iterations} timing iterations, "
          f"{args.grid_size} blocks, "
          f"{block_threads} threads, "
          f"{items_per_thread} items per thread, "
          f"{max_sm_occupancy} SM occupancy):")

    # Run kernel

    block_reduce_kernel[args.grid_size, block_threads](d_in, d_out, d_elapsed)

    # Check total aggregate
    np.testing.assert_equal(h_aggregate, d_out[0])

    # Run this several times and average the performance results
    timer = GpuTimer()
    elapsed_millis = 0.0
    elapsed_clocks = 0

    for i in range(args.timing_iterations):
        # Copy problem to device
        d_in = cuda.to_device(h_in)

        timer.start()

        # Run kernel
        block_reduce_kernel[args.grid_size, block_threads](d_in, d_out,
                                                           d_elapsed)

        timer.stop()
        elapsed_millis += timer.get_elapsed_millis()

        # Copy clocks from device
        elapsed_clocks += d_elapsed[0]

    # Check for kernel errors and STDIO from the kernel, if any
    cuda.synchronize()

    # Display timing results
    avg_millis = elapsed_millis / args.timing_iterations
    avg_items_per_sec = (tile_size * args.grid_size) / avg_millis / 1000
    avg_clocks = elapsed_clocks / args.timing_iterations
    avg_clocks_per_item = avg_clocks / tile_size

    print("\tAverage BlockReduce::Sum clocks: %.3f" % avg_clocks)
    print("\tAverage BlockReduce::Sum clocks per item: %.3f" %
          avg_clocks_per_item)
    print("\tAverage kernel millis: %.4f" % avg_millis)
    print("\tAverage million items / sec: %.4f" % avg_items_per_sec)


def main(argv):
    # Initialize command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help="Verbose output",
                        action="store_true")
    parser.add_argument('-i', '--iterations', help="Timing iterations",
                        action="store", default=100, type=int)
    parser.add_argument('-g', '--grid-size', help="Grid size",
                        action="store", default=1, type=int)
    parser.add_argument('-d', '--device', help="CUDA device ID",
                        action="store", default=0, type=int)
    args = parser.parse_args(argv[1:])

    # Initialize device
    cuda.select_device(args.device)

    # Run tests
    test(args, 1024, 1, "BLOCK_REDUCE_RAKING")
    test(args, 512, 2, "BLOCK_REDUCE_RAKING")
    test(args, 256, 4, "BLOCK_REDUCE_RAKING")
    test(args, 128, 8, "BLOCK_REDUCE_RAKING")
    test(args, 64, 16, "BLOCK_REDUCE_RAKING")
    test(args, 32, 32, "BLOCK_REDUCE_RAKING")
    test(args, 16, 64, "BLOCK_REDUCE_RAKING")

    print("-------------")

    test(args, 1024, 1, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 512, 2, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 256, 4, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 128, 8, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 64, 16, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 32, 32, "BLOCK_REDUCE_WARP_REDUCTIONS")
    test(args, 16, 64, "BLOCK_REDUCE_WARP_REDUCTIONS")


if __name__ == '__main__':
    import sys
    main(sys.argv)
