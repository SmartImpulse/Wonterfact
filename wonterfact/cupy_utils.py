# ----------------------------------------------------------------------------
# Copyright 2019 Smart Impulse SAS
#
# This file is part of Wonterfact.
#
# Wonterfact is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Wonterfact is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wonterfact. If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

"""Methods written for cupy or cuda"""

# Python System imports
import sys
import time
from functools import lru_cache  # pylint: disable=E0611

# Third-party imports
from numba import cuda, float32, float64

# Relative imports
from .glob_var_manager import glob
from . import utils

if glob.float == glob.FLOAT32:
    numba_float = float32
elif glob.float == glob.FLOAT64:
    numba_float = float64


def cupy_cumsum_2d(arr_in, arr_out, max_threads=256):
    cp = utils.xp_utils.back("cupy")
    if arr_in.shape[-1] == 1:
        arr_out[...] = arr_in
        return
    batch_dim_y = utils.next_pow_of_two(min(arr_in.shape[1], 2 * max_threads))
    block_dim_y = batch_dim_y // 2
    block_dim_x = min(arr_in.shape[0], max_threads // block_dim_y)
    blocks_number_y = (arr_in.shape[1] + batch_dim_y - 1) // batch_dim_y
    blocks_number_x = (arr_in.shape[0] + block_dim_x - 1) // block_dim_x
    print(blocks_number_y)

    if blocks_number_y > 1:
        store = True
        aux = cp.zeros((blocks_number_x, blocks_number_y))
    else:
        store = False
        aux = cp.zeros(())

    inclusive_scan_2d(
        (
            blocks_number_x,
            blocks_number_y,
        ),
        (
            block_dim_x,
            block_dim_y,
        ),
        (
            arr_in,
            arr_out,
            arr_in.shape[0],
            arr_in.shape[1],
            batch_dim_y,
            aux,
            store,
            arr_in.strides[0] // 8,
            arr_in.strides[1] // 8,
            arr_out.strides[0] // 8,
            arr_out.strides[1] // 8,
        ),
        shared_mem=(block_dim_x * (batch_dim_y + 1)) * 8,
    )
    if blocks_number_y > 1:
        incr = cp.zeros((blocks_number_x, blocks_number_y))
        cupy_cumsum_2d(aux, incr, max_threads=max_threads)
        sum_inclusive_scan_2d(
            (
                blocks_number_x,
                blocks_number_y,
            ),
            (
                block_dim_x,
                batch_dim_y,
            ),
            (
                incr,
                arr_out,
                arr_in.shape[0],
                arr_in.shape[1],
                arr_out.strides[0] // 8,
                arr_out.strides[1] // 8,
            ),
        )


@lru_cache(maxsize=1024)
def find_cumsum_max_threads(size_arr, size_cumsum):
    cp = utils.xp_utils.back("cupy")
    arr = cp.random.rand(size_arr, size_cumsum)
    out = arr.copy()
    tac_list = []
    max_threads_list = [2 ** nn for nn in range(5, 10)]
    for max_threads in max_threads_list:
        tic = time.time()
        for __ in range(100):
            cupy_cumsum_2d(arr, out, max_threads=max_threads)
        device = cp.cuda.Device()
        device.synchronize()
        tac_list.append(time.time() - tic)
    __, max_threads = min(zip(tac_list, max_threads_list), key=lambda x: x[0])
    return max_threads


min_clip = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 x, float64 min_val",  # input params
    "float64 y",  # output params
    "y = ((x < min_val) ? min_val:x)",
    "min_clip",
)


max_clip = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 x, float64 max_val",  # input params
    "float64 y",  # output params
    "y = ((x > max_val) ? max_val:x)",
    "max_clip",
)

_xlogy = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 x, float64 y",  # input params
    "float64 z",  # output params
    "z = ((x != 0.0) ? x*log(y):0.0)",
    "_xlogy",
)


def xlogy(arr1, arr2, out=None):
    if out is None:
        return _xlogy(arr1, arr2)
    return _xlogy(arr1, arr2, out)


@cuda.jit()
def normalize_l1_l2_tensor_numba_core(tensor, n_iter, prior_rate):
    """
    Core function for l1/l2 normalization of LeafGammaNorm' tensor attribute
    transforms tensor inplace
    This code is inspired by [1] for the reduction part

    input:
     - tensor: input tensor to normalize (normalization is performed inplace)
     - n_iter: number of iteration for the fix-point algorithm
     - prior_beta_arr: a cupy array with values : [2 * prior_rate, 4 * prior_rate]

    [1] https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    """

    # shared array def (for output data computation and reduction)
    # shared_memo = cuda.shared.array(shape=100, dtype=numba_float[glob.float])  # pylint disable=E1102
    shared_memo = cuda.shared.array(
        shape=100, dtype=numba_float
    )  # pylint disable=E1102

    # get block and thread numbers
    dim_0, dim_1 = cuda.blockIdx.x, cuda.threadIdx.x
    block_dim = cuda.blockDim.x

    # leave if thread is out of bound
    if dim_0 >= tensor.shape[0] and dim_1 >= tensor.shape[1]:
        return

    # import input data
    out_val_1 = tensor[dim_0, dim_1]
    dim_1_bis = dim_1 + block_dim
    if dim_1_bis < tensor.shape[1]:
        out_val_2 = tensor[
            dim_0, dim_1_bis
        ]  # each thread deals with 2 values for speed purpose (see [1]: scenario 4)
    else:
        out_val_2 = 0.0
    in_val_1 = out_val_1
    in_val_2 = out_val_2

    for __ in range(n_iter):
        # First we load data
        shared_memo[dim_1] = out_val_1 ** 2 + out_val_2 ** 2

        # A synchronize loop to performe the sum in O(log(N))
        # at the end, the sum is the first component of the shared memory
        max_idx = (block_dim) >> 1  # bin swaping is equivalent to // 2
        jump_idx = (
            block_dim + 1
        ) >> 1  # we need some tricks to deals with odd dimensions
        while max_idx > 0:
            cuda.syncthreads()
            if dim_1 < max_idx:
                shared_memo[dim_1] += shared_memo[dim_1 + jump_idx]
            max_idx = jump_idx >> 1
            jump_idx = (jump_idx + 1) >> 1

        if dim_1 == 0:
            shared_memo[1] = shared_memo[0] ** 0.5
            shared_memo[2] = 4 * prior_rate * shared_memo[1]
        # here, norm22 is in shared_mem0[0], norm2 in shared_memo[1] and shared_memo[2] contains some temp computation

        cuda.syncthreads()
        # norm22 = shared_memo[0]
        # norm2 = shared_memo[1]

        delta1 = shared_memo[0] + shared_memo[2] * in_val_1
        delta2 = shared_memo[0] + shared_memo[2] * in_val_2
        out_val_1 = (-shared_memo[1] + delta1 ** 0.5) / (2 * prior_rate)
        out_val_2 = (-shared_memo[1] + delta2 ** 0.5) / (2 * prior_rate)
        cuda.syncthreads()

    tensor[dim_0, dim_1] = out_val_1
    if dim_1_bis < tensor.shape[1]:
        tensor[
            dim_0, dim_1_bis
        ] = out_val_2  # each thread deals with 2 values for speed purpose (see [1]: scenario 4)


_set_bezier_point = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 val1, float64 val2, float64 val3, float64 p",
    "float64 z",
    """
        double omp = 1 - p;
        z = omp * omp * val1 + 2 * omp * p * val2 + p * p * val3;
    """,
    "_set_bezier_point",
)


multiply_and_sum = utils.xp_utils.back("cupy").ReductionKernel(
    "float64 x, float64 y",  # input params
    "float64 z",  # output params
    "x * y",  # map
    "a + b",  # reduce
    "z = a",  # post-reduction map
    "0",  # identity value
    "multiply_and_sum",  # kernel name
)


_exp_digamma_c_code = """
double coef_list[] = {
    0.041666666666666664,
    -0.006423611111111111,
    0.003552482914462081,
    -0.0039535574489730305,
};
double tmp2, temp;

double input_plus_n = input_val;
if (input_plus_n == 0){
    output_val = 0.0;
}
else {
    tmp2 = 0.0;
    while (input_plus_n < 10.0){
        tmp2 -= 1.0 / input_plus_n;
        input_plus_n += 1.0;
    }
    input_plus_n -= 0.5;
    output_val = input_plus_n;
    temp = input_plus_n;
    input_plus_n *= input_plus_n;
    for (int idx = 0; idx < 4; ++idx){
        temp /= input_plus_n;
        output_val += coef_list[idx] * temp;
    }
    if (tmp2 != 0.0){
        output_val *= exp(tmp2);
    }
}
"""


# This method corresponds to utils._exp_digamma method
exp_digamma = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 input_val",  # input params
    "float64 output_val",  # output params
    _exp_digamma_c_code,
    "exp_digamma",
)


_hyp0f1ln_c_code = """
double temp_r, temp_a, output_temp_val, temp_v_val;
bool keep_going;

temp_r = 1.0;
temp_a = 1.0;
temp_v_val = v_val;
keep_going = true;
output_temp_val = 1.0;
output_val = 0.0;

while (keep_going){
    if (output_temp_val > 1e300){
        output_val += log(output_temp_val);
        temp_a /= output_temp_val;
        output_temp_val = 1.0;
    }
    temp_a *= z_val / (temp_r * temp_v_val);
    output_temp_val += temp_a;
    temp_r += 1.0;
    temp_v_val += 1.0;
    keep_going = temp_a > (tol * output_temp_val);
}
output_val += log(output_temp_val);
"""

_hyp0f1ln = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 v_val, float64 z_val, float64 tol",  # input params
    "float64 output_val",  # output params
    _hyp0f1ln_c_code,
    "_hyp0f1ln",
)


# This method corresponds to utils.hyp0f1ln method
def hyp0f1ln(v_arr, z_arr, tol=1e-16):
    return _hyp0f1ln(v_arr, z_arr, tol)


_bessel_ratio_c_code = """
double z_arr2, temp_pr, temp_v0, temp_v, temp_u, temp_w, temp_p, tol2, temp_t;
bool keep_going;

z_arr2 = z_arr / 2.0;
output_val = 1.0;
temp_pr = 1.0;
temp_v0 = v_arr + z_arr2 + 1.0;
temp_v = v_arr + z_arr + 1.5;
temp_u = (v_arr + 1 + z_arr) * temp_v;
temp_w = z_arr2 * (v_arr + 1.5);
temp_p = temp_w / (temp_v0 * temp_v - temp_w);
tol2 = tol * (1 + temp_p);
temp_pr *= temp_p;
output_val += temp_pr;
keep_going = true;
while (keep_going){
    temp_u += temp_v;
    temp_v += 0.5;
    temp_w += z_arr2;
    temp_t = temp_w * (1 + temp_p);
    temp_p = temp_t / (temp_u - temp_t);
    temp_pr *= temp_p;
    output_val += temp_pr;
    keep_going = temp_pr > tol2;
}
output_val *= z_arr * z_arr / (z_arr+ 2 * v_arr + 2);
"""

_bessel_ratio = utils.xp_utils.back("cupy").ElementwiseKernel(
    "float64 v_arr, float64 z_arr, float64 tol",  # input params
    "float64 output_val",  # output params
    _bessel_ratio_c_code,
    "_bessel_ratio",
)


# This method corresponds to utils.bessel_ratio method
def bessel_ratio(v_arr, z_arr, tol=1e-16):
    return _bessel_ratio(v_arr, z_arr, tol)


inclusive_scan_2d = utils.xp_utils.back("cupy").RawKernel(
    r"""
// Inclusive scan on CUDA.
extern "C" __global__
void inclusive_scan_2d(
    double *d_array,
    double *d_result,
    int dimRow,
    int dimCol,
    int batchDim,  // must be power of two
    double *d_aux,
    bool store,
    int strideRowIn,
    int strideColIn,
    int strideRowOut,
    int strideColOut
) {
    extern __shared__ double temp[];  // dim is blockDim.x * (batchDim + 1)

    // index of input and result arrays
    int realIndexRow = blockDim.x * blockIdx.x + threadIdx.x;
    int realIndexCol = 2 * (blockDim.y * blockIdx.y + threadIdx.y);  // 2 * blockDim.y must be == batchDim
    // int realIndexFlat = realIndexRow * dimCol + realIndexCol;
    int realIndexFlatIn = realIndexRow * strideRowIn + realIndexCol * strideColIn;
    int realIndexFlatOut = realIndexRow * strideRowOut + realIndexCol * strideColOut;

    int threadIndexRow = threadIdx.x;
    int threadIndexCol = threadIdx.y;

    // index of temp arr
    int indexRow = threadIndexRow;
    int indexCol = 2 * threadIndexCol;
    int indexStartFlat = indexRow * (batchDim + 1);
    int indexFlat = indexStartFlat + indexCol;

    int offset = 1;

    // Copy from the array to shared memory.
    if (realIndexCol < (dimCol - 1)){
        temp[indexFlat] = d_array[realIndexFlatIn];
        temp[indexFlat + 1] = d_array[realIndexFlatIn + strideColIn];
    }
    else if (realIndexCol == (dimCol - 1)){
        temp[indexFlat] = d_array[realIndexFlatIn];
        temp[indexFlat + 1] = 0.;
    }
    else{
        temp[indexFlat] = 0.;
        temp[indexFlat + 1] = 0.;
    }
    // Reduce by storing the intermediate values. The last element will be
    // the sum of n-1 elements.
    for (int d = blockDim.y; d > 0; d = d/2) {
        __syncthreads();

        // Regulates the amount of threads operating.
        if (threadIndexCol < d) {
            // Swap the numbers
            int current = offset * (indexCol + 1) - 1;
            int next = offset * (indexCol + 2) - 1;
            if (next < batchDim){
                temp[indexStartFlat + next] += temp[indexStartFlat + current];
            }
            /*
            temp[indexStartFlat + next] += temp[indexStartFlat + current];
            */
        }

        // Increase the offset by multiple of 2.
        offset *= 2;
    }

    // Only one thread performs this.
    if (threadIndexCol == 0) {
        // Store the sum on the last index of temp
        temp[indexStartFlat + batchDim] = temp[indexStartFlat + batchDim - 1];
        // Store the sum to the auxiliary array.
        if (store) {
            d_aux[blockIdx.x * gridDim.y + blockIdx.y] = temp[indexStartFlat + batchDim];
        }
        // Reset the last element with identity. Only the first thread will do
        // the job.
        temp[indexStartFlat + batchDim - 1] = 0;
    }
    // Down sweep to build scan.
    for (int d = 1; d < blockDim.y*2; d *= 2) {

        // Reduce the offset by division of 2.
        offset = offset / 2;

        __syncthreads();

        if (threadIndexCol < d)
        {
            int current = offset * (indexCol + 1) - 1;
            int next = offset * (indexCol + 2) - 1;

            // Swap
            if (next < batchDim){
                double tempCurrent = temp[indexStartFlat + current];
                temp[indexStartFlat + current] = temp[indexStartFlat + next];
                temp[indexStartFlat + next] += tempCurrent;
            } else if (current < batchDim){  // peut-être pas nécessaire...
                temp[indexStartFlat + current] = 0;
            }
            /*
            double tempCurrent = temp[indexStartFlat + current];
            temp[indexStartFlat + current] = temp[indexStartFlat + next];
            temp[indexStartFlat + next] += tempCurrent;
            */
        }
    }
    __syncthreads();

    if (realIndexCol >= dimCol) {return;}
    d_result[realIndexFlatOut] = temp[indexFlat + 1]; // write results to device memory
    if (realIndexCol < dimCol - 1){
        d_result[realIndexFlatOut + strideColOut] = temp[indexFlat + 2];
    }
}
""",
    "inclusive_scan_2d",
)


sum_inclusive_scan_2d = utils.xp_utils.back("cupy").RawKernel(
    r"""
extern "C" __global__
void sum_inclusive_scan_2d(
    double *d_incr,
    double *d_result,
    int dimRow,
    int dimCol,
    int strideOutRow,
    int strideOutCol
) {
    if (blockIdx.y > 0){
        double addThis = d_incr[blockIdx.x * gridDim.y + blockIdx.y - 1];
        int tid_row = threadIdx.x + blockDim.x * blockIdx.x;
        int tidCol = threadIdx.y + blockDim.y * blockIdx.y;
        if ((tidCol < dimCol) && (tid_row < dimRow)){
            d_result[tid_row * strideOutRow + tidCol * strideOutCol] += addThis;
        }
    }
}
""",
    "sum_inclusive_scan_2d",
)
