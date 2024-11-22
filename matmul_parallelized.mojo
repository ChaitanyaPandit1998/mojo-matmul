import benchmark
from sys.intrinsics import strided_load
from memory import memset_zero
from random import rand, random_float64
from sys.info import simdwidthof
from algorithm import vectorize
from algorithm import parallelize

# simdwidthof = number of float32 elements that fit into a single SIMD register
# using a 2x multiplier allows some SIMD operations to run in the same cycle
alias nelts = simdwidthof[DType.float32]() * 2
alias type = DType.float32

struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data.address, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data.address, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

alias M = 128
alias N = 128
alias K = 128

@always_inline
fn bench[
    func: fn (Matrix, Matrix, Matrix) -> None](base_gflops: Float64):
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9
    var speedup: Float64 = gflops / base_gflops
    # print(secs," seconds")
    # print(gflops, "GFLOP/s")
    # print("matrix size:", M)
    print(gflops, "GFLOP/s, a", speedup, "x speedup over Python")


fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, nelts, size = C.cols]()
    parallelize[calc_row](C.rows, C.rows)

def main():
    bench[matmul_parallelized](0.0021852577360293567)
