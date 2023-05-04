#include <cumpsgemm/cumpsgemm.hpp>
#include <sstream>
#include <stdexcept>

#ifndef CHECK_ERROR
#define CHECK_ERROR(status) cuda_check(status, __FILE__, __LINE__, __func__)
#endif

namespace {
inline void cuda_check(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss<< cudaGetErrorString( error );
		ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
}

constexpr std::size_t N = 2048;

int main() {
	float *a_ptr, *b_ptr, *c_ptr;
	CHECK_ERROR(cudaMalloc(&a_ptr, sizeof(float) * N * N));
	CHECK_ERROR(cudaMalloc(&b_ptr, sizeof(float) * N * N));
	CHECK_ERROR(cudaMalloc(&c_ptr, sizeof(float) * N * N));

	cumpsgemm::handle_t cumpsgemm_handle;
	cumpsgemm::create(cumpsgemm_handle);
	//cumpsgemm::set_stream(cumpsgemm_handle, cuda_stream);

	float alpha = 1.f, beta = 0.f;
	cumpsgemm::gemm(
			cumpsgemm_handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			N, N, N,
			&alpha,
			a_ptr, N,
			b_ptr, N,
			&beta,
			c_ptr, N,
			CUMPSGEMM_TF32TCEC
			);

	CHECK_ERROR(cudaFree(a_ptr));
	CHECK_ERROR(cudaFree(b_ptr));
	CHECK_ERROR(cudaFree(c_ptr));
}
