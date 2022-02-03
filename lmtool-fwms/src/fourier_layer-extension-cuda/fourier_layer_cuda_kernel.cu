#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES


namespace{
//const float eps = 1e-31f;

template <typename scalar_t>
__global__ void fourier_layer_cuda_forward_kernel(
          scalar_t* p_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N){
	  
	//extract n,b,q,k
	// size = [qlen, klen bsz, n_head]
	int q =   i / (klen*bsz*n_head);
	int k = ( i % (klen*bsz*n_head) ) / (bsz*n_head) ;
	int b = ( i % (bsz*n_head)     ) / n_head;
	int n =   i % n_head;
	
	//scalar_t& result = p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
	scalar_t& result = p_Y[i];
	
	const scalar_t* p_head_q_i = p_head_q + q*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	const scalar_t* p_head_k_i = p_head_k + k*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	result=1.0f;
	
	//sum on d
	for(int d=0; d<d_head; d++){	
	  //float diff = ( p_head_q[n,b,q,d] - p_head_k[n,b,k,d] ) 	 
	  scalar_t diff = ( p_head_q_i[d] - p_head_k_i[d] ) * p_paramR[0];
				
	  if(abs(diff)<1e-30f) diff=1;
	  else diff = sinf(diff)/diff;
								
	  result = result * diff ;
	}
	
	result *= __powf(p_paramR[0],d_head);
	
	
  }//end if	
  
}


// compute grad_head_q
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_q(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_head_q,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N_q) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N_q){
	// size(head_q) = [qlen, bsz, n_head, d_head]
	int q =   i / (bsz*n_head*d_head);
	int b = ( i % (bsz*n_head*d_head) ) / (n_head*d_head) ;
	int n = ( i % (n_head*d_head)     ) / d_head;
	int d =   i %  d_head;
	
	//scalar_t& result = p_grad_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	scalar_t& result = p_grad_head_q[i];
	const scalar_t& p_head_q_i  = p_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	const scalar_t* p_head_k_i  = p_head_k + b*n_head*d_head + n*d_head + d;
	result = 0;
	const scalar_t* p_Y_i      = p_Y      + q*klen*bsz*n_head  + b*n_head + n;
	const scalar_t* p_grad_Y_i = p_grad_Y + q*klen*bsz*n_head  + b*n_head + n;
	//sum on k
	for(int k=0; k<klen; k++){		
	  scalar_t diff = ( p_head_q_i - p_head_k_i[0] ) *p_paramR[0];
	  p_head_k_i += bsz*n_head*d_head;
				
	  if(abs(diff)<1.0e-30f) diff=0;
	  else diff =  1.0f/tanf(diff) - 1.0f/diff  ;

	  result     += diff * p_Y_i[0] * p_grad_Y_i[0];
	  p_Y_i      += bsz*n_head;
	  p_grad_Y_i += bsz*n_head;
	}
	result *= p_paramR[0];
  }
  
}


// compute grad_head_k
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_k(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_head_k,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N_k) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N_k){
	// size(head_k) = [klen, bsz, n_head, d_head]
	int k =   i / (bsz*n_head*d_head);
	int b = ( i % (bsz*n_head*d_head) ) / (n_head*d_head) ;
	int n = ( i % (n_head*d_head)     ) / d_head;
	int d =   i %  d_head;

	//float& result = p_grad_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	scalar_t& result = p_grad_head_k[i];
	const scalar_t* p_head_q_i = p_head_q + b*n_head*d_head + n*d_head + d;
	const scalar_t& p_head_k_i = p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	const scalar_t* p_Y_i      = p_Y      + k*bsz*n_head + b*n_head + n;
	const scalar_t* p_grad_Y_i = p_grad_Y + k*bsz*n_head + b*n_head + n;
	result = 0;
	//product on q
	for(int q=0; q<qlen; q++){		
	  float diff = ( p_head_q_i[0] - p_head_k_i ) *p_paramR[0];
	  p_head_q_i += bsz*n_head*d_head;
				
	  if(abs(diff)<1.0e-30f) diff=1;
	  else diff = 1.0f/tanf(diff) - 1.0f/diff  ;
		
	  result     -= diff * p_Y_i[0] * p_grad_Y_i[0];	
	  p_Y_i      += klen*bsz*n_head;
	  p_grad_Y_i += klen*bsz*n_head;
	}
	result *= p_paramR[0];
			
  }
  
}



// compute grad_paramR
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_paramR(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_paramR,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N_paramR) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N_paramR){
	// s [qlen,klen, bsz, n_head ]
	int q =   i / (klen*bsz*n_head);
	int k = ( i % (klen*bsz*n_head) ) / (bsz*n_head) ;
	int b = ( i % (bsz*n_head)     ) / n_head;
	int n =   i %  n_head;

	const scalar_t* p_head_q_i = p_head_q + q*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	const scalar_t* p_head_k_i = p_head_k + k*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	scalar_t temp = 0;
	for(int d=0; d<d_head; d++){
	  scalar_t diff = ( p_head_q_i[d] - p_head_k_i[d] );
				
	  if(abs(diff)<1.0e-30f) temp += 1.0f/p_paramR[0];
	  else                   temp += diff / tanf(p_paramR[0]*diff) ;
	}
	p_grad_paramR[0] += temp * p_grad_Y[i] * p_Y[i];
			
  }
  
}


}//end namespace
  
  
// 1. forward  
//template <typename T>
torch::Tensor fourier_layer_cuda_forward(
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR)
{ 
  const auto n_head     = head_q.size(2);
  const auto bsz        = head_q.size(1);
  const auto qlen       = head_q.size(0);
  const auto klen       = head_k.size(0);
  const auto d_head     = head_k.size(3);
  
  const auto N = qlen* klen * bsz * n_head;
  
  const int threads = 1024;
  const dim3 blocks((N + threads - 1) / threads, bsz);
  //const int blocks =  (N + threads - 1) / threads ;

  auto options = torch::TensorOptions().dtype(head_q.dtype())
                                       .layout(torch::kStrided)
                                       .device(torch::kCUDA, 0)
                                       .requires_grad(true);
	
  auto Y = torch::zeros( {qlen, klen, bsz, n_head}, options );

  //float* p_Y       = Y.data<float>();         // [qlen, klen, bsz, n_head]
  //float* p_head_q  = head_q.data<float>();    // [qlen, bsz, n_head, d_head]
  //float* p_head_k  = head_k.data<float>();    // [klen, bsz, n_head, d_head]
  //float* p_paramR  = paramR.data<float>();     
  
  
  AT_DISPATCH_FLOATING_TYPES(head_q.type(), "fourier_layer_forward_cuda", ([&] {
    fourier_layer_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
  Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(), 
  n_head, bsz, qlen, klen, d_head, N);
  }));
  //printf("Error in cuda: %s\n", cudaGetLastError());

  return Y;
}



std::vector<torch::Tensor> fourier_layer_cuda_backward(
//torch::Tensor my_linear_backward(
		const torch::Tensor& grad_Y,
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR,
		const torch::Tensor& Y)
{
  int n_head     = head_q.size(2);
  int bsz        = head_q.size(1);
  int qlen       = head_q.size(0);
  int klen       = head_k.size(0);
  int d_head     = head_k.size(3);
  
  const int threads=1024;
  auto const N_q = qlen*bsz*n_head*d_head;
  auto const N_k = klen*bsz*n_head*d_head;
  auto const N_paramR = qlen*klen*bsz*n_head;
  const int blocks_q = (N_q + threads - 1)/threads;
  const int blocks_k = (N_k + threads - 1)/threads;
  const int blocks_paramR = (N_paramR + threads - 1)/threads;
	
  auto grad_head_q = torch::zeros_like(head_q); //[qlen,bsz,n_head,d_head]
  auto grad_head_k = torch::zeros_like(head_k); //[klen,bsz,n_head,d_head]
  auto grad_paramR = torch::zeros_like(paramR); //[1]
  
  AT_DISPATCH_FLOATING_TYPES(head_q.type(), "fourier_layer_cuda_backward_kernel_q", 
  ([&] {fourier_layer_cuda_backward_kernel_q<scalar_t><<<blocks_q, threads>>>(
  grad_Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(),
  Y.data<scalar_t>(),  
  grad_head_q.data<scalar_t>(),
  n_head, bsz, qlen, klen, d_head, N_q);
  }));  
	
  AT_DISPATCH_FLOATING_TYPES(head_k.type(), "fourier_layer_cuda_backward_kernel_k", 
  ([&] {fourier_layer_cuda_backward_kernel_k<scalar_t><<<blocks_k, threads>>>(
  grad_Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(),
  Y.data<scalar_t>(),  
  grad_head_k.data<scalar_t>(),
  n_head, bsz, qlen, klen, d_head, N_k);
  }));  
  
  AT_DISPATCH_FLOATING_TYPES(head_k.type(), "fourier_layer_cuda_backward_kernel_paramR", 
  ([&] {fourier_layer_cuda_backward_kernel_paramR<scalar_t><<<blocks_paramR, threads>>>(
  grad_Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(),
  Y.data<scalar_t>(),  
  grad_paramR.data<scalar_t>(),
  n_head, bsz, qlen, klen, d_head, N_paramR);
  }));    
  
  return {grad_head_q, grad_head_k, grad_paramR};
}