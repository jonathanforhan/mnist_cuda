cmake_minimum_required (VERSION 3.8)
project(mnist_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

if(MSVC)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)

	# Enable Hot Reload for MSVC compilers if supported.
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()
endif()

add_executable(
	mnist_cuda
		"src/main.cpp"
		"src/MNISTImage.cpp"
		"src/NeuralNetwork.cpp"
		"src/kernels.cu"
)

target_include_directories(
	mnist_cuda
	PRIVATE
		"src/"
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
	mnist_cuda
	PRIVATE
		CUDA::cublas
		CUDA::cudart
		CUDA::curand
)

execute_process(
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/data"
		"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data"
)
