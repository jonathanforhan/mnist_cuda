cmake_minimum_required (VERSION 3.8)
project(mnist_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )

	# Enable Hot Reload for MSVC compilers if supported.
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()

	set_property(GLOBAL PROPERTY VS_STARTUP_PROJECT mnist_cuda)
else()
	add_compile_options(
		-Wall -Wextra -Wpedantic -Wconversion
	)
endif()

execute_process(
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/data"
		"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data"
)

# LIBRARY
add_library(
	mnist_cuda_lib
	STATIC
		"src/MNISTImporter.cpp"
		"src/NeuralNetwork.cpp"
		"src/Tensor.hpp"
		"src/kernels.cu"
)

target_link_libraries(
	mnist_cuda_lib
	PUBLIC
		CUDA::cublas
		CUDA::cudart
		CUDA::curand
)

target_include_directories(
	mnist_cuda_lib
	PUBLIC
		"src/"
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# APPLICATION
add_executable(
	mnist_cuda
		"app/main.cpp"
)

target_link_libraries(
	mnist_cuda
	PRIVATE
		mnist_cuda_lib
)

# TESTS
add_executable(
	mnist_cuda_tests
		"tests/test.cpp"
		"tests/Tensor.test.cpp"
		"tests/MNISTImporter.test.cpp"
)

target_link_libraries(
	mnist_cuda_tests
	PRIVATE
		mnist_cuda_lib
)

add_test(NAME mnist_cuda_tests COMMAND mnist_cuda_tests)

add_custom_command(
	TARGET mnist_cuda_tests
	POST_BUILD
		COMMAND ${CMAKE_CTEST_COMMAND} -R "^mnist_cuda_tests$" "--output-on-failure"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		COMMENT "running post build tests..."
)
