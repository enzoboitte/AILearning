cmake_minimum_required(VERSION 3.20)
project(Neuronal VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flags d'optimisation
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -Wpedantic -fsanitize=address")

# Trouver jsoncpp et BLAS
find_package(jsoncpp REQUIRED)
find_package(BLAS REQUIRED)

# Sources
set(SOURCES
    src/CModel.cpp
    src/CEntraineur.cpp
    main.cpp
)

# Headers (optionnel mais propre)
set(HEADERS
    src/include/CModel.hpp
    src/include/CEntraineur.hpp
)

# Exécutable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Lien jsoncpp
target_link_libraries(${PROJECT_NAME}
    jsoncpp_lib
    jsoncpp
    ${BLAS_LIBRARIES}
)

# Includes
target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${JSONCPP_INCLUDE_DIRS}
    ${BLAS_INCLUDE_DIRS}
)

# Optimisations
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -Wall -Wextra -Wpedantic>
)

include(CTest)
enable_testing()

# Packaging
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
