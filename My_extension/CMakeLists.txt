cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(My_extension)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/My_extension")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Dang The Hung (King's College London)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/My_extension.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/My_extension/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ImageThreshold)
add_subdirectory(Print_hello_world)
add_subdirectory(Guided_examples)
add_subdirectory(Segmentation_k_mean)
add_subdirectory(Functions)
add_subdirectory(Functions)
add_subdirectory(My_function)
add_subdirectory(Pathway_planning)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
