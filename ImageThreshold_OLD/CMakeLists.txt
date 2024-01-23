cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ImageThreshold)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ImageThreshold")
set(EXTENSION_CATEGORY "Excercise")
set(EXTENSION_CONTRIBUTORS "Dang The Hung (King's College London)")
set(EXTENSION_DESCRIPTION "Learning to codeeeeeeeeeeee")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/ImageThreshold.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/ImageThreshold/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(t_ApplyThreshold)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
