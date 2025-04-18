module local_POD_overlap




# Include submodules
include("module_1D.jl")
include("module_2D.jl")

# Make them accessible via MyPackage.SubModuleA, etc.
using .module_1D
using .module_2D

end
