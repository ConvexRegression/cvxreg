"""
constant for string options
"""

# Shape of the function
convex = "convex"
"""
Convex: convex function.
"""

concave = "concave"
"""
Concave: concave funciton.
"""

# Monotonicity of the function
increasing = "increasing"
"""
Monotonic increasing: the function is non-decreasing.
"""
decreasing = "decreasing"   
"""
Monotonic decreasing: the function is non-increasing.
"""

FUN_Categories = {
    convex: "convex function",
    concave: "concave funciton",
    increasing: "monotonic increasing function",
    decreasing: "monotonic decreasing function" 
}

# Optimization
OPT_LOCAL = "local"
OPT_DEFAULT = None
