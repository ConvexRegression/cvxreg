# Composite error term
CET_ADDI = "addi"
"""
CET_ADDI: Additive composite error term.
"""

CET_MULT = "mult"
"""
CET_MULT: Multiplicative composite error term.
"""

CET_Categories = {
    CET_ADDI: "Additive composite error term",
    CET_MULT: "Multiplicative composite error term"
}

CET_Model_Categories = {
    CET_ADDI: "additive model",
    CET_MULT: "multiplicative model"
}

# Frontier
FUN_CVX = "convex"
"""
FUN_CVX: convex function.
"""

FUN_CCV = "concave"
"""
FUN_CCV: concave funciton.
"""

FUN_Categories = {
    FUN_CVX: "convex function",
    FUN_CCV: "concave funciton"
}

# Return to scale
RTS_VRS = "vrs"
"""
RTS_VRS: Variable returns to scale.
"""

RTS_CRS = "crs"
"""
RTS_CRS: Constant returns to scale.
"""

RTS_Categories = {
    RTS_VRS: "Variable returns to scale",
    RTS_CRS: "Constant returns to scale"
}

# Optimization
OPT_LOCAL = "local"
OPT_DEFAULT = None
