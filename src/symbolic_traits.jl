"""
    SymbolicTraits

Trait system for symbolic vs numeric array operations.

This module defines trait types that CTParser owns, enabling clean dispatch
without type piracy in the core implementation.
"""

using ExaModels: AbstractNode

# ============================================================================
# TRAIT TYPE DEFINITIONS (CTParser owns these!)
# ============================================================================

"""
    IsSymbolic

Trait indicating an array contains symbolic nodes (AbstractNode elements).
Used for dispatch in symbolic operations.
"""
struct IsSymbolic end

"""
    IsNotSymbolic

Trait indicating an array contains non-symbolic elements (Numbers).
Used for dispatch in symbolic operations.
"""
struct IsNotSymbolic end

# ============================================================================
# TRAIT DETECTION FUNCTIONS
# ============================================================================

"""
    symbolic_trait(T::Type) -> Union{IsSymbolic, IsNotSymbolic}

Determine if a type contains symbolic elements.

# Examples
```julia
symbolic_trait(Vector{Float64})           # IsNotSymbolic()
symbolic_trait(Vector{Node1{...}})        # IsSymbolic()
symbolic_trait(Matrix{AbstractNode})      # IsSymbolic()
```
"""
# For arrays
symbolic_trait(::Type{<:AbstractArray{<:AbstractNode}}) = IsSymbolic()
symbolic_trait(::Type{<:AbstractArray{<:Number}}) = IsNotSymbolic()
symbolic_trait(::Type{<:AbstractArray}) = IsNotSymbolic()  # Fallback

# For scalars
symbolic_trait(::Type{<:AbstractNode}) = IsSymbolic()
symbolic_trait(::Type{<:Number}) = IsNotSymbolic()

# Convenience: work on values directly
symbolic_trait(x) = symbolic_trait(typeof(x))
