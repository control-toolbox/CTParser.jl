# Typed CTBase errors for the argument checks.
# Part of the extension CTParserExaModels, included by CTParserExaModels.jl.

# ============================================================================
# Argument checks
#
# Typed CTBase errors rather than upstream's assertions, which can be compiled out:
# incompatible dimensions relate two operands (`PreconditionError`), a single operand
# of the wrong shape is an `IncorrectArgument`.
# ============================================================================

function _dimension_mismatch(op, reason)
    throw(
        CTBase.PreconditionError(
            "operand dimensions do not match for $op";
            reason=reason,
            suggestion="pass operands with compatible dimensions",
            context="linear algebra on ExaModels nodes",
        ),
    )
end

function _bad_operand(op, got, expected)
    throw(
        CTBase.IncorrectArgument(
            "unsupported operand for $op";
            got=got,
            expected=expected,
            suggestion="pass an operand of the expected shape",
            context="linear algebra on ExaModels nodes",
        ),
    )
end
