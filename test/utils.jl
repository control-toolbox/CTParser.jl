# utils.jl (for tests only)
#

# `Logging` is a standard library, but adding it to the test environment makes Pkg.test ask
# the registry for it ("expected package Logging to be registered"); Base exposes the same API.
using Base.CoreLogging: CoreLogging, AbstractLogger, global_logger, with_logger

macro ignore(e)
    return :()
end

"""
    DropLogger(inner, pattern)

A logger that forwards every message to `inner` except those whose text matches `pattern`.
Used to hide a warning that a test provokes on purpose, while any other warning stays
visible (a global `NullLogger` would hide them all).
"""
struct DropLogger{L<:AbstractLogger} <: AbstractLogger
    inner::L
    pattern::Regex
end
CoreLogging.min_enabled_level(l::DropLogger) = CoreLogging.min_enabled_level(l.inner)
CoreLogging.shouldlog(l::DropLogger, args...) = CoreLogging.shouldlog(l.inner, args...)
CoreLogging.catch_exceptions(l::DropLogger) = CoreLogging.catch_exceptions(l.inner)
function CoreLogging.handle_message(l::DropLogger, level, message, args...; kwargs...)
    occursin(l.pattern, string(message)) && return nothing
    return CoreLogging.handle_message(l.inner, level, message, args...; kwargs...)
end

"""
    capture_stdout(f) -> (result, output)

Run `f()` with `stdout` redirected to a temporary file. `result` is the value of `f()`,
or the exception it threw; `output` is everything printed meanwhile. A file rather than
a `Pipe`, so a large output cannot fill the pipe buffer and block the writer.
"""
function capture_stdout(f)
    return mktemp() do path, io
        result = redirect_stdout(io) do
            try
                f()
            catch err
                err
            end
        end
        flush(io)
        return result, read(path, String)
    end
end

# Mock up of CTDirect.discretise for tests

function discretise_exa(
    ocp;
    scheme=CTParser.__default_scheme_exa(),
    grid_size=CTParser.__default_grid_size_exa(),
    backend=CTParser.__default_backend_exa(),
    init=CTParser.__default_init_exa(),
    base_type=CTParser.__default_base_type_exa(),
)
    build_exa = CTModels.get_build_examodel(ocp)
    return build_exa(;
        scheme=scheme, grid_size=grid_size, backend=backend, init=init, base_type=base_type
    )[1]
end

function discretise_exa_full(
    ocp;
    scheme=CTParser.__default_scheme_exa(),
    grid_size=CTParser.__default_grid_size_exa(),
    backend=CTParser.__default_backend_exa(),
    init=CTParser.__default_init_exa(),
    base_type=CTParser.__default_base_type_exa(),
)
    build_exa = CTModels.get_build_examodel(ocp)
    return build_exa(;
        scheme=scheme, grid_size=grid_size, backend=backend, init=init, base_type=base_type
    )
end
