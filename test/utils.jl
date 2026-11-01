# utils.jl (for tests only)
#

macro ignore(e)
    return :()
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
