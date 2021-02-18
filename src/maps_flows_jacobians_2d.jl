"""
    wrap2d(i, j, nx)
    Convience function for consistently converting from two dimensional array to
    a one dimensional array. Works with rectangular set of coordinates.
"""
function wrap2d(i, j, nx)
    return i + (j-1)*nx
end

"""
    unwrap2d(k, nx)
    Convience function for consistently converting back from one dimensional array
    to two dimensional array returns coordinates in second dimensional  
"""
function unwrap2d(k, nx)
    (j, i) = divrem(k, nx)
    if i == 0
        i = nx;
    else
        j += 1;
    end
    return (i, j)
end

# write 2D version of modelA

function modelAMap2D(y, p)
    # general timestep function for 1D lattice with periodic B.C.
    # h = size(y, 1) # number of rows
    (nr, nc) = size(y); # y is two dimensional
    x = zeros((nr, nc)) # copy(y)
    # timestep middle of lattice
    for i = 2:nr-1, j=2:nc-1
        x[i, j] = (y[i, j] + ag(y[i, j], p) + af(y[i-1, j], p) + af(y[i+1, j], p) +
        af(y[i, j-1], p) + af(y[i, j+1], p) - 4*af(y[i, j], p))
    end
    # timestep first and last rows
    for j=2:nc-1
        # first row
        x[1, j] = (y[1, j] + ag(y[1, j], p) + af(y[nr, j], p) + af(y[2, j], p) +
        af(y[1, j-1], p) + af(y[1, j+1], p) - 4*af(y[1, j], p))
        # last row
        x[nr, j] = (y[nr, j] + ag(y[nr, j], p) + af(y[nr-1, j], p) + af(y[1, j], p) +
        af(y[nr, j-1], p) + af(y[nr, j+1], p) - 4*af(y[nr, j], p))
    end
    # timestep first and last columns
    for i=2:nr-1
        # first column
        x[i, 1] = (y[i, 1] + ag(y[i, 1], p) + af(y[i-1, 1], p) + af(y[i+1, 1], p) +
        af(y[i, 2], p) + af(y[i, nc], p) - 4*af(y[i, 1], p))
        # last column
        x[i, nc] = (y[i, nc] + ag(y[i, nc], p) + af(y[i-1, nc], p) + af(y[i+1, nc], p) +
        af(y[i, 1], p) + af(y[i, nc-1], p) - 4*af(y[i, nc], p))
    end
    # four corners
    x[1, 1] = (y[1, 1] + ag(y[1, 1], p) + af(y[nr, 1], p) + af(y[2, 1], p) +
                af(y[1, 2], p) + af(y[1, nc], p) - 4*af(y[1, 1], p))

    x[nr, 1] = (y[nr, 1] + ag(y[nr, 1], p) + af(y[nr-1, 1], p) + af(y[1, 1], p) +
                af(y[nr, 2], p) + af(y[nr, nc], p) - 4*af(y[nr, 1], p))

    x[1, nc] = (y[1, nc] + ag(y[1, nc], p) + af(y[1, nc-1], p) + af(y[1, 1], p) +
                af(y[nc-1, nc], p) + af(y[2, nc], p) - 4*af(y[1, nc], p))

    x[nr, nc] = (y[nr, nc] + ag(y[nr, nc], p) + af(y[nr, 1], p) + af(y[nr, nc-1], p) +
                af(y[1, nc], p) + af(y[nr-1, nc], p) - 4*af(y[nr, nc], p))
    return x
end
export modelAMap2D

function modelAMap2Dflat(y, p)
    # general timestep function for 2D lattice with periodic B.C.
    # reserve slots 5 and 6 of parameters for number of rows and columns
    nr = p[5]; nc = p[6];
    @assert length(y) == nr*nc;
    # h = size(y, 1) # number of rows
    y = reshape(y, (nr, nc));
    # (nr, nc) = size(y); # y is two dimensional
    # x = zeros((nr, nc)) # copy(y)
    x = modelAMap2D(y, p);

    # # timestep middle of lattice
    # for i = 2:nr-1, j=2:nc-1
    #     x[i, j] = (y[i, j] + ag(y[i, j], p) + af(y[i-1, j], p) + af(y[i+1, j], p) +
    #     af(y[i, j-1], p) + af(y[i, j+1], p) - 4*af(y[i, j], p))
    # end
    # # timestep first and last rows
    # for j=2:nc-1
    #     # first row
    #     x[1, j] = (y[1, j] + ag(y[1, j], p) + af(y[nr, j], p) + af(y[2, j], p) +
    #     af(y[1, j-1], p) + af(y[1, j+1], p) - 4*af(y[1, j], p))
    #     # last row
    #     x[nr, j] = (y[nr, j] + ag(y[nr, j], p) + af(y[nr-1, j], p) + af(y[1, j], p) +
    #     af(y[nr, j-1], p) + af(y[nr, j+1], p) - 4*af(y[nr, j], p))
    # end
    # # timestep first and last columns
    # for i=2:nr-1
    #     # first column
    #     x[i, 1] = (y[i, 1] + ag(y[i, 1], p) + af(y[i-1, 1], p) + af(y[i+1, 1], p) +
    #     af(y[i, 2], p) + af(y[i, nc], p) - 4*af(y[i, 1], p))
    #     # last column
    #     x[i, nc] = (y[i, nc] + ag(y[i, nc], p) + af(y[i-1, nc], p) + af(y[i+1, nc], p) +
    #     af(y[i, 1], p) + af(y[i, nc-1], p) - 4*af(y[i, nc], p))
    # end
    # # four corners
    # x[1, 1] = (y[1, 1] + ag(y[1, 1], p) + af(y[nr, 1], p) + af(y[2, 1], p) +
    #             af(y[1, 2], p) + af(y[1, nc], p) - 4*af(y[1, 1], p))

    # x[nr, 1] = (y[nr, 1] + ag(y[nr, 1], p) + af(y[nr-1, 1], p) + af(y[1, 1], p) +
    #             af(y[nr, 2], p) + af(y[nr, nc], p) - 4*af(y[nr, 1], p))

    # x[1, nc] = (y[1, nc] + ag(y[1, nc], p) + af(y[1, nc-1], p) + af(y[1, 1], p) +
    #             af(y[nc-1, nc], p) + af(y[2, nc], p) - 4*af(y[1, nc], p))

    # x[nr, nc] = (y[nr, nc] + ag(y[nr, nc], p) + af(y[nr, 1], p) + af(y[nr, nc-1], p) +
    #             af(y[1, nc], p) + af(y[nr-1, nc], p) - 4*af(y[nr, nc], p))
    # flatten output
    x = reshape(x, :);
    return x
end
export modelAMap2Dflat

function modelAJacobian2D(x, p)
    # calculates Jacobian at point x
    # (nr, nc) = size(x); # x is two dimensional
    h = length(x); # total size of the array
    # construct Jacobian
    J = zeros(h, h);
    # iterate through middle lattice points
    for i=2:nr-1, j=2:nc-1
        iC = wrap2d(i, j, nr)
        iA = wrap2d(i, j+1, nr)
        iB = wrap2d(i, j-1, nr)
        iL = wrap2d(i-1, j, nr)
        iR = wrap2d(i+1, j, nr)
        J[iC, iA] = afprime(x[iA], p)
        J[iC, iB] = afprime(x[iB], p)
        J[iC, iL] = afprime(x[iL], p)
        J[iC, iR] = afprime(x[iR], p)
        J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    end
    # rows
    for j=2:nc-1
        # first row
        iC = wrap2d(1, j, nr)
        iA = wrap2d(1, j+1, nr)
        iB = wrap2d(1, j-1, nr)
        iL = wrap2d(nr, j, nr)
        iR = wrap2d(2, j, nr)
        J[iC, iA] = afprime(x[iA], p)
        J[iC, iB] = afprime(x[iB], p)
        J[iC, iL] = afprime(x[iL], p)
        J[iC, iR] = afprime(x[iR], p)
        J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
        # last row
        iC = wrap2d(nr, j, nr)
        iA = wrap2d(nr, j+1, nr)
        iB = wrap2d(nr, j-1, nr)
        iL = wrap2d(nr-1, j, nr)
        iR = wrap2d(nr, j, nr)
        J[iC, iA] = afprime(x[iA], p)
        J[iC, iB] = afprime(x[iB], p)
        J[iC, iL] = afprime(x[iL], p)
        J[iC, iR] = afprime(x[iR], p)
        J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    end
    # columns
    for i=2:nr-1
        # first column
        iC = wrap2d(i, 1, nr)
        iA = wrap2d(i, 2, nr)
        iB = wrap2d(i, nc, nr)
        iL = wrap2d(i-1, 1, nr)
        iR = wrap2d(i+1, 1, nr)
        J[iC, iA] = afprime(x[iA], p)
        J[iC, iB] = afprime(x[iB], p)
        J[iC, iL] = afprime(x[iL], p)
        J[iC, iR] = afprime(x[iR], p)
        J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
        # last column
        iC = wrap2d(i, nc, nr)
        iA = wrap2d(i, 1, nr)
        iB = wrap2d(i, nc-1, nr)
        iL = wrap2d(i-1, nc, nr)
        iR = wrap2d(i+1, nc, nr)
        J[iC, iA] = afprime(x[iA], p)
        J[iC, iB] = afprime(x[iB], p)
        J[iC, iL] = afprime(x[iL], p)
        J[iC, iR] = afprime(x[iR], p)
        J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    end
    # four corners
    # bottom left corner
    iC = wrap2d(1, 1, nr)
    iA = wrap2d(1, 2, nr)
    iB = wrap2d(1, nc, nr)
    iL = wrap2d(nr, 1, nr)
    iR = wrap2d(2, 1, nr)
    J[iC, iA] = afprime(x[iA], p)
    J[iC, iB] = afprime(x[iB], p)
    J[iC, iL] = afprime(x[iL], p)
    J[iC, iR] = afprime(x[iR], p)
    J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    # bottom right corner
    iC = wrap2d(nr, 1, nr)
    iA = wrap2d(nr, 2, nr)
    iB = wrap2d(nr, nc, nr)
    iL = wrap2d(nr-1, 1, nr)
    iR = wrap2d(1, 1, nr)
    J[iC, iA] = afprime(x[iA], p)
    J[iC, iB] = afprime(x[iB], p)
    J[iC, iL] = afprime(x[iL], p)
    J[iC, iR] = afprime(x[iR], p)
    J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    # top left corner
    iC = wrap2d(1, nc, nr)
    iA = wrap2d(1, 1, nr)
    iB = wrap2d(1, nc-1, nr)
    iL = wrap2d(nr, nc, nr)
    iR = wrap2d(2, nc, nr)
    J[iC, iA] = afprime(x[iA], p)
    J[iC, iB] = afprime(x[iB], p)
    J[iC, iL] = afprime(x[iL], p)
    J[iC, iR] = afprime(x[iR], p)
    J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    # top right corner
    iC = wrap2d(nr, nc, nr)
    iA = wrap2d(nr, 1, nr)
    iB = wrap2d(nr, nc-1, nr)
    iL = wrap2d(nr-1, nc, nr)
    iR = wrap2d(1, nc, nr)
    J[iC, iA] = afprime(x[iA], p)
    J[iC, iB] = afprime(x[iB], p)
    J[iC, iL] = afprime(x[iL], p)
    J[iC, iR] = afprime(x[iR], p)
    J[iC, iC] = 1 + agprime(x[iC], p) - 4*afprime(x[iC], p)
    return J
end
export modelAJacobian2D