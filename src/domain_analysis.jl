"""
    edge_location_zc(data, zc=None)

    finds index locations of edges or defects in data
    based on when data crosses the value zc, or average when not specified
"""
function edge_location_zc(data, zc=NaN)
    if isnan(zc)
        zc = mean(data)
    end
    # determine when signal is above/below zc
    pm = sign.(data .- zc)
    # find indicies where sign changes
    edges = [ind for (ind, val) in enumerate(abs.(diff(pm))) if val > 0]
    return edges
end
export edge_location_zc

"""
    avginband(N, xL, xH, c)

    returns a 1D array of size N between bounds (xL, xH) with an average c.
"""
function avginband(N, xL, xH, c)
    @assert xL < c
    @assert c < xH
    return suminband(N, xL, xH, c*N);
end
export avginband
"""
    suminband(N, lb, ub, sum)

    returns a 1D array of size N between bounds (lb, ub) that totals to sum.
"""
function suminband(N, lb, ub, sum)
    # creates lattice of length N with
    # each element between lb < x < ub
    # and totally up to sum
    @assert lb < sum/N
    @assert sum/N < ub
    x = zeros(N)
    tsum = copy(sum)
    for i = 1:N
        nlb = max(lb, tsum - ub*(N-i))
        nub = min(ub, tsum - lb*(N-i))
        x[i] = rand()*(nub-nlb) + nlb
        tsum -= x[i]
    end
    return x
end
export suminband