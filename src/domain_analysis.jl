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