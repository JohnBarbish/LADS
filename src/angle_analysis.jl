#-----------------------------------------------------------------------------#
# functions for calculating angle between CLVs along with subspaces spanned by
# several CLVs into a manifold

function minimumManifoldAngle(datafile::String, uInd, sInd)
    # calculate minimum principal angle of specified subspaces in C
    # initializes theta matrix to contain the mimimum angle
    fid = h5open(datafile, "r")
    # rH = fid["r"];
    cH = fid["c"];
    ns = size(cH, 3);
    ne = size(cH, 1);
    θ = zeros(ns)
    cTemp = zeros(ne, ne);
    @showprogress "Manifold Angle " for t=1:ns
        cTemp = reshape(cH[:, :, t], (ne, ne));
        θ[t] = minimumManifoldAngle(cTemp, uInd, sInd)
    end
    close(fid)
    # return theta array containing minimum principal angle at each timestep
    return θ
end

function minimumManifoldAngle(datafile::String, uInd, sInd, nsim)
    # calculate minimum principal angle of specified subspaces in C
    # initializes theta matrix to contain the mimimum angle
    fid = h5open(datafile, "r")
    # rH = fid["r"];
    cH = fid["c"];
    ns = size(cH, 3);
    ne = size(cH, 1);
    nsResets = Int(ns/nsim);
    θ = zeros(ns);
    cTemp = zeros(ne, ne, nsim);
    @showprogress "Manifold Angle " for t=1:nsResets
        trng = range((t-1)*nsim+1, length=nsim)
        cTemp = cH[:, :, trng];
        θ[trng] = minimumManifoldAngle(cTemp, uInd, sInd)
    end
    close(fid)
    # return theta array containing minimum principal angle at each timestep
    return θ
end

function minimumManifoldAngle(C::Array{Float64, 3}, uInd, sInd)
    # calculate minimum principal angle of specified subspaces in C
    # initializes theta matrix to contain the mimimum angle
    θ = zeros(size(C, 3))
    ns = size(C, 3)
    for t=1:ns
        θ[t] = minimumManifoldAngle(C[:, :, t], uInd, sInd)
    end
    # return theta array containing minimum principal angle at each timestep
    return θ
end

function minimumManifoldAngle(C::Array{Float64, 2}, uInd, sInd)
    u1 = C[:, uInd]; u2 = C[:, sInd];
    # use Array function to get correct size Q from QR decomposition quickly
    q1 = Array(qr(u1).Q); q2 = Array(qr(u2).Q);
    theta = acos(round(svdvals(q2'q1)[1], digits=14));
    return theta
end
export minimumManifoldAngle

"""
  pdf(data::Array{Float64, 1}, nbins::Int)
  returns a pdf version of the input data.  a primitive version of a histogram.
"""
function pdf(data::Array{Float64, 1}, nbins::Int)
  mx = maximum(data)
  mx /= nbins
  dist = zeros(nbins)
  for datum in data
    if !isapprox(datum, mx*nbins)
      dist[floor(Int, div(datum, mx)) + 1] += 1
    else
      dist[nbins] += 1
    end
  end
  dist /= length(data)
  return dist
end

"""
  pdf(data::Array{Float64, 1}, nbins::Int, xlims::Tuple)
  returns a pdf version of the input data.  a primitive version of a histogram.
"""
function pdf(data::Array{Float64, 1}, nbins::Int, xlims)
    mn, mx = xlims
    boxes = range(mn, length=nbins, stop=mx)
    boxWidth = (mx - mn)/nbins
    dist = zeros(nbins)
    for datum in data
        if !isapprox(datum, mx*nbins)
            dist[floor(Int, div(datum-mn, boxWidth)) + 1] += 1
        else
            dist[nbins] += 1
        end
    end
    # renormalize distribution so integral is one
    dist /= length(data)
    # @assert abs(sum(dist) - 1) < 10^-10
    return boxes, dist
end
"""
  pdf(data::Array{Float64, 1}, nbins::Int, xlims::Tuple)
  returns a pdf version of the input data.  a primitive version of a histogram.
"""
function pdf(data::Array{Float64, 1},
    theta::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})
    mn, mx, nbins = theta[1], theta[end], length(theta)
    xlims = (mn, mx)
    a, dist = pdf(data, nbins, xlims)
    return dist
end
export pdf

"""
  angle(v1::Arrray{Float64, 1}, v2::Array{Float64, 1})

  Returns angle between two one dimensional vectors. Assumes equal length.
"""
function angle(v1::Array{Float64, 1}, v2::Array{Float64, 1})
  return acos(round(dot(v1, v2)/(norm(v1)*norm(v2)), digits=14))
end
export angle

"""
  allAngles(C::Array{Float64, 2})

  Returns angle between all vectors in set, assuming normal vectors
"""
function allAngles(C::Array{Float64, 2}, normalized=true)
    if !normalized
        for i = 1:size(C, 2)
            C[:, i] = C[:, i]/norm(C[:, i])
        end
    end
    return acos.(round.(C'*C, digits=14))
end
"""
  allAngles(C::Array{Float64, 3})

  Returns angle between all vectors in set, assuming normal vectors
"""
function allAngles(C::Array{Float64, 3}, normalized=true)
    ns = size(C, 3);
    data = zeros((size(C, 2), size(C, 2), ns));
    for t = 1:ns
        data[:, :, t] = allAngles(C[:, :, t], normalized)
    end
    return data
end
export allAngles
"""
  allAngleDistribution(C::Array{Float64, 3}, normalized=true, nbins=100, xlims=(0, pi))

  Returns angle between all vectors in set, assuming normal vectors.
"""
function allAngleDistribution(C::Array{Float64, 3},normalized::Bool,
     nbins::Int64, xlims::Tuple{Real, Real})
    data = zeros(size(C));
    ns = size(C, 3);
    for t = 1:ns
        data[:, :, t] = allAngles(C[:, :, t], normalized)
    end
    ne = size(C, 1);
    dist = zeros(ne, ne, nbins);
    # timeseries = zeros(ns);
    for i = 1:ne
        for j=1:ne
            # println("i:\t$i,\tj:\t$j")
            a, dist[i, j, :] = pdf(data[i, j, :], nbins, xlims)
        end
    end
    return dist
end
"""
  allAngleDistribution(C::Array{Float64, 3},
  theta::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})

  Returns angle between all vectors in set, assuming normal vectors.
"""
function allAngleDistribution(C::Array{Float64, 3},
normalized=true,
theta::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=range(0, stop=pi, length=100))
    data = zeros(size(C));
    ns = size(C, 3);
    nbins = length(theta);
    for t = 1:ns
        data[:, :, t] = allAngles(C[:, :, t], normalized)
    end
    ne = size(C, 1);
    dist = zeros(ne, ne, nbins);
    # timeseries = zeros(ns);
    for i = 1:ne
        for j=1:ne
            # println("i:\t$i,\tj:\t$j")
            dist[i, j, :] = pdf(data[i, j, :], theta)
        end
    end
    return dist
end
"""
  allAngleDistribution(datafile::String,
  theta::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})

  Returns angle between all vectors in set, assuming normal vectors.
"""
function allAngleDistribution(datafile::String, normalized, nsim::Int64,
theta::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}=range(0, stop=pi, length=100))
    fid = h5open(datafile, "r");
    vH = fid["v"];
    ne = size(vH, 2); ns = size(vH, 3);
    mn, mx, nbins = theta[1], theta[end], length(theta);
    nsResets = Int(ns/nsim);
    data = zeros(ne, ne, nbins);
    count = zeros(ne, ne, nbins);
    boxWidth = (mx - mn)/nbins
    @showprogress "Calculating Angles " for t = 1:nsResets
        trng = range((t-1)*nsim+1, length=nsim)
        data = allAngles(vH[:, :, trng], normalized)
        for i=1:ne
            for j=1:ne
                count[i, j, :] += pdf(data[i, j, :], theta)*nsim*boxWidth
            end
        end
    end
    close(fid)
    count /= (boxWidth*ns)
    return count
end

export allAngleDistribution