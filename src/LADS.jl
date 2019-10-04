module LADS

using Statistics, Random, LinearAlgebra
import JLD: jldopen, read, close
using HDF5
# We make the follow definitions for notation
# Maps  - descrete time
# Flows - continuous time



#-----------------------------------------------------------------------------#
# functions for integrating functions with finite difference
# Define Runge-Kutta 4 timestepping function
function rk4(v,X,p,h,n)
    # RK4   Runge-Kutta scheme of order 4
    #   performs n steps of the scheme for the vector field v
    #   using stepsize h on each row of the matrix X
    #   v maps an (m x d)-matrix to an (m x d)-matrix

    for i = 1:n
        k1 = v(X, p);
        k2 = v(X + h/2*k1, p);
        k3 = v(X + h/2*k2, p);
        k4 = v(X + h*k3, p);
        X = X + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
    return X
end
# export rk4

# Define 1st order finite difference scheme
function fd1(v, x, p, h, n)
    # first order finite difference
    # performs n steps of the scheme for the vector field v
    # using stepsize h on each row of the vector X
    # v is the first order derivative of vector X
    # returns the vector field after n steps
    for i=1:n
        x = x + h*v(x, p);
    end
    return x
end
# export fd1

#-----------------------------------------------------------------------------#
# functions for Lorenz System

function lorenzFlow(x, p)
    sigma = p[1];
    rho = p[2];
    beta = p[3];
    return [sigma*(x[2]-x[1]),              # the Lorenz system
          rho*x[1]-x[2]-x[1].*x[3],
          x[1].*x[2]-beta*x[3]];

end
# export lorenzFlow

function lorenzJacobian(xvec, p)
    sigma = p[1]; rho = p[2]; beta = p[3];
    x = xvec[1]; y = xvec[2]; z = xvec[3];
    J = [[-sigma sigma 0];
         [(rho - z) -1 -x];
         [y x -beta]]
    return J
end
# export lorenzJacobian

#-----------------------------------------------------------------------------#
# functions for model A System

function modelAMap(y, p)
  # general timestep function for 1D lattice with periodic B.C.
  h = size(y, 1) # number of rows
  x = zeros(h) # copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    x[i] = y[i] + af(y[i-1], p) + af(y[i+1], p) - 2*af(y[i], p)
  end
  # timestep first site
  x[1] = y[1] + af(y[h], p) + af(y[2], p) - 2*af(y[1], p)
  # timestep last site
  x[h] = y[h] + af(y[h-1], p) + af(y[1], p) - 2*af(y[h], p)
  return x
end
# export modelAMap

function modelAJacobian(x, p)
  # calculates Jacobian at point x
  h = size(x, 1) # number of rows
  # construct Jacobian
  J = zeros(h, h);
  for i in 2:h-1
    J[i, i-1] = afprime(x[i-1], p)
    J[i, i] = 1 - 2*afprime(x[i], p)
    J[i, i+1] = afprime(x[i+1], p)
  end
  J[1, 1] = 1 - 2*afprime(x[1], p)
  J[1, h] = afprime(x[h], p)
  J[1, 2] = afprime(x[2], p)
  J[h, h] = 1 - 2*afprime(x[h], p)
  J[h, 1] = afprime(x[1], p)
  J[h, h-1] = afprime(x[h-1], p)
  return J
end
# export modelAJacobian

function af(xi, p)::Float64
  z = mod(xi, 1)
  return p[1]*xi + p[2]*z*(1-z)
end
"""
  afprime(x)

v07
  function description
"""
function afprime(x, p)::Float64
  return p[1] + p[2]*(1 - 2*mod(x,1))
end


#-----------------------------------------------------------------------------#
# functions useful for calculating CLVs for flows

function advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
    ht = length(x0);
    v = zeros(ht);
    for i=1:nsps
        v = flow(x0, p);
        J = jacobian(x0, p);
        # advance delta and x0 by δt
        delta = (J*δt + Matrix(1.0I, ht, ht))*delta
        x0 = v*δt + x0
    end
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign.(diag(r))))
    r = sgn*r
    delta = q*sgn

    return x0, v, delta, r
end



function covariantLyapunovVectorsWarmupGSV(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypspecGS = clvGinelliForwardWarmupGSV(flow, jacobian, p, δt, x0,
                                                  delay, ns, ne, cdelay, nsps)
    CS = clvGinelliBackwards(QS, RS, Rw)
    lypspecCLV = lyapunovSpectrumCLV(CS, RS, nsps, δt)
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)

    return yS, QS, RS, CS, lypspecGS, lypspecCLV
end
export covariantLyapunovVectorsWarmupGSV

function covariantLyapunovVectors(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypspecGS = clvGinelliForward(flow, jacobian, p, δt, x0,
                                                  delay, ns, ne, cdelay, nsps)
    CS = clvGinelliBackwards(QS, RS, Rw)
    lypspecCLV = lyapunovSpectrumCLV(CS, RS, nsps, δt)
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)

    return yS, QS, RS, CS, lypspecGS, lypspecCLV
end
export covariantLyapunovVectors

function clvGinelliForward(flow, jacobian, p, δt, x0, delay, ns, ne, cdelay, nsps)
    # initialize local variables
    ht = length(x0)
    # construct set of perturbation vectors
    delta = zeros(ht, ne)
    for i=1:ne
        delta[i, i] = 1;
    end
    # the number of total steps for the system is the number of samples (ns)
    # * the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice
    x0 = fd1(flow, x0, p, δt, delay)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, ns);
    vn = zeros(ht, ns);
    RS = zeros(ne, ne, ns);
    QS = zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    for i=1:ns
        yS[:, i] = x0;
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        vn[:, i] = v/norm(v);
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*δt*nsps)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    for i=1:cdelay
        # advance delta and x0 by δt
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        Rw[:, :, i] = r
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS
end


function clvGinelliForwardWarmupGSV(flow, jacobian, p, δt, x0, delay, ns, ne, cdelay, nsps)
    # initialize local variables
    ht = length(x0)
    # construct set of perturbation vectors
    delta = zeros(ht, ne)
    for i=1:ne
        delta[i, i] = 1;
    end
    # the number of total steps for the system is the number of samples (ns)
    # * the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice and Gram-Schmidt Vectors
    # x0 = fd1(flow, x0, p, δt, delay)
    for i=1:delay
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
    end
    println("lattice and GSV warmed up, now recording GSV evolution.")

    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, ns);
    # deltaS = zeros(ht, ne, ns);
    vn = zeros(ht, ns);
    RS = zeros(ne, ne, ns);
    QS = zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    for i=1:ns
        yS[:, i] = x0;
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        vn[:, i] = v/norm(v);
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*δt*nsps)
    end

    # lypspecGS /= (ns*δt*nsps)

    println("collected QR data.")

    # create R data for warming up C
    # yc = zeros(ht, cdelay);
    # deltac = zeros(ht, ne, cdelay);
    Rw = zeros(ne, ne, cdelay);
    for i=1:cdelay
    #     yc[:, i] = x0;
    #     deltac[:, :, i] = delta;
        # advance delta and x0 by δt
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        Rw[:, :, i] = r
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS
end
# backwards evolution to get CLVs

# inputs:
# QS, RS, YS, CS, jacobian, Rw


function cEvolve(C, Rw, cdelay)
    # evolve C backwards through Rw
    # assumes that size(Rw, 3) == cdelay
    ne = size(C, 2);
    for i = cdelay:-1:1
        # renormalizes C for each vector
        for j=1:ne
            C[:, j] = C[:, j]/norm(C[:, j])
        end
        # creates preceding C vector
        C = inv(Rw[:, :, i])*C
    end
    return C
end

function clvInstantGrowth(C1::Array{Float64, 2}, R2::Array{Float64, 2})
  C2 = R2*C1
  ne = size(C1, 1)
  CLV_growth = zeros(ne)
  for i=1:ne
    CLV_growth[i] = (norm(reshape(C2[:, i], ne))/
    norm(reshape(C1[:, i], ne)))
  end
  return CLV_growth #::Array{Float64, 1} size(C, 1)
end

function lyapunovSpectrumCLV(CS, RS, nsps, δt)
    ns =size(CS, 3);
    ne = size(RS, 1);
    lypspec = zeros(ne);
    for i = 2:ns
        lypspec += log.(clvInstantGrowth(CS[:, :, i-1], RS[:, :, i]))
    end
    lypspec /= (ns*nsps*δt)
    return lypspec
end

function clvGinelliBackwards(QS, RS, Rw)
    # initialize items for backward evolution
    ht = size(QS, 1)
    ne = size(RS, 1)
    ns = size(QS, 3);
    CS = zeros(ne, ne, ns);
    # warm up C vectors with backward transient Rw matrices
    C = Matrix(1.0I, ne, ne)
    cdelay = size(Rw, 3);
    C = cEvolve(C, Rw, cdelay)
    println("finished warming up C. Size is: ", size(C))
    CS[:, :, ns] = C
    println("set IC for calculating V.")
    for i = ns-1:-1:1
        if Int(rem(i, ns/10)) == 0
            println("percent completion: ", (ns-i)/ns*100);
        end
        CS[:, :, i] = (RS[:, :, i+1])\CS[:, :, i+1]
        # renormalizes C for each vector
        for j=1:ne
            CS[:, j, i] = CS[:, j, i]/norm(CS[:, j, i])
        end
    end
    # return results of backward component of Ginelli's method for CLV computation
    return CS
end

function clvGinelliBackwardsDebug(QS, RS, Rw)
    # initialize items for backward evolution
    ht = size(QS, 1)
    ne = size(RS, 1)
    ns = size(QS, 3);
    CS = zeros(ne, ne, ns);
    cdelay = size(Rw, 3);
    Cw = zeros(ne, ne, cdelay);
    # warm up C vectors with backward transient Rw matrices
    C = Matrix(1.0I, ne, ne)
    Cw[:, :, cdelay] = C;
    # C = cEvolve(C, Rw, cdelay)
    for i = cdelay-1:-1:1
        Cw[:, :, i] = Rw[:, :, i+1]\Cw[:, :, i+1]
        # renormalize Cw
        for j=1:ne
            Cw[:, j, i] = Cw[:, j, i]/norm(Cw[:, j, i])
        end
    end
    println("finished warming up C. Size is: ", size(C))
    CS[:, :, ns] = Cw[:, :, 1]
    println("set IC for calculating V.")
    for i = ns-1:-1:1
        if Int(rem(i, ns/10)) == 0
            println("percent completion: ", (ns-i)/ns*100);
        end
        CS[:, :, i] = (RS[:, :, i+1])\CS[:, :, i+1]
        # renormalizes C for each vector
        for j=1:ne
            CS[:, j, i] = CS[:, j, i]/norm(CS[:, j, i])
        end
    end
    # return results of backward component of Ginelli's method for CLV computation
    return CS, Cw
end

function clvGinelliForwardDebug(flow, jacobian, p, δt, x0, delay, ns, ne, cdelay, nsps)
    # initialize local variables
    ht = length(x0)
    # construct set of perturbation vectors
    delta = zeros(ht, ne)
    for i=1:ne
        delta[i, i] = 1;
    end
    # the number of total steps for the system is the number of samples (ns)
    # * the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice
    x0 = fd1(flow, x0, p, δt, delay)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, ns);
    vn = zeros(ht, ns);
    RS = zeros(ne, ne, ns);
    QS = zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    for i=1:ns
        yS[:, i] = x0;
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        vn[:, i] = v/norm(v);
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*δt*nsps)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    Qw = zeros(ht, ht, cdelay);
    for i=1:cdelay
        # advance delta and x0 by δt
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        Rw[:, :, i] = r
        Qw[:, :, i] = delta
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS, Qw
end

function covariantLyapunovVectorsDebug(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypspecGS, Qw = clvGinelliForwardDebug(flow, jacobian, p, δt, x0,
                                                  delay, ns, ne, cdelay, nsps)
    CS, Cw = clvGinelliBackwardsDebug(QS, RS, Rw)
    lypspecCLV = lyapunovSpectrumCLV(CS, RS, nsps, δt)
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)

    return yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw
end
export covariantLyapunovVectorsDebug
#-----------------------------------------------------------------------------#
# dynamical systems functions

"""
  kyd(spectrum::Array{Float64, 1})
  Determines the Kaplan York Dimesion (kyd) based on an input lyapunov spectrum.
"""
function kyd(spectrum::Array{Float64, 1})::Float64
    ne = length(spectrum);
    tempsum = 0.0;
    KYD = -1.0;
    i = 1;
    KYD_bool = true;
    while ( i <= ne )
      tempsum += spectrum[i];
      if tempsum < 0
        # linear interpolation to determine fractional dimension
        # y = mx + b
        # 0 = (lypspec[i] - lypspec[i-1])/1*dim -(i-1)
        KYD = (i-1) + sum(spectrum[1:(i-1)])/abs(spectrum[i]);
        break
      end
      i += 1;
    end

    if KYD == -1.0
      println("Kaplan York Dimension Does not exist.")
    end

    return KYD
end
export kyd



#-----------------------------------------------------------------------------#
# general purpose functions

# take array of numbers in and return 'histogram' of array
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
    dist /= length(data)
    return boxes, dist
end
export pdf

"""
  sumu!(y, u)

  modifies y array so that the average lattice point value is u.
"""
function sumu!(y, u)
  Random.seed!(10)
  h = size(y, 1)
  # Resets y to a random array
  for i=1:h
    y[i] = rand(Float64)
  end
  # y[1] = u; y[h] = u;
  s = sum(y);
  s = s - u*h;
  s = s/h;
  for i=1:h
    y[i] -= s;
  end
  println("New u of y: ", mean(y))
  # y[h] -= sum(y)
  @assert(abs(sum(y)/h - u) < 0.1^10)
end
export sumu!

"""
  remove_zero_datapoints(x, y)

  Returns two new arrays where y is not zero for any datapoints.
"""
function remove_zero_datapoints(x, y)
    @assert length(x) == length(y)
    lenx = length(x)
    xclean = []; yclean = []
    for i=1:lenx
        if !isapprox(y[i], 0)
            append!(xclean, x[i])
            append!(yclean, y[i])
        end
    end
    return xclean, yclean
end
export remove_zero_datapoints
"""
  zero_index(lypspec::Array{Float64, 1})

  Returns index at which lyapunov spectrum crosses zero.
"""
function zero_index(lypspec::Array{Float64, 1})
    len = length(lypspec)
    val = 1.0
    i = 0;
    while val > 0
      i += 1;
      val = lypspec[i]
    end
    println("index of crossing is at: ", i)
    # Determine linear interpolation between last positive exponent and first
    # negative exponent
    i -= 1;
    a = lypspec[i+1] - lypspec[i];
    b = 1/2*(lypspec[i+1] + lypspec[i] - a*(2*i +1));
    ind_cross = -b/a;
    return ind_cross
end
export zero_index
"""
  angle(v1::Arrray{Float64, 1}, v2::Array{Float64, 1})

  Returns angle between two one dimensional vectors. Assumes equal length.
"""
function angle(v1::Array{Float64, 1}, v2::Array{Float64, 1})
  return acos(round(dot(v1, v2)/(norm(v1)*norm(v2)), digits=14))
end
export angle

end # module
