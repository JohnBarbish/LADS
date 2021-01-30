module LADS

using Statistics, Random, LinearAlgebra
import JLD: jldopen, read, close
using HDF5, ProgressMeter, FFTW
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
export lorenzFlow

function lorenzJacobian(xvec, p)
    sigma = p[1]; rho = p[2]; beta = p[3];
    x = xvec[1]; y = xvec[2]; z = xvec[3];
    J = [[-sigma sigma 0];
         [(rho - z) -1 -x];
         [y x -beta]]
    return J
end
export lorenzJacobian

#-----------------------------------------------------------------------------#
# functions for model A System with Broken Conservation Law
# taken from Grigoriev and Cross 1997.

function modelAMap(y, p)
  # general timestep function for 1D lattice with periodic B.C.
  h = size(y, 1) # number of rows
  x = zeros(h) # copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    x[i] = y[i] + ag(y[i], p) + af(y[i-1], p) + af(y[i+1], p) - 2*af(y[i], p)
  end
  # timestep first site
  x[1] = y[1] + ag(y[1], p) + af(y[h], p) + af(y[2], p) - 2*af(y[1], p)
  # timestep last site
  x[h] = y[h] + ag(y[h], p) + af(y[h-1], p) + af(y[1], p) - 2*af(y[h], p)
  return x
end
export modelAMap

function modelAJacobian(x, p)
  # calculates Jacobian at point x
  h = size(x, 1) # number of rows
  # construct Jacobian
  J = zeros(h, h);
  for i in 2:h-1
    J[i, i-1] = afprime(x[i-1], p)
    J[i, i] = 1 + agprime(x[i], p) - 2*afprime(x[i], p)
    J[i, i+1] = afprime(x[i+1], p)
  end
  J[1, 1] = 1 + agprime(x[1], p) - 2*afprime(x[1], p)
  J[1, h] = afprime(x[h], p)
  J[1, 2] = afprime(x[2], p)
  J[h, h] = 1 + agprime(x[h], p) - 2*afprime(x[h], p)
  J[h, 1] = afprime(x[1], p)
  J[h, h-1] = afprime(x[h-1], p)
  return J
end
export modelAJacobian

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

function ag(x, p)::Float64
    return p[3]*(p[4] - x)^3
end

function agprime(x, p)::Float64
    return -3*p[3]*(p[4] - x)^2
end
#-----------------------------------------------------------------------------#
# functions for tent Map System from Takeuchi

function tentMap(x, p)
    L = length(x);
    mu = p[1];
    K = p[2];
    y = zeros(L);
    y[1] = tentf(x[1], mu) + K/2*(tentf(x[L], mu) - 2*tentf(x[1], mu) + tentf(x[2], mu))
    for i=2:L-1
        y[i] = tentf(x[i], mu) + K/2*(tentf(x[i-1], mu) - 2*tentf(x[i], mu) + tentf(x[i+1], mu))
    end
    y[L] = tentf(x[L], mu) + K/2*(tentf(x[L-1], mu) - 2*tentf(x[L], mu) + tentf(x[1], mu))
    return y
end
export tentMap

function tentf(x, mu)
    return 1 - mu*abs(x);
end

function tentdf(x, mu);
    if x < 0
        return mu
    else
        return -mu
    end
end

function tentJacobian(y, p)
    mu = p[1]; K = p[2];
    L = length(y) # number of rows
    J = zeros(L, L);
    # construct evolution matrix
    for i in 2:L-1
        J[i, i-1] = K/2*tentdf(y[i-1], mu)
        J[i, i] = (1 - K)*tentdf(y[i], mu)
        J[i, i+1] = K/2*tentdf(y[i+1], mu)
    end
    J[1, 1] = (1 - K)*tentdf(y[1], mu)
    J[1, L] = K/2*tentdf(y[L], mu)
    J[1, 2] = K/2*tentdf(y[2], mu)
    J[L, L] = (1 - K)*tentdf(y[L], mu)
    J[L, 1] = K/2*tentdf(y[1], mu)
    J[L, L-1] = K/2*tentdf(y[L-1], mu)
    return J
end
export tentJacobian

#-----------------------------------------------------------------------------#
# functions for tent Map System from Takeuchi with Rigid Boundary Conditions

function tentCMapRBC(x, p)
    L = length(x);
    mu = p[1];
    K = p[2];
    y = zeros(L);
    y[1] = tentf(x[1], mu) + K/2*(tentf(x[L], mu) - 2*tentf(x[1], mu) + tentf(x[2], mu))
    for i=2:L-2
        y[i] = tentf(x[i], mu) + K/2*(tentf(x[i-1], mu) - 2*tentf(x[i], mu) + tentf(x[i+1], mu))
    end
    y[1:L-2] = y[1:L-2] + tentCg(x[1:L-2], p)*ones(L-2);
    y[L-1] = x[L-1]
    y[L] = x[L]
    return y
end
export tentCMapRBC
# in development
function tentCJacobianRBC(y, p)
    mu = p[1]; K = p[2]; C = p[3]; beta = p[4];
    L = length(y) # number of rows
    J = zeros(L, L);
    # useful constants
    a = K/2 - beta/L; c = -beta/L; d = 1 - K-beta/L;
    for i=1:L-2
        J[i, 1:L-2] = c*tentdf.(y[1:L-2], mu);
    end
    # construct evolution matrix
    for i in 2:L-3
        J[i, i-1] = a*tentdf(y[i-1], mu)
        J[i, i]   = d*tentdf(y[i], mu)
        J[i, i+1] = a*tentdf(y[i+1], mu)
    end
    J[1, 1] = d*tentdf(y[1], mu)
    J[1, 2] = a*tentdf(y[2], mu)
    J[L-2, L-2] = d*tentdf(y[L-2], mu)
    J[L-2, L-3] = a*tentdf(y[L-3], mu)
    return J
end
export tentCJacobianRBC
#-----------------------------------------------------------------------------#
# functions for tent Map System with Conservation Law from mean-field-type
# coupling
# parameter definitions: p = (mu, K, C)
function tentCMap(x, p)
    L = length(x);
    mu = p[1];
    K = p[2];
    y = zeros(L);
    y[1] = tentf(x[1], mu) + K/2*(tentf(x[L], mu) - 2*tentf(x[1], mu) + tentf(x[2], mu))
    for i=2:L-1
        y[i] = tentf(x[i], mu) + K/2*(tentf(x[i-1], mu) - 2*tentf(x[i], mu) + tentf(x[i+1], mu))
    end
    y[L] = tentf(x[L], mu) + K/2*(tentf(x[L-1], mu) - 2*tentf(x[L], mu) + tentf(x[1], mu))
    y = y + tentCg(x, p)*ones(L);
    return y
end
export tentCMap
# in development
function tentCJacobian(y, p)
    mu = p[1]; K = p[2]; C = p[3]; beta = p[4];
    L = length(y) # number of rows
    J = zeros(L, L);
    # useful constants
    a = K/2 - beta/L; c = -beta/L; d = 1 - K-beta/L;
    for i=1:L
        J[i, :] = c*tentdf.(y, mu);
    end

    # construct evolution matrix
    for i in 2:L-1
        J[i, i-1] = a*tentdf(y[i-1], mu)
        J[i, i]   = d*tentdf(y[i], mu)
        J[i, i+1] = a*tentdf(y[i+1], mu)
    end
    J[1, 1] = d*tentdf(y[1], mu)
    J[1, L] = a*tentdf(y[L], mu)
    J[1, 2] = a*tentdf(y[2], mu)
    J[L, L] = d*tentdf(y[L], mu)
    J[L, 1] = a*tentdf(y[1], mu)
    J[L, L-1] = a*tentdf(y[L-1], mu)
    return J
end
export tentCJacobian

function tentCg(x, p)
    N = length(x);
    mu = p[1]; K = p[2]; C = p[3]; beta = p[4];
    return beta/N*(C - sum(tentf.(x, mu)))
end
#-----------------------------------------------------------------------------#
# functions useful for calculating CLVs for flows

function advanceQRMap(map, jacobian, x0, delta, p, nsps)
    ht = length(x0);
    v = zeros(ht);
    x00 = x0;
    for i=1:nsps
        # xn = map(x0, p);
        J = jacobian(x0, p);
        # advance delta and x0 by δt
        delta = J*delta
        x0 = map(x0, p); # v*δt + x0
    end
    v = (x0 - x00)/(nsps) # estimate on velocity?
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign.(diag(r))))
    r = sgn*r
    delta = q*sgn

    return x0, v, delta, r
end

function backwardsRC(C1, R1)
    # evolves C1 backwards to C0
    C0 = R1\C1
    for j=1:size(C0, 2)
        C0[:, j] = C0[:, j]/norm(C0[:, j])
    end
    return C0
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
    lypspec /= ((ns-1)*nsps*δt); # don't use 1st timestep in calculation
    return lypspec
end

function lyapunovSpectrumCLVMap(CS, RS, nsps)
    return lyapunovSpectrumCLV(CS, RS, nsps, 1)
end

function lyapunovSpectrumCLVMap(datafile)
    h5open(datafile, "r") do fid
        global lypspec
        cH = fid["c"]; rH = fid["r"]; nsps = read(fid["nsps"]);
        ns =size(cH, 3); ne = size(cH, 1);
        lypspec = zeros(ne);
        C = zeros(ne, ne); R = zeros(ne, ne);
        @showprogress "CLV Exponents " for i = 2:ns
            C = reshape(cH[:, :, i-1], (ne, ne));
            R = reshape(rH[:, :, i], (ne, ne));
            lypspec += log.(clvInstantGrowth(C, R))
        end
    lypspec /= ((ns-1)*nsps)
    end
    return lypspec
end

function lyapunovSpectrumCLV(datafile)
    h5open(datafile, "r") do fid
        global lypspec
        cH = fid["c"]; rH = fid["r"]; nsps = read(fid["nsps"]);
        dt = read(fid["dt"]);
        ns =size(cH, 3); ne = size(cH, 1);
        lypspec = zeros(ne);
        C = zeros(ne, ne); R = zeros(ne, ne);
        @showprogress "CLV Exponents " for i = 2:ns
            C = reshape(cH[:, :, i-1], (ne, ne));
            R = reshape(rH[:, :, i], (ne, ne));
            lypspec += log.(clvInstantGrowth(C, R))
        end
    lypspec /= (ns*nsps*dt)
    end
    return lypspec
end

"""
    clvInstantaneous(datafile::String, tS)
    reads datafile and calculates the instantaneous clv's for each timestep tS,
    assuming a sample exists after each specified element in tS
"""
function clvInstantaneous(datafile::String, tS)
    h5open(datafile, "r") do fid
        ns = length(tS); # number of samples to store
        cH = fid["c"]; rH = fid["r"]; nsps = read(fid["nsps"]);
        ne = size(cH, 1);
        global clvInst = zeros(ne, ns);
        C = zeros(ne, ne); R = zeros(ne, ne);
        for (i, t) in enumerate(tS)
            C = reshape(cH[:, :, t], (ne, ne));
            R = reshape(rH[:, :, t+1], (ne, ne));
            clvInst[:, t] = log.(clvInstantGrowth(C, R))/nsps
        end
    end
    return clvInst
end
export clvInstantaneous

function clvGinelliBackwards(QS, RS, Rw)
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
    @showprogress "Warmup Completion " for i = cdelay-1:-1:1
        C = backwardsRC(C, Rw[:, :, i+1]);
        Cw[:, :, i] = C;
    end
    C = backwardsRC(Cw[:, :, 1], Rw[:, :, 1]);
    CS[:, :, end] = C;
    println("set IC for calculating V.")
    @showprogress "Sample Completion " for i = ns-1:-1:1
        C = backwardsRC(C, RS[:, :, i+1]);
        CS[:, :, i] = C;
    end
    # return results of backward component of Ginelli's method for CLV computation
    return CS, Cw
end

function clvGinelliMapForward(map, jacobian, p, x0, delay, ns, ne, cdelay, nsps)
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
    lambdaInst = zeros(ne, delay)
    @showprogress "Delay Completed " for i=1:delay
        x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        lambdaInst[:, i] = log.(diag(r))/(nsps)
    end
    # println(delta)
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
    @showprogress "Sample Calculations Completed " for i=1:ns
        yS[:, i] = x0;
        x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        vn[:, i] = v/norm(v);
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*nsps)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    Qw = zeros(ht, ne, cdelay);
    @showprogress "Forward Warmup Completed " for i=1:cdelay
        # advance delta and x0 by δt
        x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        Rw[:, :, i] = r
        Qw[:, :, i] = delta
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
end

# using HDF5
function clvGinelliLongMapForward(map, jacobian, p, x0, delay, ns, ne, cdelay, nsps, nsim, filename)
    fid = h5open(filename, "w");
    # initialize local variables
    ht = length(x0)
    delayResets = Int(delay/nsim)
    nsResets = Int(ns/nsim)
    cdelayResets = Int(cdelay/nsim)
    # create data handles for storing the data and allocate the necesary
    cH = d_create(fid, "c", datatype(Float64), dataspace(ne, ne, ns))
    rH = d_create(fid, "r", datatype(Float64), dataspace(ne, ne, ns))
    qH = d_create(fid, "q", datatype(Float64), dataspace(ht, ne, ns))
 #     jH = d_create(fid, "J", datatype(Float64), dataspace(ht, ht, ns),
 #                      "chunk", (ht, ht, nsrs))
    yH = d_create(fid, "y", datatype(Float64), dataspace(ht, ns))
    rwH = d_create(fid, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
    cwH = d_create(fid, "cw", datatype(Float64), dataspace(ne, ne, cdelay))
    λH = d_create(fid, "lambdaInst", datatype(Float64), dataspace(ne, delay))
    # store parameters of the run
    fid["parameters"] = p;
    fid["delay"] = delay;
    fid["ns"] = ns;
    fid["ne"] = ne;
    fid["cdelay"] = cdelay;
    fid["nsps"] = nsps;
    fid["nsim"] = nsim;
    println("nsim is recorded as : ", nsim);
    fid["ht"] = ht;
    # construct set of perturbation vectors
    delta = zeros(ht, ne)
    for i=1:ne
        delta[i, i] = 1;
    end
    # the number of total steps for the system is the number of samples (ns)
    # * the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice and Gram-Schmidt Vectors
    λi = zeros(ne, nsim) # zeros(ne, delayResets)
    @showprogress "Delay Completed " for i=1:delayResets
        for j=1:nsim
            x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            λi[:, j] = log.(diag(r))/(nsps)
        end
        λH[:, range((i-1)*nsim+1, length=nsim)] = λi
    end
    # println(delta)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, nsim); # zeros(ht, ns);
    vn = zeros(ht, nsim); # zeros(ht, ns);
    RS = zeros(ne, ne, nsim); # zeros(ne, ne, ns);
    QS = zeros(ht, ne, nsim); # zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress "Sample Calculations Completed " for i=1:nsResets
        for j=1:nsim
            yS[:, j] = x0;
            x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            # vn[:, j] = v/norm(v);
            RS[:, :, j] = r
            QS[:, :, j] = delta
            lypspecGS += log.(diag(r))/(ns*nsps)
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        yH[:, trng] = yS;
        # vn[:, trng] = vn;
        rH[:, :, trng] = RS;
        qH[:, :, trng] = QS;
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    Qw = zeros(ht, ne, nsim); # zeros(ht, ne, cdelay);
    @showprogress "Forward Warmup Completed " for i=1:cdelayResets
        for j=1:nsim
            x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            RS[:, :, j] = r
            Qw[:, :, j] = delta
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        rwH[:, :, trng] = RS;
        # qH[:, :, trng] = Qw;
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # save GS method lyapunov spectrum
    fid["lypGS"] = lypspecGS;
    # return results of forward component of Ginelli's method for CLV computation
    # return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
    close(fid)
end

function clvGinelliLongForward(flow, jacobian, p, δt, x0, delay, ns, ne, cdelay, nsps, nsim, filename)
    fid = h5open(filename, "w");
    # initialize local variables
    ht = length(x0)
    delayResets = Int(delay/nsim)
    nsResets = Int(ns/nsim)
    cdelayResets = Int(cdelay/nsim)
    # create data handles for storing the data and allocate the necesary
    cH = d_create(fid, "c", datatype(Float64), dataspace(ne, ne, ns))
    rH = d_create(fid, "r", datatype(Float64), dataspace(ne, ne, ns))
    qH = d_create(fid, "q",datatype(Float64), dataspace(ht, ne, ns))
 #     jH = d_create(fid, "J", datatype(Float64), dataspace(ht, ht, ns),
 #                      "chunk", (ht, ht, nsrs))
    yH = d_create(fid, "y", datatype(Float64), dataspace(ht, ns))
    rwH = d_create(fid, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
    cwH = d_create(fid, "cw", datatype(Float64), dataspace(ne, ne, cdelay))
    λH = d_create(fid, "lambdaInst", datatype(Float64), dataspace(ne, delay))
    # store parameters of the run
    fid["parameters"] = p;
    fid["delay"] = delay;
    fid["ns"] = ns;
    fid["ne"] = ne;
    fid["dt"] = δt;
    fid["cdelay"] = cdelay;
    fid["nsps"] = nsps;
    fid["nsim"] = nsim;
    # println("nsim is recorded as : ", nsim);
    fid["ht"] = ht;
    # construct set of perturbation vectors
    delta = zeros(ht, ne)
    for i=1:ne
        delta[i, i] = 1;
    end
    # the number of total steps for the system is the number of samples (ns)
    # * the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice and Gram-Schmidt Vectors
    λi = zeros(ne, nsim) # zeros(ne, delayResets)
    @showprogress "Delay Completed " for i=1:delayResets
        for j=1:nsim
            x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
            λi[:, j] = log.(diag(r))/(nsps*δt)
        end
        λH[:, range((i-1)*nsim+1, length=nsim)] = λi
    end
    # println(delta)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, nsim); # zeros(ht, ns);
    vn = zeros(ht, nsim); # zeros(ht, ns);
    RS = zeros(ne, ne, nsim); # zeros(ne, ne, ns);
    QS = zeros(ht, ne, nsim); # zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress "Sample Calculations Completed " for i=1:nsResets
        for j=1:nsim
            yS[:, j] = x0;
            x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
            # vn[:, j] = v/norm(v);
            RS[:, :, j] = r
            QS[:, :, j] = delta
            lypspecGS += log.(diag(r))/(ns*nsps*δt)
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        yH[:, trng] = yS;
        # vn[:, trng] = vn;
        rH[:, :, trng] = RS;
        qH[:, :, trng] = QS;
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    Qw = zeros(ht, ne, nsim); # zeros(ht, ne, cdelay);
    @showprogress "Forward Warmup Completed " for i=1:cdelayResets
        for j=1:nsim
            x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
            RS[:, :, j] = r
            Qw[:, :, j] = delta
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        rwH[:, :, trng] = RS;
        # qH[:, :, trng] = Qw;
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # save GS method lyapunov spectrum
    fid["lypGS"] = lypspecGS;
    # return results of forward component of Ginelli's method for CLV computation
    # return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
    close(fid)
end

function clvGinelliLongBackwards(datafile, keepCLVWarmup)
    # initialize items for backward evolution
    fid = h5open(datafile, "r+")
    nsim = read(fid["nsim"]);
    println("nsim in file is: ", nsim)
    delay = read(fid["delay"]);
    ns = read(fid["ns"]);
    cdelay = read(fid["cdelay"]);
    # create local variables for reading and writing into batches
    delayResets = Int(delay/nsim)
    nsResets = Int(ns/nsim)
    cdelayResets = Int(cdelay/nsim)
    # create handles to data stored in file
    qH = fid["q"]; rH = fid["r"]; yH = fid["y"]; cH = fid["c"];
    rwH = fid["rw"]; cwH = fid["cw"];
    ht = size(yH, 1); ne = size(rH, 1); ns = size(qH, 3);
    # CS = zeros(ne, ne, ns);
    # cdelay = size(rwH, 3);
    # warm up C vectors with backward transient Rw matrices
    Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # RS = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    C = Matrix(1.0I, ne, ne);
    # C0 = copy(C1); # zeros(ne, ne);
    # cwH[:,:, end] = C1;
    # CS[:, :, end] = C1;
    # warms up C matrix
    @showprogress "Warmup Completion " for i = cdelayResets:-1:1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Warmup Completion:\t $(round((cdelayResets-i)/cdelayResets*100))%,\t Range:\t $trng")
        Rw = rwH[:, :, trng];
        # sets IC of CS
        Cw[:, :, end] = C;
        for j=nsim-1:-1:1
            C = backwardsRC(C, Rw[:, :, j+1]);
            Cw[:, :, j] = C;
        end
        # assign final value and prepare for next set of data
        # Cw[:, :, 1] = C;
        C = backwardsRC(C, Rw[:, :, 1])
        # assign values to dataset
        cwH[:, :, trng] = Cw;
    end

    println("Warmup Completion:\t 100%")
    # assign C from last warm up data to first recorded data
    # C1 = C;
    cH[:, :, end] = C
    println("set IC for calculating V.")
    # Resize Cw and Rw arrays for datarecording
    # Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # #Rw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # C1 = Matrix(1.0I, ne, ne);
    # C0 = zeros(ne, ne);
    # cwH[:,:, end] = C1;
    @showprogress "Sample Completion " for i = nsResets:(-1):1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Completion:\t $(round((nsResets-i)/nsResets*100))%,\t Range:\t $trng")
        RS = rH[:, :, trng];
        Cw[:, :, end] = C;
        for j=nsim-1:-1:1
            C = backwardsRC(C, RS[:, :, j+1]);
            Cw[:, :, j] = C;
        end
        # assign final value and prepare for next set of data
        # Cw[:, :, 1] = C;
        C = backwardsRC(C, RS[:, :, 1])
        # assign values to dataset
        cH[:, :, trng] = Cw;
    end
    # implementing lazy delete, could revise data storage to more efficiently
    # overwrite variables instead of creating new ones
    if !keepCLVWarmup
        o_delete(fid, "rw")
        o_delete(fid, "cw")
    end
    # close file
    close(fid)
    # return results of backward component of Ginelli's method for CLV computation
    # return CS, Cw
end

function covariantLyapunovVectorsMap(map, jacobian, p, x0, delay::Int64,
            ns::Int64, ne::Int64, cdelay::Int64, nsps::Int64, nsim::Int64,
            filename, keepCLVWarmup=false)
    clvGinelliLongMapForward(map, jacobian, p, x0, delay, ns, ne, cdelay, nsps,
                                nsim, filename)
    clvGinelliLongBackwards(filename, keepCLVWarmup)
    lypspecCLV = lyapunovSpectrumCLVMap(filename) # , nsim)
    h5open(filename, "r+") do fid
        write(fid, "lypspecCLV", lypspecCLV)
    end
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)
#     return yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw, lambdaInst
end

function covariantLyapunovVectorsMap(map, jacobian, p, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst = clvGinelliMapForward(map,
                                                  jacobian, p, x0,
                                                  delay, ns, ne, cdelay, nsps)
    CS, Cw = clvGinelliBackwards(QS, RS, Rw)
    lypspecCLV = lyapunovSpectrumCLVMap(CS, RS, nsps)
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)

    return yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw, lambdaInst, Rw
end
export covariantLyapunovVectorsMap

function covariantLyapunovVectors(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst = clvGinelliForward(flow, jacobian,
                                                                    p, δt, x0,
                                                  delay, ns, ne, cdelay, nsps)
    CS, Cw = clvGinelliBackwards(QS, RS, Rw)
    lypspecCLV = lyapunovSpectrumCLV(CS, RS, nsps, δt)
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)

    return yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw, lambdaInst, Rw
end


function covariantLyapunovVectors(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps, nsim::Int64, filename, keepCLVWarmup=false)
    clvGinelliLongForward(flow, jacobian, p, δt, x0, delay, ns, ne,
                            cdelay, nsps, nsim, filename)
    clvGinelliLongBackwards(filename, keepCLVWarmup)
    lypspecCLV = lyapunovSpectrumCLV(filename) # , nsim)
    h5open(filename, "r+") do fid
        write(fid, "lypspecCLV", lypspecCLV)
    end
    println("CLV Lyapunov Spectrum: ")
    println(lypspecCLV)
end
export covariantLyapunovVectors
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
    # warm up the lattice and Gram-Schmidt Vectors
    # x0 = fd1(flow, x0, p, δt, delay)
    lambdaInst = zeros(ne, delay)
    @showprogress "Delay Completed " for i=1:delay
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        lambdaInst[:, i] = log.(diag(r))/(nsps*δt)
    end
    # println(delta)
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
    @showprogress "Sample Calculations Completed " for i=1:ns
        yS[:, i] = x0;
        x0, v, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        vn[:, i] = v/norm(v);
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*nsps*δt)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    Qw = zeros(ht, ne, cdelay);
    @showprogress "Forward Warmup Completed " for i=1:cdelay
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
    return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
end


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


"""
lyapunovSpectrumGSMap(map, jacobian, p, x0, delay, ns, ne, nsps)

Returns the Lyapunov Spectrum using the Gram-Schmidt Method for computing the
exponents.

"""
function lyapunovSpectrumGSMap(map, jacobian, p, x0, delay, ns, ne, nsps; saverunavg=false)
    # initialize local variables
    ht = length(x0)
    lattice = copy(x0)
    lypspecGS = zeros(ne) # will contain lyapunov spectrum to be returned
    delta = zeros(ht, ne) # matrix of the number of exponents (ne) to determine
    # initializes CLVs to orthonormal set of vectors
    for i=1:ne
      delta[i, i] = 1;
    end

    # the number of total steps for the system is the number of samples (ns)
    # times the number of steps per sample (nsps)
    numsteps = ns*nsps
    # warm up the lattice (x0) and perturbations (delta)
    @showprogress "Delay Completed " for i in 1:delay
        x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        # timestep(lattice)
    end
    println("lattice warmed up, starting GSV evolution.")
    # calculate Lyapunov Spectrum for the given number samples (ns) and
    # given spacing (nsps)
    if saverunavg
        lypspecGS = zeros(ne, ns)
        lsGSravg = zeros(ne)
        @showprogress "Sample Calculations Completed " for i=1:ns
        # for i=1:ns
            x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            lsGSravg += log.(diag(r))
            lypspecGS[:, i] = lsGSravg/(i*nsps)
        end
        println("the GSV Lyapunov Spectrum is:")
        println(lypspecGS[:, end])
    else
        @showprogress "Sample Calculations Completed " for i=1:ns
            x0, v, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            lypspecGS += log.(diag(r))/(ns*nsps)
        end
        println("the GSV Lyapunov Spectrum is:")
        println(lypspecGS)
    end
    # finished evolution of lattice
    return lypspecGS
end
export lyapunovSpectrumGSMap
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
# functions useful for calculating Domination of Osledet Splitting
"""
  DOS(datafile::String, window::Int64)
Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(datafile::String, window::Int64) # ::HDF5.HDF5File) problem with
    # HDF5 type currently, need to fix
    fid = h5open(datafile, "r")
    rH = fid["r"];
    cH = fid["c"];
    ns = size(rH, 3);
    ne = size(cH, 1);
    nu = zeros(ne, ne);
    nutemp = zeros(ne, ne);
    CLV_growth = zeros(ne);
    # @assert(window < ns && window > 0) # checks that window is not too large
    @showprogress "Calculating DOS " for i=1:ns-window
        # added for loop to go have the instantaneous CLV growth averaged over window
        CLV_growth = zeros(ne);
        for j=0:window-1
            C1 = reshape(cH[:, :, i+j], (ne, ne));
            R2 = reshape(rH[:, :, i+1+j], (ne, ne));
            CLV_growth += LADS.clvInstantGrowth(C1, R2);
        end
        CLV_growth /= window;
        nutemp = dosViolations(CLV_growth)
        nu += nutemp;
    end
    close(fid)
    nu /= (ns-window)
    return nu
end

"""
  DOS(datafile::String, windows)
Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(datafile::String, windows)
    fid = h5open(datafile, "r")
    rH = fid["r"];
    cH = fid["c"];
    ns = size(rH, 3);
    ne = size(cH, 1);
    nu = zeros(ne, ne);
    nuDict = Dict();
    nutemp = zeros(ne, ne);
    CLV_growth = zeros(ne);
    # @assert(window < ns && window > 0) # checks that window is not too large
    mktemp() do path, io
        fid = h5open(path, "w")
        growth = d_create(fid, "growth", datatype(Float64), dataspace(ne, ns))
        @showprogress "Instant Growths " for i = 1:ns-1
            C1 = reshape(cH[:, :, i], (ne, ne));
            R2 = reshape(rH[:, :, i+1], (ne, ne));
            fid["growth"][:, i] = LADS.clvInstantGrowth(C1, R2);
        end
        for window in windows
            nu = zeros(ne, ne);
            if window > 3
                # new process, saves computation for larger window sizes
                clvGrowth = zeros(ne);
                for j=1:window
                   clvGrowth += reshape(fid["growth"][:, j], ne);
                end
                clvGrowth /= window
                nutemp = dosViolations(clvGrowth)
                @showprogress "Window: $window " for i = 2:ns-window
                   clvGrowth -= reshape(fid["growth"][:, i-1], ne)/window;
                   clvGrowth += reshape(fid["growth"][:, i+window-1], ne)/window;
                   nutemp += dosViolations(clvGrowth);
                end
                nu = nutemp/(ns-window);
                nuDict[window] = nu;
            else
                # regular process
                @showprogress "Window: $window " for i=1:ns-window
                    # added for loop to go have the instantaneous CLV growth
                    # averaged over window
                    CLV_growth = zeros(ne);
                    for j=0:window-1
                        CLV_growth += reshape(fid["growth"][:, i+j], ne);
                    end
                    CLV_growth /= window;
                    nutemp = dosViolations(CLV_growth)
                    nu += nutemp;
                end
                nu /= (ns-window)
                nuDict[window] = nu;
            end

        end
    end
    return nuDict
end
"""
  DOS(RS, CS, window::Int64)
Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(RS, CS, window::Int64) # ::HDF5.HDF5File) problem with HDF5 type
    ## currently, need to fix
  # fid = h5open(datafile, "r")
  #   Rhandle = fid["R"];
  #   Chandle = fid["C"];
  ns = size(RS, 3);
  ne = size(CS, 1);
  nu = zeros(ne, ne);
  nutemp = zeros(ne, ne);
  CLV_growth = zeros(ne);
  # @assert(window < ns && window > 0) # checks that window is not too large
  @showprogress "Calculating DOS " for i=1:ns-window
    # added for loop to go have the instantaneous CLV growth averaged over window
    CLV_growth = zeros(ne);
    for j=0:window-1
      C1 = CS[:, :, i+j]
      R2 = RS[:, :, i+1+j]
      CLV_growth += LADS.clvInstantGrowth(C1, R2);
    end
    CLV_growth /= window;
    nutemp = dosViolations(CLV_growth)
    nu += nutemp;
  end
  nu /= (ns-window)
  return nu
end
export DOS

"""
  DOS(datafile::String, windows)
Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(RS, CS, windows)
    ns = size(RS, 3);
    ne = size(CS, 1);
    nu = zeros(ne, ne);
    nutemp = zeros(ne, ne);
    CLV_growth = zeros(ne);
    nuDict = Dict();
    # @assert(window < ns && window > 0) # checks that window is not too large
    mktemp() do path, io
        fid = h5open(path, "w")
        growth = d_create(fid, "growth", datatype(Float64), dataspace(ne, ns))
        @showprogress "Instant Growths " for i = 1:ns-1
            C1 = CS[:, :, i]
            R2 = RS[:, :, i+1]
            fid["growth"][:, i] = LADS.clvInstantGrowth(C1, R2);
        end
        for window in windows
            nu = zeros(ne, ne);
            if window > 3
                # new process, saves computation for larger window sizes
                clvGrowth = zeros(ne);
                for j=1:window
                   clvGrowth += reshape(fid["growth"][:, j], ne);
                end
                clvGrowth /= window
                nutemp = dosViolations(clvGrowth)
                @showprogress "Window: $window " for i = 2:ns-window
                   clvGrowth -= reshape(fid["growth"][:, i-1], ne)/window;
                   clvGrowth += reshape(fid["growth"][:, i+window-1], ne)/window;
                   nutemp += dosViolations(clvGrowth);
                end
                nu = nutemp/(ns-window);
                nuDict[window] = nu;
            else
                # regular process
                @showprogress "Window: $window " for i=1:ns-window
                    # added for loop to go have the instantaneous CLV growth
                    # averaged over window
                    CLV_growth = zeros(ne);
                    for j=0:window-1
                        CLV_growth += reshape(fid["growth"][:, i+j], ne);
                    end
                    CLV_growth /= window;
                    nutemp = dosViolations(CLV_growth)
                    nu += nutemp;
                end
                nu /= (ns-window)
                nuDict[window] = nu;
            end

        end
    end
    return nuDict
end


"""
  dosViolations(CLV_growth::Array{Float64, 1})
Determines nu matrix of total violations from list of instantaneous growth.
"""
function dosViolations(CLV_growth::Array{Float64, 1})
  ne = size(CLV_growth, 1);
  nu = Matrix(1.0I, ne, ne);
  # for i=1:ne
  #   nu[i, i] = 1;
  # end
  for i=1:ne
    for j=1:i-1
      if CLV_growth[i] > CLV_growth[j]
        nu[i, j] = 1; nu[j, i] = 1; # 1;
      else
        nu[i, j] = 0; nu[j, i] = 0; # 0;
      end
    end
  end
  return nu
end


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

function minimumManifoldAngleRange(V)
    ne = size(V, 2);
    Q1 = Array(qr(V).Q); Q2 = Array(qr(reverse(V, dims=2)).Q);
    angles = zeros(ne);
    angles[1] = acos(round(svdvals(reshape(Q1[:, 1]'Q2[:, 1:ne-1], (:, 1)))[1], digits=14));
    angles[ne] = acos(round(svdvals(reshape(Q1[:, 1:ne-1]'Q2[:, 1], (:, 1)))[1], digits=14));
    for i=2:ne-1
        angles[i] = acos(round(svdvals(Q1[:, 1:i]'Q2[:, 1:ne-i])[1], digits=14));
    end
    return angles
end

#-----------------------------------------------------------------------------#
# general purpose functions
"""
  powerSpectrum(s)
  returns the wavenumber distribution along with the power spectrum for the input
  signal array s.
"""
function powerSpectrum(s, dim)
   N = size(s, dim);
   psdx = abs2.(rfft(s, dim));
   psdx ./= sum(abs2.(psdx), dims=dim);
   krng = 0:Int(floor(N/2)); # using floor for odd cases
   return krng, psdx
end
export powerSpectrum
"""
  powerSpectralDensity(s, Fs)
  returns the wavenumber distribution along with the power spectrum for the input
  signal array s. dims specifies the dimension(s) to perform the fft over.
  For example, if the 1st dimension contains the spatial information for the CLV
  specified by the 2nd dimension at the timestep specified by the 3rd, then
  setting dims=1 finds the power spectrum for every CLV and timestep in the 2nd
  and 3rd dimensions and returns it in the same format as the input.
"""
function powerSpectralDensity(s, Fs)
   N = length(s);
   # xdft = fft(s);
   # xdft = xdft[1:Int(N/2)+1];
   xdft = rfft(s);
   psdx = (1/(Fs*N))*abs2.(xdft);
   psdx[2:end-1] = 2*psdx[2:end-1];
   freq = 0:Fs/N:Fs/2;
   return freq, psdx
end

function powerSpectralDensity(s, Fs, dims)
    psd(sig) = powerSpectralDensity(sig, Fs)
    return mapslices(psd, s, dims=dims)
end


export powerSpectralDensity
"""
  averagePowerSpectrum(datafile, nsim, fftdim, timedim)
"""
function averagePowerSpectrum(datafile::String)
    # calculate minimum principal angle of specified subspaces in C
    # initializes theta matrix to contain the mimimum angle
    global krng, psdx
    h5open(datafile, "r") do fid
        global krng, psdx
        qH = fid["q"]; cH = fid["c"];
        ht = size(qH, 1); ns = size(cH, 3); ne = size(cH, 1);
        # nsResets = Int(ns/nsim);
        # cTemp = zeros(ne, ne, nsim); qTemp = zeros(ht, ne, nsim);
        qTemp = zeros(ht, ne); cTemp = zeros(ne, ne);
        vTemp = zeros(ht, ne);
        krng, w = powerSpectrum(vTemp, 1); # assumes dx=1
        nk = length(krng);
        psdx = zeros(nk, ht);
        @showprogress "Power Spectrum " for t=1:ns # t=1:nsResets
            qTemp = reshape(qH[:, :, t], (ht, ne));
            cTemp = reshape(cH[:, :, t], (ne, ne));
            vTemp = qTemp*cTemp;
            krng, psdxTemp = powerSpectrum(vTemp, 1); # assumes dx=1
            psdx += psdxTemp
            # trng = range((t-1)*nsim+1, length=nsim)
            # cTemp = cH[:, :, trng]; qTemp = qH[:, :, trng];
            # for t in trng
            #     vTemp = qTemp[:, :, t]*cTemp[:, :, t]
            #     krng, psdxTemp = powerSpectrum(vTemp, 1)
            #     psdx += psdxTemp
            # end
        end
        psdx /= ns
    end
    return krng, psdx
end
export averagePowerSpectrum

function cumulativeAverage(x::Array{Float64, 1})
    ht = length(x)
    xavg = zeros(ht)
    xavg[1] = x[1];
    for i=2:ht
        xavg[i] = (xavg[i-1]*(i-1) + x[i])/i
    end
    return xavg
end
export cumulativeAverage

function binSelelector(datum, mn, mx, nbins)
    if !isapprox(datum, mx*nbins)
        return floor(Int, div(datum, mx)) + 1
    else
        return nbins
    end
end
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
  zeroIndex(lypspec::Array{Float64, 1})

  Returns index at which lyapunov spectrum crosses zero.
"""
function zeroIndex(lypspec::Array{Float64, 1})
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
export zeroIndex
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
    data = zeros(size(C));
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
    cH = fid["c"];
    ne = size(cH, 1); ns = size(cH, 3);
    mn, mx, nbins = theta[1], theta[end], length(theta);
    nsResets = Int(ns/nsim);
    data = zeros(ne, ne, nbins);
    count = zeros(ne, ne, nbins);
    boxWidth = (mx - mn)/nbins
    @showprogress "Calculating Angles " for t = 1:nsResets
        trng = range((t-1)*nsim+1, length=nsim)
        data = allAngles(cH[:, :, trng], normalized)
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

# functions for calculating multiple manifold angles simultaneously
function multiMinimumManifoldAngle(V, rng)
    ne = size(V, 2);
    Q1 = Array(qr(V).Q); Q2 = Array(qr(reverse(V, dims=2)).Q);
    angles = zeros(length(rng));
    # angles = zeros(ne);
    # angles[1] = acos(round(svdvals(reshape(Q1[:, 1]'Q2[:, 1:ne-1], (:, 1)))[1], digits=14));
    # angles[ne] = acos(round(svdvals(reshape(Q1[:, 1:ne-1]'Q2[:, 1], (:, 1)))[1], digits=14));
    for (ind, i) in enumerate(rng) # 2:ne-1
        angles[ind] = acos(round(svdvals(Q1[:, 1:i]'Q2[:, 1:ne-i])[1], digits=14));
    end
    return angles
end

# small difference between the two methods, with the first slightly faster.
# Let's check how it works when dealing with the large datasets
function multiMinimumManifoldAngle(datafile::String, rng)
    # calculate minimum principal angle of specified subspaces in C
    # initializes theta matrix to contain the mimimum angle
    fid = h5open(datafile, "r")
    # rH = fid["r"];
    cH = fid["c"];
    ns = size(cH, 3);
    ne = size(cH, 1);
    ntheta = length(rng);
    θ = zeros(ntheta, ns)
    cTemp = zeros(ne, ne);
    @showprogress "Manifold Angle " for t=1:ns
    # for t=1:ns
        cTemp = reshape(cH[:, :, t], (ne, ne));
        θ[:, t] = multiMinimumManifoldAngle(cTemp, rng)
    end
    close(fid)
    # return theta array containing minimum principal angle at each timestep
    return θ
end

export multiMinimumManifoldAngle
"""
    clvfromQR(datafile, storefile)

    Calculates the CLVs from Q & R in datafile and stores CLVs ining in another file
"""
function clvfromQR(datafile, storefile)
    # open up read file for data
    h5open(datafile, "r") do fid
        cH = fid["c"]; qH = fid["q"];
        ht = size(qH, 1);
        ns = size(cH, 3);
        ne = size(cH, 1);
        # open write file for clv data
        fidw = h5open(storefile, "w")
        vH = d_create(fidw, "v",datatype(Float64), dataspace(ht, ne, ns));
        clv_loc = zeros(ne);
        @showprogress "Calculating CLVs" 10 for i = 1:ns
            Ci = cH[:, :, i];
            Qi = qH[:, :, i];
            Ci = reshape(Ci, (ne, ne));
            Qi = reshape(Qi, (ht, ne));
            # calculate and store clv at each timestep
            Vi = Qi*Ci;
            vH[:, :, i] = Vi;
            # running total of localization
            clv_loc += [localization(Vi[i, :]) for i in 1:ne];
        end
        # average CLV localization and stored
        clv_loc /= ns;
        fidw["clv_localization"] = clv_loc;
        close(fidw);
    end
end

export clvfromQR

"""
    localization(v, normalized=true)

    Returns localization of vector based on the definition as outlined in
    K. Kaneko, "Lyapunov analysis and information flow in coupled map lattices",
    Physica D, vol. 23(1-3), 436-447, 1986.

"""
function localization(v, normalized=true)
    if !normalized
        return norm(v, 4)/norm(v)
    else
        return norm(v, 4)
    end
end

export localization

function averageLocalization(datafile::String, storefile::String, nsim)
    # nsim = 10000;
    fid = h5open(datafile, "r");
    cH = fid["c"];
    qH = fid["q"];
    ht = size(qH, 1); ne = size(cH, 1); ns = size(cH, 3);
    locCLV = zeros(ne);
    Ci = zeros(ne, ne); Qi = zeros(ht, ne); VS = zeros(ht, ne);
    CS = zeros(ne, ne, nsim); QS = zeros(ht, ne, nsim);
    nsResets = Int(ns/nsim);
    @showprogress "Calculating Localization " for t1 = 1:nsResets
        trng = range((t1-1)*nsim+1, length=nsim);
        CS = cH[:, :, trng]; QS = qH[:, :, trng];
        for t = 1:nsim
            Ci = reshape(CS[:, :, t], (ne, ne));
            Qi = reshape(QS[:, :, t], (ht, ne));
            VS = Qi*Ci;
            for i=1:ne
                locCLV[i] += LADS.localization(VS[:, i], true)
            end
        end
    end
    locCLV = locCLV/ns;
    close(fid)
    h5open(storefile, "cw") do file
        write(file, "locCLV", locCLV)
    end
    return locCLV
end


end # module
