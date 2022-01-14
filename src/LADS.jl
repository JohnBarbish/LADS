module LADS

using Statistics, Random, LinearAlgebra
import JLD: jldopen, read, close
using HDF5, ProgressMeter, FFTW
# We make the follow definitions for notation
# Maps  - discrete time
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
# functions useful for calculating CLVs for flows

function advanceQRMap(map, jacobian, x0, delta, p, nsps)
    for i=1:nsps
        # xn = map(x0, p);
        J = jacobian(x0, p);
        # advance delta and x0 by δt
        delta = J*delta
        x0 = map(x0, p); # v*δt + x0
    end
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign.(diag(r))))
    r = sgn*r
    delta = q*sgn

    return x0, delta, r
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

function clvInstantGrowth(jacobian, x, p, v1)
    # map CLVs forward
    v2 = jacobian(x, p)*v1;
    # find the growth as the magnitudes of v2/v1
    clv_growth = mapslices(v-> norm(v), v2, dims=1)./mapslices(v-> norm(v), v1, dims=1);
    return reshape(clv_growth, :)
  end

function lyapunovSpectrumCLV(jacobian, p, yS, VS, nsps, δt)
    ns =size(VS, 3);
    ne = size(VS, 2);
    lypspec = zeros(ne);
    for t = 1:ns
        lypspec += log.(clvInstantGrowth(jacobian, yS[:, t], p, VS[:, :, t]))
    end
    lypspec /= (ns*nsps*δt); # don't use 1st timestep in calculation
    return lypspec
end

function lyapunovSpectrumCLVMap(jacobian, p, yS, VS, nsps)
    return lyapunovSpectrumCLV(jacobian, p, yS, VS, nsps, 1)
end

function lyapunovSpectrumCLVMap(jacobian, datafile; saverunavg=false, tstart=0, tend=0)
    h5open(datafile, "r") do fid
        global lypspec
        vH = fid["v"]; yH = fid["y"]; nsps = read(fid["nsps"]); p = read(fid["p"]);
        ne = size(vH, 2);
        if tstart == 0 || tend == 0
            # calculate for all times
            ns = size(vH, 3);
            ts = 1:ns;
        else
            # calculate for specified samples
            ns = Int(tend - tstart);
            ts = tstart:tend;
        end
        if saverunavg
            # save running avereage
            lypspec = zeros(ne, ns);
            lypspectmp = zeros(ne);
            @showprogress 10 "CLV Exponents " for (i, t) in enumerate(ts)
                lypspectmp += log.(clvInstantGrowth(jacobian, yH[:, t], p, vH[:, :, t]));
                lypspec[:, i] = lypspectmp/((i)*nsps);
            end
        else
            # just calculate value at end
            lypspec = zeros(ne);
            @showprogress 10 "CLV Exponents " for (i, t) in enumerate(ts)
                lypspec += log.(clvInstantGrowth(jacobian, yH[:, t], p, vH[:, :, t]));
            end
            lypspec /= ((ns)*nsps)
        end
    end
    return lypspec
end
export lyapunovSpectrumCLVMap

function lyapunovSpectrumCLV(jacobian, datafile; saverunavg=false, tstart=0, tend=0)
    h5open(datafile, "r") do fid
        global lypspec
        vH = fid["v"]; yH = fid["y"]; nsps = read(fid["nsps"]); p = read(fid["p"]);
        ne = size(vH, 2);
        dt = read(fid["dt"]);
        if tstart == 0 || tend == 0
            # calculate for all times
            ns = size(cH, 3);
            ts = 1:ns;
        else
            # calculate for specified samples
            ns = Int(tend - tstart);
            ts = tstart+1:tend;
        end
        if saverunavg
            # save runnning average
            lypspectmp = zeros(ne);
            lypspec = zeros(ne, ns);
            @showprogress 10 "CLV Exponents " for (i, t) in enumerate(ts)
                lypspectmp += log.(clvInstantGrowth(jacobian, yH[:, t], p, vH[:, :, t]))
                lypspec[:, i] = lypspectmp/(i*nsps*dt);
            end
        else
            # otehrwise just calc value at end
            lypspec = zeros(ne);
            @showprogress 10 "CLV Exponents " for (i, t) in enumerate(ts)
                lypspec += log.(clvInstantGrowth(jacobian, yH[:, t], p, vH[:, :, t]))
            end
            lypspec /= (ns*nsps*dt)
        end
    end
    return lypspec
end
export lyapunovSpectrumCLV

# """
#     clvInstantaneous(datafile::String, tS)
#     reads datafile and calculates the instantaneous clv's for each timestep tS,
#     assuming a sample exists after each specified element in tS
# """
# function clvInstantaneous(datafile::String, tS)
#     h5open(datafile, "r") do fid
#         ns = length(tS); # number of samples to store
#         cH = fid["c"]; rH = fid["r"]; nsps = read(fid["nsps"]);
#         ne = size(cH, 1);
#         global clvInst = zeros(ne, ns);
#         C = zeros(ne, ne); R = zeros(ne, ne);
#         for (i, t) in enumerate(tS)
#             C = reshape(cH[:, :, t], (ne, ne));
#             R = reshape(rH[:, :, t+1], (ne, ne));
#             clvInst[:, t] = log.(clvInstantGrowth(C, R))/nsps
#         end
#     end
#     return clvInst
# end
# export clvInstantaneous

function clvGinelliBackwards(QS, RS, Rw)
    # initialize items for backward evolution
    ht = size(QS, 1)
    ne = size(RS, 1)
    ns = size(QS, 3);
    VS = zeros(ht, ne, ns);
    cdelay = size(Rw, 3);
    # warm up C vectors with backward transient Rw matrices
    C = Matrix(1.0I, ne, ne)
    @showprogress 10 "Warmup Completion " for i = cdelay-1:-1:1
        C = backwardsRC(C, Rw[:, :, i+1]);
    end
    C = backwardsRC(C, Rw[:, :, 1]);
    VS[:, :, end] = QS[:, :, end]*C;
    println("set IC for calculating V.")
    @showprogress 10 "Sample Completion " for i = ns-1:-1:1
        C = backwardsRC(C, RS[:, :, i+1]);
        VS[:, :, i] = QS[:, :, i]*C;
    end
    # return results of backward component of Ginelli's method for CLV computation
    return VS
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
    @showprogress 10 "Delay Completed " for i=1:delay
        x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        lambdaInst[:, i] = log.(diag(r))/(nsps)
    end
    # println(delta)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, ns);
    RS = zeros(ne, ne, ns);
    QS = zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress 10 "Sample Calculations Completed " for i=1:ns
        x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        yS[:, i] = x0;
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*nsps)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    # Qw = zeros(ht, ne, cdelay);
    @showprogress 10 "Forward Warmup Completed " for i=1:cdelay
        # advance delta and x0 by δt
        x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        Rw[:, :, i] = r
        # Qw[:, :, i] = delta
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS, lambdaInst
end

# using HDF5
function clvGinelliLongMap(map, jacobian, p, x0, delay, ns, ne, cdelay, nsps, nsim, filename; smallfile=true)
    fid = h5open(filename, "w");
    # initialize local variables
    ht = length(x0)
    delayResets = Int(delay/nsim)
    nsResets = Int(ns/nsim)
    cdelayResets = Int(cdelay/nsim)
    # create data handles for storing the data and allocate the necesary
    vH = create_dataset(fid, "v", datatype(Float64), dataspace(ht, ne, ns))
    yH = create_dataset(fid, "y", datatype(Float64), dataspace(ht, ns))
    # minimizes file size if desired
    if smallfile
        # only saves what's necessary for a minimal file size. Creates a temporary file for
        # necessary data to get CLVs
        wd = dirname(filename);
        tmpfile = joinpath(wd, "tmpdata.h5");
        fidtmp = h5open(tmpfile, "w");
        rH = create_dataset(fidtmp, "r", datatype(Float64), dataspace(ne, ne, ns))
        qH = create_dataset(fidtmp, "q", datatype(Float64), dataspace(ht, ne, ns))
        rwH = create_dataset(fidtmp, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
    else
        rH = create_dataset(fid, "r", datatype(Float64), dataspace(ne, ne, ns))
        qH = create_dataset(fid, "q", datatype(Float64), dataspace(ht, ne, ns))
        cH = create_dataset(fid, "c", datatype(Float64), dataspace(ne, ne, ns))
        rwH = create_dataset(fid, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
        cwH = create_dataset(fid, "cw", datatype(Float64), dataspace(ne, ne, cdelay))
    end
    λH = create_dataset(fid, "lambdaInst", datatype(Float64), dataspace(ne, delay))
    # store parameters of the run
    fid["parameters"] = p;
    fid["delay"] = delay;
    fid["ns"] = ns;
    fid["ne"] = ne;
    fid["cdelay"] = cdelay;
    fid["nsps"] = nsps;
    fid["nsim"] = nsim;
    fid["p"] = p;
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
    @showprogress 10 "Delay Completed " for i=1:delayResets
        for j=1:nsim
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            λi[:, j] = log.(diag(r))/(nsps)
        end
        λH[:, range((i-1)*nsim+1, length=nsim)] = λi
    end
    # println(delta)
    println("lattice warmed up, starting GSV evolution. Now at time: t(a)")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, nsim); # zeros(ht, ns);
    Rtmp = zeros(ne, ne, nsim); # zeros(ne, ne, ns);
    Qtmp = zeros(ht, ne, nsim); # zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress 10 "t(a) -> t(b): Sample Calculations Completed " for i=1:nsResets
        for j=1:nsim
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            yS[:, j] = x0;
            Rtmp[:, :, j] = r
            Qtmp[:, :, j] = delta
            lypspecGS += log.(diag(r))/(ns*nsps)
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        yH[:, trng] = yS;
        # vn[:, trng] = vn;
        rH[:, :, trng] = Rtmp;
        qH[:, :, trng] = Qtmp;
    end
    println("collected QR data.")
    # create R data for warming up C
    @showprogress 10 "t(b) -> t(c): Forward Warmup Completed " for i=1:cdelayResets
        for j=1:nsim
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            Rtmp[:, :, j] = r
            # Qtmp[:, :, j] = delta
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        rwH[:, :, trng] = Rtmp;
        # qH[:, :, trng] = Qw;
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs. Now at time: t(c). Proceeding backwards in time.")
    # save GS method lyapunov spectrum
    fid["lypGS"] = lypspecGS;
    # return results of forward component of Ginelli's method for CLV computation
    # return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
    
    # initialize items for backward evolution
    # warm up C vectors with backward transient Rw matrices
    Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # RS = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    C = Matrix(1.0I, ne, ne);
    # C0 = copy(C1); # zeros(ne, ne);
    # cwH[:,:, end] = C1;
    # CS[:, :, end] = C1;
    # warms up C matrix
    @showprogress 10 "t(c) -> t(b): Warmup Completion " for i = cdelayResets:-1:1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Warmup Completion:\t $(round((cdelayResets-i)/cdelayResets*100))%,\t Range:\t $trng")
        Rtmp = rwH[:, :, trng];
        # sets IC of CS
        if !smallfile; Cw[:, :, end] = C; end;
        for j=nsim-1:-1:1
            C = backwardsRC(C, Rtmp[:, :, j+1]);
            if !smallfile; Cw[:, :, j] = C ; end; # saves everything if not small file
        end
        # assign final value and prepare for next set of data
        # Cw[:, :, 1] = C;
        C = backwardsRC(C, Rtmp[:, :, 1])
        # assign values to dataset
        if !smallfile; cwH[:, :, trng] = Cw; end;
    end

    println("Warmup Completion:\t 100%")
    # assign C from last warm up data to first recorded data
    # C1 = C;
    if !smallfile; cH[:, :, end] = C; end;
    println("set IC for calculating V. Now at time: t(b)")
    # Resize Cw and Rw arrays for datarecording
    # Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # #Rw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # C1 = Matrix(1.0I, ne, ne);
    # C0 = zeros(ne, ne);
    # cwH[:,:, end] = C1;
    @showprogress 10 "t(b) -> t(a): Sample Completion " for i = nsResets:(-1):1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Completion:\t $(round((nsResets-i)/nsResets*100))%,\t Range:\t $trng")
        Rtmp = rH[:, :, trng];
        Qtmp = qH[:, :, trng];
        Vtmp = vH[:, :, trng];
        Vtmp[:, :, end] = Qtmp[:, :, end]*C;
        if !smallfile; Cw[:, :, end] = C; end;
        for j=nsim-1:-1:1
            C = backwardsRC(C, Rtmp[:, :, j+1]);
            Vtmp[:, :, j] = Qtmp[:, :, j]*C;
            if !smallfile; Cw[:, :, j] = C; end;
        end
        # prepare C for next set of data
        C = backwardsRC(C, Rtmp[:, :, 1])
        # assign values to dataset
        if !smallfile; cH[:, :, trng] = Cw; end;
        vH[:, :, trng] = Vtmp;
    end
    # closes temporary file and deletes it if necessary.
    if smallfile
        close(fidtmp);
        rm(tmpfile);
    end
    # close file
    close(fid)
    # return results of backward component of Ginelli's method for CLV computation
    # return CS, Cw
end

function clvGinelliLong(flow, jacobian, p, δt, x0, delay, ns, ne, cdelay, nsps, nsim, filename; smallfile=true)
    fid = h5open(filename, "w");
    # initialize local variables
    ht = length(x0)
    delayResets = Int(delay/nsim)
    nsResets = Int(ns/nsim)
    cdelayResets = Int(cdelay/nsim)
    # create data handles for storing the data and allocate the necesary
    vH = create_dataset(fid, "v", datatype(Float64), dataspace(ne, ne, ns))
    yH = create_dataset(fid, "y", datatype(Float64), dataspace(ht, ns))
    # minimizes file size if desired
    if smallfile
        # only saves what's necessary for a minimal file size. Creates a temporary file for
        # necessary data to get CLVs
        wd = dirname(filename);
        tmpfile = joinpath(wd, "tmpdata.h5");
        fidtmp = h5open(tmpfile, "w");
        rH = create_dataset(fidtmp, "r", datatype(Float64), dataspace(ne, ne, ns))
        qH = create_dataset(fidtmp, "q", datatype(Float64), dataspace(ht, ne, ns))
        rwH = create_dataset(fidtmp, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
    else
        rH = create_dataset(fid, "r", datatype(Float64), dataspace(ne, ne, ns))
        qH = create_dataset(fid, "q", datatype(Float64), dataspace(ht, ne, ns))
        cH = create_dataset(fid, "c", datatype(Float64), dataspace(ne, ne, ns))
        rwH = create_dataset(fid, "rw", datatype(Float64), dataspace(ne, ne, cdelay))
        cwH = create_dataset(fid, "cw", datatype(Float64), dataspace(ne, ne, cdelay))
    end
    λH = create_dataset(fid, "lambdaInst", datatype(Float64), dataspace(ne, delay))
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
    @showprogress 10 "Delay Completed " for i=1:delayResets
        for j=1:nsim
            x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
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
    Rtmp = zeros(ne, ne, nsim); # zeros(ne, ne, ns);
    Qtmp = zeros(ht, ne, nsim); # zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress 10 "t(a) -> t(b): Sample Calculations Completed " for i=1:nsResets
        for j=1:nsim
            x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
            yS[:, j] = x0;
            Rtmp[:, :, j] = r
            Qtmp[:, :, j] = delta
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
    @showprogress 10 "t(b) -> t(c): Forward Warmup Completed " for i=1:cdelayResets
        for j=1:nsim
            x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
            Rtmp[:, :, j] = r
            # Qw[:, :, j] = delta
        end
        # assign values to dataset
        trng = range((i-1)*nsim+1, length=nsim)
        rwH[:, :, trng] = Rtmp;
        # qH[:, :, trng] = Qw;
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs. Now at time: t(c). Proceeding backwards in time.")
    # save GS method lyapunov spectrum
    fid["lypGS"] = lypspecGS;
    # return results of forward component of Ginelli's method for CLV computation
    # return yS, QS, RS, Rw, lypspecGS, Qw, lambdaInst
    


    # initialize items for backward evolution
    # warm up C vectors with backward transient Rw matrices
    Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # RS = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    C = Matrix(1.0I, ne, ne);
    # warms up C matrix
    @showprogress 10 "t(c) -> t(b): Warmup Completion " for i = cdelayResets:-1:1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Warmup Completion:\t $(round((cdelayResets-i)/cdelayResets*100))%,\t Range:\t $trng")
        Rtmp = rwH[:, :, trng];
        # sets IC of CS
        if !smallfile; Cw[:, :, end] = C; end;
        for j=nsim-1:-1:1
            C = backwardsRC(C, Rw[:, :, j+1]);
            if !smallfile; Cw[:, :, j] = C ; end; # saves everything if not small file
        end
        # assign final value and prepare for next set of data
        # Cw[:, :, 1] = C;
        C = backwardsRC(C, Rw[:, :, 1])
        # assign values to dataset
        if !smallfile; cwH[:, :, trng] = Cw; end;
    end

    println("Warmup Completion:\t 100%")
    # assign C from last warm up data to first recorded data
    # C1 = C;
    if !smallfile; cH[:, :, end] = C; end;
    println("set IC for calculating V.")
    # Resize Cw and Rw arrays for datarecording
    # Cw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # #Rw = zeros(ne, ne, nsim); # zeros(ne, ne, cdelay);
    # C1 = Matrix(1.0I, ne, ne);
    # C0 = zeros(ne, ne);
    # cwH[:,:, end] = C1;
    @showprogress 10 "t(b) -> t(a): Sample Completion " for i = nsResets:(-1):1
        # read values from dataset
        trng = range((i-1)*nsim+1, length=nsim)
        # println("Completion:\t $(round((nsResets-i)/nsResets*100))%,\t Range:\t $trng")
        Rtmp = rH[:, :, trng];
        Qtmp = qH[:, :, trng];
        Vtmp = vH[:, :, trng];
        Vtmp[:, :, end] = Qtmp[:, :, end]*C;
        if !smallfile; Cw[:, :, end] = C; end;
        for j=nsim-1:-1:1
            C = backwardsRC(C, RS[:, :, j+1]);
            Vtmp[:, :, j] = Qtmp[:, :, j]*C;
            if !smallfile; Cw[:, :, j] = C; end;
        end
        # assign final value and prepare for next set of data
        # Cw[:, :, 1] = C;
        C = backwardsRC(C, RS[:, :, 1])
        # assign values to dataset
        if !smallfile; cH[:, :, trng] = Cw; end;
        vH[:, :, trng] = Vtmp;
    end
    # closes temporary file and deletes it if necessary.
    if !keepCLVWarmup
        close(fidtmp);
        rm(tmpfile);
    end
    # close file
    close(fid)
    # return results of backward component of Ginelli's method for CLV computation
    # return CS, Cw
end

function covariantLyapunovVectorsMap(map, jacobian, p, x0, delay::Int64,
            ns::Int64, ne::Int64, cdelay::Int64, nsps::Int64, nsim::Int64,
            filename; smallfile=true, saverunavg=false)
    clvGinelliLongMap(map, jacobian, p, x0, delay, ns, ne, cdelay, nsps,
                                nsim, filename, smallfile=smallfile)
    lypCLV = lyapunovSpectrumCLVMap(jacobian, filename, saverunavg=saverunavg) # , nsim)
    h5open(filename, "r+") do fid
        write(fid, "lypCLV", lypCLV)
    end
    println("CLV Lyapunov Spectrum: ")
    println(lypCLV[:, end])
#     return yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw, lambdaInst
end

function covariantLyapunovVectorsMap(map, jacobian, p, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypGS, lambdaInst = clvGinelliMapForward(map,
                                                  jacobian, p, x0,
                                                  delay, ns, ne, cdelay, nsps)
    # CS, Cw = clvGinelliBackwards(QS, RS, Rw)
    VS = clvGinelliBackwards(QS, RS, Rw)
    lypCLV = lyapunovSpectrumCLVMap(jacobian, p, yS, VS, nsps)
    println("CLV Lyapunov Spectrum: ")
    println(lypCLV)

    return yS, VS, lypGS, lypCLV, lambdaInst
end
export covariantLyapunovVectorsMap

function covariantLyapunovVectors(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps)
    yS, QS, RS, Rw, lypGS, lambdaInst = clvGinelliForward(flow, jacobian,
                                                                    p, δt, x0,
                                                  delay, ns, ne, cdelay, nsps)
    # CS, Cw = clvGinelliBackwards(QS, RS, Rw)
    VS = clvGinelliBackwards(QS, RS, Rw)
    lypCLV = lyapunovSpectrumCLV(jacobian, p, yS, VS, nsps, δt)
    println("CLV Lyapunov Spectrum: ")
    println(lypCLV)

    return yS, VS, lypGS, lypCLV, lambdaInst
end


function covariantLyapunovVectors(flow, jacobian, p, δt, x0, delay, ns, ne,
                                  cdelay, nsps, nsim::Int64, filename; smallfile=true, saverunavg=true)
    clvGinelliLong(flow, jacobian, p, δt, x0, delay, ns, ne,
                            cdelay, nsps, nsim, filename, smallfile=smallfile)
    lypCLV = lyapunovSpectrumCLV(jacobian, filename, saverunavg=saverunavg) # , nsim)
    h5open(filename, "r+") do fid
        write(fid, "lypCLV", lypCLV)
    end
    println("CLV Lyapunov Spectrum: ")
    println(lypCLV)
end
export covariantLyapunovVectors
function advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
    ht = length(x0);
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

    return x0, delta, r
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
    @showprogress 10 "Delay Completed " for i=1:delay
        x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        lambdaInst[:, i] = log.(diag(r))/(nsps*δt)
    end
    # println(delta)
    println("lattice warmed up, starting GSV evolution.")
    #= initialize  variables for evolution of delta and lattice, calculating Q
    and lypspec at all times =#
    J = Matrix(1.0I, ht, ht)
    # evolve delta and lattice by numsteps in forward direction
    yS = zeros(ht, ns);
    RS = zeros(ne, ne, ns);
    QS = zeros(ht, ne, ns);
    lypspecGS = zeros(ne)
    @showprogress 10 "Sample Calculations Completed " for i=1:ns
        x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        yS[:, i] = x0;
        RS[:, :, i] = r
        QS[:, :, i] = delta
        lypspecGS += log.(diag(r))/(ns*nsps*δt)
    end
    println("collected QR data.")
    # create R data for warming up C
    Rw = zeros(ne, ne, cdelay);
    Qw = zeros(ht, ne, cdelay);
    @showprogress 10 "Forward Warmup Completed " for i=1:cdelay
        # advance delta and x0 by δt
        x0, delta, r = advanceQR(flow, jacobian, x0, delta, p, δt, nsps)
        Rw[:, :, i] = r
        Qw[:, :, i] = delta
    end
    # finished forward evolution of lattice
    # write lyapunov spectrum as calculated by the R components to file
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS)
    println("finished forward evolution of CLVs")
    # return results of forward component of Ginelli's method for CLV computation
    return yS, QS, RS, Rw, lypspecGS, lambdaInst
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
    @showprogress 10 "Delay Completed " for i in 1:delay
        x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        # timestep(lattice)
    end
    println("lattice warmed up, starting GSV evolution.")
    # calculate Lyapunov Spectrum for the given number samples (ns) and
    # given spacing (nsps)
    if saverunavg
        lypspecGS = zeros(ne, ns)
        lsGSravg = zeros(ne)
        @showprogress 10 "Sample Calculations Completed " for i=1:ns
        # for i=1:ns
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            lsGSravg += log.(diag(r))
            lypspecGS[:, i] = lsGSravg/(i*nsps)
        end
        println("the GSV Lyapunov Spectrum is:")
        println(lypspecGS[:, end])
    else
        @showprogress 10 "Sample Calculations Completed " for i=1:ns
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            lypspecGS += log.(diag(r))/(ns*nsps)
        end
        println("the GSV Lyapunov Spectrum is:")
        println(lypspecGS)
    end
    # finished evolution of lattice
    return lypspecGS
end
export lyapunovSpectrumGSMap

"""
lyapunovSpectrumGSMapDynamics(map, jacobian, p, x0, delay, ns, ne, nsps, delta0)

Returns the Lyapunov Spectrum using the Gram-Schmidt Method for computing the
exponents. Returns the running average spectrum along with the lattice dynamics.

"""
function lyapunovSpectrumGSMapDynamics(map, jacobian, p, x0, delay, ns, ne, nsps, delta0=Matrix(1.0I, length(x0), ne); tout=ns*nsps)
    # initialize local variables
    ht = length(x0)
    lattice = copy(x0)
    lypspecGS = zeros(ne) # will contain lyapunov spectrum to be returned
    delta = copy(delta0)

    # the number of total steps for the system is the number of samples (ns)
    # times the number of steps per sample (nsps)
    # numsteps = ns*nsps
    # warm up the lattice (x0) and perturbations (delta)
    @showprogress 10 "Delay Completed " for i in 1:delay
        x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
        # timestep(lattice)
    end
    println("lattice warmed up, starting GSV evolution.")
    # calculate Lyapunov Spectrum for the given number samples (ns) and
    # given spacing (nsps)

    # determine appropriate times to report out the spectrum
    tsteps = Int.(floor.(tout/nsps));
    tout = tsteps*nsps;
    tcum = cumsum(diff(tsteps));
    # intialize output arrays
    ns = length(tsteps);
    lypspecGS = zeros(ne, ns);
    lsGSravg = zeros(ne);
    x = zeros(ht, ns);
    x[:, 1] = x0;
    @showprogress 10 "Sample Calculations Completed " for (ind, nstep) in enumerate(diff(tsteps))
        for i in 1:nstep
            x0, delta, r = advanceQRMap(map, jacobian, x0, delta, p, nsps)
            lsGSravg += log.(diag(r))
        end
        lypspecGS[:, ind+1] = lsGSravg/(tcum[ind]*nsps);
        x[:, ind+1] = x0;
    end
    println("the GSV Lyapunov Spectrum is:")
    println(lypspecGS[:, end])
    # finished evolution of lattice, return desired quantities
    return x, lypspecGS, delta, tout
end
export lyapunovSpectrumGSMapDynamics
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
    @showprogress 10 "Calculating DOS " for i=1:ns-window
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
  DOS(datafile::String, windows; ts=0)
Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(datafile::String, windows; ts=0)
    fid = h5open(datafile, "r")
    rH = fid["r"];
    cH = fid["c"];
    if ts == 0
        # default is to use all data points from sampling
        ns = size(rH, 3);
        ts = 1:ns-1;
    else
        # set number of samples as the length of specified ts
        # presumes ts is sequential timesteps
        ns = length(ts);
        ts = ts[1]:ts[end-1]
    end
    ne = size(cH, 1);
    nu = zeros(ne, ne);
    nuDict = Dict();
    nutemp = zeros(ne, ne);
    CLV_growth = zeros(ne);
    # @assert(window < ns && window > 0) # checks that window is not too large
    mktemp() do path, io
        fid = h5open(path, "w")
        growth = create_dataset(fid, "growth", datatype(Float64), dataspace(ne, ns))
        @showprogress 10 "Instant Growths " for (i, t) in enumerate(ts)
            C1 = reshape(cH[:, :, t], (ne, ne));
            R2 = reshape(rH[:, :, t+1], (ne, ne));
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
                @showprogress 10 "Window: $window " for i = 2:ns-window
                   clvGrowth -= reshape(fid["growth"][:, i-1], ne)/window;
                   clvGrowth += reshape(fid["growth"][:, i+window-1], ne)/window;
                   nutemp += dosViolations(clvGrowth);
                end
                nu = nutemp/(ns-window);
                nuDict[window] = nu;
            else
                # regular process
                @showprogress 10 "Window: $window " for i=1:ns-window
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
  @showprogress 10 "Calculating DOS " for i=1:ns-window
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
        growth = create_dataset(fid, "growth", datatype(Float64), dataspace(ne, ns))
        @showprogress 10 "Instant Growths " for i = 1:ns-1
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
                @showprogress 10 "Window: $window " for i = 2:ns-window
                   clvGrowth -= reshape(fid["growth"][:, i-1], ne)/window;
                   clvGrowth += reshape(fid["growth"][:, i+window-1], ne)/window;
                   nutemp += dosViolations(clvGrowth);
                end
                nu = nutemp/(ns-window);
                nuDict[window] = nu;
            else
                # regular process
                @showprogress 10 "Window: $window " for i=1:ns-window
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
        @showprogress 10 "Power Spectrum " for t=1:ns # t=1:nsResets
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
    @showprogress 10 "Manifold Angle " for t=1:ns
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
        @showprogress 10 "Calculating CLVs" for i = 1:ns
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
    @showprogress 10 "Calculating Localization " for t1 = 1:nsResets
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

# additional functions, moved to a new file for convience
include("maps_flows_jacobians.jl")
include("maps_flows_jacobians_2d.jl")
include("angle_analysis.jl")
include("domain_analysis.jl")

end # module
