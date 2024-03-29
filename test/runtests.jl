using LADS
using Test
using Statistics, LinearAlgebra, HDF5
import Random.seed!
# set random seed
seed!(1234);

@testset "LADS.jl" begin

include("maps_flows_jacobians_test.jl")
include("maps_flows_jacobians_2d_test.jl")
include("angle_analysis_test.jl")
include("domain_analysis_test.jl")
include("clv_test.jl")
#------------------------------------------------------------------------------
# unit test for zeroIndex function
a = rand(100);
sort(a)
a[10] = 0;
@test isapprox(zeroIndex(a), 10) # 10.5 "zeroIndex Error: improper index found"
# println("function:\tzeroIndex\t-\tpassed")

#------------------------------------------------------------------------------
# Unit testing for reomve_zero_datapoints function
x = rand(100); y = rand(100);
y[1] = 0; y[54] = 0; y[32] = 0; y[85] = 0.0000001; y[48] = 0.00000000000001;

xc, yc = remove_zero_datapoints(x, y)

@test all(yc != 0) # "There is a zero datapoint in y dataset."
# println("function:\tremove_zero_datapoints\t-\tpassed")

#------------------------------------------------------------------------------
# Unit testing for principal_angle_distribution function


#------------------------------------------------------------------------------
# Unit testing for angle_distribution function


#------------------------------------------------------------------------------
# Unit testing for principal_angle function


#------------------------------------------------------------------------------
# Unit testing for clean_log function


#------------------------------------------------------------------------------
# Unit testing for Kaplan_York_Dimension function


#------------------------------------------------------------------------------
# Unit testing for CLV_instant_growth function


#------------------------------------------------------------------------------
# Unit testing for DOS_violations function


#------------------------------------------------------------------------------
# Unit testing for DOS function


#------------------------------------------------------------------------------
# Unit testing for CLV function
@testset "CLV function" begin
# test for CLV function in the simplest case where jacobian is a constant and
# the eigenvectors form identity matrix and the lyapunov exponents are just the
# logarithm of the eigenvalues.
    function simpleMap(x, p)
        (p1, c1, p2, c2, p3, c3) = p;
        return [p1*x[1] + c1,
                p2*x[2] + c2,
                p3*x[3] + c3];

    end

    function simpleJacobian(xvec, p)
        (p1, c1, p2, c2, p3, c3) = p;
        J = [[p1    0   0];
             [0     p2  0];
             [0     0   p3]]
        return J
    end

    # setting initial conditions to one of the four chaotic bands
    p1, p2, p3 = 3, 2, 0.5;
    p = [p1 0 p2 0 p3 0]; # (p1, c1, p2, c2, p3, c3)
    ne = 3; x0 = ones(3);
    nsps = 1;
    tConverge = 10; # number of time units to look for convergence of QR values
    delay = Int(tConverge/(nsps));
    cdelay = Int(tConverge/(nsps));
    tSample = 10; # 500000
    ns = Int(tSample/(nsps));
    ht = length(x0);
    ##------------------------------------------------------------------------------
    ##Code for testing long time function
    (yS, VS, lypspecGS, lypspecCLV, lambdaInst) = covariantLyapunovVectorsMap(simpleMap,
                            simpleJacobian, p, x0, delay, ns, ne, cdelay, nsps)

    # sanity check: lyapunov spectrum
    # @test isapprox(log.(diag(RS[:, :, 1])), lypspecGS);
    @test isapprox(log.([p1, p2, p3]), lypspecGS);
    @test isapprox(lypspecCLV, lypspecGS);
    # check all Q's are eigenvectors (identity matrix)
    # goodQS = true; goodRS = true; goodCS = true; goodlambdaInst = true;
    goodVS = true; goodlambdaInst = true;
    for t=1:ns
        if !isapprox(VS[:, :, t], Matrix(1.0I, ne, ne))
            goodVS = false
        end
        if !isapprox(lambdaInst[:, t], lypspecGS)
            goodlambdaInst = false
        end
    end
    # @test goodQS; @test goodRS; @test goodCS; @test goodlambdaInst;
    @test goodVS; @test goodlambdaInst;
end

@testset "CLV function in/out memory" begin
#------------------------------------------------------------------------------
# Comparison of CLV function in memory and out of memory
# flow = linearFlow; jacobian = linearJacobian;
# sigma = 10; rho = 28; beta = 8/3;
K = 0.65; mu = 1.1; L = 20; seed!(1);
x0 = rand(L)*0.03 - 0.1*ones(L); x1 = copy(x0);
p = [mu K]; # [sigma rho beta];
ne = 19;
nsps = 2;
tConverge = 200; # number of time units to look for convergence of QR values
delay = Int(tConverge/(nsps));
cdelay = Int(tConverge/(nsps));
tSample = 100;
ns = Int(tSample/(nsps));

ht = length(x0);
yS, VS, lypGS, lypCLV, lambdaInst = covariantLyapunovVectorsMap(tentMap, tentJacobian, p, x0, delay, ns, ne, cdelay, nsps)

nsim = 25 # 50 # 10;
datafile = "testTentMap2.h5"
keepCLVWarmup = true;
covariantLyapunovVectorsMap(tentMap, tentJacobian, p, x0, delay, ns, ne, cdelay, nsps, nsim, datafile)
lypclvFile = zeros(ne); 
# cFile = zeros(ne, ne, ns); rFile = zeros(ne, ne, ns);
# cwFile = zeros(ne, ne, cdelay); rwFile = zeros(ht, ne, cdelay);
fid = h5open(datafile, "r")
# global lypFile, cFile, rFile, cwFile, rwFile
lypclvFile = read(fid["lypCLV"])
lypgsFile = read(fid["lypGS"])
vFile = read(fid, "v")
# cFile = read(fid, "c")
# rFile = read(fid, "r")
# cwFile = read(fid, "cw")
# rwFile = read(fid, "rw")
# end
close(fid)
println("Error between in memory and file code versions is: ", norm(lypgsFile - lypGS))

println("Error in V matchup: \t", norm(VS - vFile))
# println("Error in C matchup: \t", norm(CS - cFile))
# println("Error in R matchup: \t", norm(RS - rFile))
# println("Error in Cw matchup: \t", norm(Cw - cwFile))
# println("Error in Rw matchup: \t", norm(Rw - rwFile))

@test norm(lypCLV-lypclvFile) == 0
@test norm(lypGS-lypgsFile) == 0
@test norm(VS-vFile) == 0
# @test norm(CS-cFile) == 0
# @test norm(RS-rFile) == 0
# @test norm(Cw-cwFile) == 0
# @test norm(Rw-rwFile) == 0
# remove data file
rm(datafile)
end

#------------------------------------------------------------------------------
# Unit testing for sumu_rbc! function


#------------------------------------------------------------------------------
# Unit testing for sumu! function
y = rand(100); u = 1.0;
sumu!(y, u)
@test isapprox(mean(y), u)

#------------------------------------------------------------------------------
# Unit testing for powerSpectrum function
# note that pi/h = N/2 where N is the number of points and h is the distance
# between the points.
# k = 10; L = 25; N = 100; # h = 2*pi/N;
# mag = 2;
# # lambda = 12; k = 2*pi/lambda;
# dx = L/N;
# xrng = 0:dx:L-dx; # 0:2pi/N:2pi*(1-1/N);
# s = [mag*sin(k*x) for x in xrng]
# krng, psdx = newPowerSpectrum(s, dx, 1) # powerSpectrum
# tpsdx = zeros(Int(floor(N/2))+1); tpsdx[k+1] = mag;
# plot(xrng, s)
# plot(krng, psdx, reuse=false, label="calculated")
# plot!(krng, tpsdx, markershape=:auto, label="expected")

# Fs = 1000; # 1 kHz
# fsample = 100; # 100 Hz signal
# T = 100;
# t = 0:1/Fs:T-1/Fs; # time sampling
# N = length(t);
# # tpsdx = zeros(Int(floor(N/2))+1); tpsdx[Int(fsample*N/Fs+1)] = 0.5*N/Fs;
# s = sin.(2*pi*fsample*t); # 100 Hz signal
# pxx = periodogram(s, fs=Fs)
# plot(pxx.freq, pxx.power, label="DSP")
# freq, psdx = powerSpectralDensity(s, Fs);
# # plot!(freq, psdx, label="mine")
# findmax(psdx)
# isapprox(psdx, tpsdx)
# # plot(t, s)
# # plot(freq, psdx, reuse=false, label="calculated")
# # plot!(freq, tpsdx, markershape=:auto, label="expected")
# @test isapprox(psdx, tpsdx)
# @test length(psdx)==length(krng)
# potential for developing higher dimension versions of code and testing
# s2 = cat(s, s, dims=2);
# krng, psdx = powerSpectralDensity(s2, Fs, 1);
# for i in size(s2, 2)
#     @test isapprox(psdx[:, i], tpsdx)
#     @test length(psdx[:, i])==length(krng)
# end
# s3 = cat(s2, s2, s2, dims=3);
# dims = (2, 3);
# tdims = setdiff(1:ndims(s3), dims)
#
# krng, psdx = powerSpectralDensity(s3, dx, 1);
# for i in size(s3, 2), j in size(s3, 3)
#     @test isapprox(psdx[:, i, j], tpsdx)
#     @test length(psdx[:, i, j])==length(krng)
# end

#------------------------------------------------------------------------------
# Unit testing for pdf function
xlims = (0, 2);
data = rand(10000); nbins = 250; data = data.*(xlims[2]-xlims[1]) .+ xlims[1];
boxes, dist = pdf(data, nbins, xlims);
@test isapprox(sum(dist), 1.0)

#------------------------------------------------------------------------------
# Unit testing for isturbulent function


#------------------------------------------------------------------------------
# Unit testing for turbulentarray! function


#------------------------------------------------------------------------------
# Unit testing for autocorr function


#------------------------------------------------------------------------------
# Unit testing for coarsegrain function


#------------------------------------------------------------------------------
# Unit testing for calcanglebetween function


#------------------------------------------------------------------------------
# Unit testing for isorthogonal function
# x = zeros(10); y = zeros(10); x[1] = 1; y[2] = 1;
# @test isorthogonal(x, y) == true
# y[1] = 1; y[2] = 0;
# @test isorthogonal(x, y) == false


end
