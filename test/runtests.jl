using LADS
using Test
using Statistics, LinearAlgebra, HDF5
import Random.seed!
# set random seed
seed!(1234);

@testset "LADS.jl" begin

#------------------------------------------------------------------------------
# unit test for lorenzFlow and lorenzJacobian functions
# Get the Lorenz System running
sigma = 10; rho = 28; beta = 8/3;
p = [sigma, rho, beta];
x0 = rand(3); # δx = zeros(3); δx[1] = 10^-10;
J = lorenzJacobian(x0, p);
f0 = lorenzFlow(x0, p);
dfdx = zeros(3, 3);
magδx = 10^-8;
for j = 1:3
    δx = zeros(3); δx[j] = magδx;
    dfdx[:, j] = ((lorenzFlow(x0+δx, p) - f0)/magδx)
end

# println(J)
# println(dfdx)
# println(J-dfdx)
# println(norm(J - dfdx))
# println(norm(J - dfdx) < length(J)*magδx*10)
@test norm(J - dfdx) < size(J, 1)*magδx*10


#------------------------------------------------------------------------------
# unit test for modelAMap and modelAJacobian functions
# Set parameters and test point
a = 0.4; b = 1.3; L = 128; eps = 0.02;
x0 = rand(L); # δx = zeros(3); δx[1] = 10^-10;
u0 = sum(x0)/L;
p = [a, b, eps, u0];
J = modelAJacobian(x0, p);
f0 = modelAMap(x0, p);
dfdx = zeros(L, L);
magδx = 10^-8;
for j = 1:L
    δx = zeros(L); δx[j] = magδx;
    dfdx[:, j] = ((modelAMap(x0+δx, p) - f0)/magδx)
end

# println(J)
# println(dfdx)
# println(J-dfdx)
# println(norm(J - dfdx))
# println(norm(J - dfdx) < length(J)*magδx*10)
@test norm(J - dfdx) < size(J, 1)*magδx*10


#------------------------------------------------------------------------------
# unit test for tentMap and tentJacobian functions
# Set parameters and test point
mu = 0.2; K = 1.3; L = 128;
p = [mu, K];
x0 = rand(L); # δx = zeros(3); δx[1] = 10^-10;
J = tentJacobian(x0, p);
f0 = tentMap(x0, p);
dfdx = zeros(L, L);
magδx = 10^-8;
for j = 1:L
    δx = zeros(L); δx[j] = magδx;
    dfdx[:, j] = ((tentMap(x0+δx, p) - f0)/magδx)
end

# println(J)
# println(dfdx)
# println(J-dfdx)
# println(norm(J - dfdx))
# println(norm(J - dfdx) < length(J)*magδx*10)
@test norm(J - dfdx) < size(J, 1)*magδx*10


#------------------------------------------------------------------------------
# unit test for tentCMap and tentCJacobian functions
# Set parameters and test point
mu = 0.2; K = 1.3; C = 1.0; beta = 0.75; L = 128;
p = [mu, K, C, beta];
x0 = rand(L); # δx = zeros(3); δx[1] = 10^-10;
J = tentCJacobian(x0, p);
f0 = tentCMap(x0, p);
dfdx = zeros(L, L);
magδx = 10^-8;
for j = 1:L
    δx = zeros(L); δx[j] = magδx;
    dfdx[:, j] = ((tentCMap(x0+δx, p) - f0)/magδx)
end

# println(J)
# println(dfdx)
# println(J-dfdx)
# println(norm(J - dfdx))
# println(norm(J - dfdx) < length(J)*magδx*10)
@test norm(J - dfdx) < size(J, 1)*magδx*10


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
    (yS, QS, RS, CS, lypspecGS,
    lypspecCLV, Qw, Cw, lambdaInst, Rw) = covariantLyapunovVectorsMap(simpleMap,
                            simpleJacobian, p, x0, delay, ns, ne, cdelay, nsps)

    # sanity check: lyapunov spectrum
    @test isapprox(log.(diag(RS[:, :, 1])), lypspecGS);
    @test isapprox(log.([p1, p2, p3]), lypspecGS);
    @test isapprox(lypspecCLV, lypspecGS);
    # check all Q's are eigenvectors (identity matrix)
    goodQS = true; goodRS = true; goodCS = true; goodlambdaInst = true;
    for t=1:ns
        if !isapprox(QS[:, :, t], Matrix(1.0I, ne, ne))
            goodQS = false
        end
        if !isapprox(CS[:, :, t], Matrix(1.0I, ne, ne))
            goodCS = false
        end
        if !isapprox(diag(RS[:, :, t]), [p1, p2, p3])
            goodRS = false
        end
        if !isapprox(lambdaInst[:, t], lypspecGS)
            goodlambdaInst = false
        end
    end
    @test goodQS; @test goodRS; @test goodCS; @test goodlambdaInst;
end


#------------------------------------------------------------------------------
# Comparison of CLV function in memory and out of memory
# flow = linearFlow; jacobian = linearJacobian;
# sigma = 10; rho = 28; beta = 8/3;
K = 0.65; mu = 1.1; L = 256; seed!(1);
x0 = rand(L)*0.03 - 0.1*ones(L); x1 = copy(x0);
p = [mu K]; # [sigma rho beta];
ne = 256;
nsps = 2;
tConverge = 200; # number of time units to look for convergence of QR values
delay = Int(tConverge/(nsps));
cdelay = Int(tConverge/(nsps));
tSample = 100;
ns = Int(tSample/(nsps));

ht = length(x0);
yS, QS, RS, CS, lypspecGS, lypspecCLV, Qw, Cw, lambdaInst, Rw = covariantLyapunovVectorsMap(tentMap, tentJacobian, p, x0, delay, ns, ne, cdelay, nsps)

nsim = 25 # 50 # 10;
datafile = "testTentMap2.h5"
covariantLyapunovVectorsMap(tentMap, tentJacobian, p, x0, delay, ns, ne, cdelay, nsps, nsim, datafile)
lypFile = zeros(ne); cFile = zeros(ne, ne, ns); rFile = zeros(ne, ne, ns);
cwFile = zeros(ne, ne, cdelay); rwFile = zeros(ht, ne, cdelay);
fid = h5open(datafile, "r")
# global lypFile, cFile, rFile, cwFile, rwFile
lypFile = read(fid["lypspecCLV"])
cFile = read(fid, "c")
rFile = read(fid, "r")
cwFile = read(fid, "cw")
rwFile = read(fid, "rw")
# end
close(fid)
println("Error between in memory and file code versions is: ", norm(lypFile - lypspecCLV))

println("Error in C matchup: \t", norm(CS - cFile))
println("Error in R matchup: \t", norm(RS - rFile))
println("Error in Cw matchup: \t", norm(Cw - cwFile))
println("Error in Rw matchup: \t", norm(Rw - rwFile))

@test norm(lypspecCLV-lypFile) == 0
@test norm(CS-cFile) == 0
@test norm(RS-rFile) == 0
@test norm(Cw-cwFile) == 0
@test norm(Rw-rwFile) == 0

#------------------------------------------------------------------------------
# Unit testing for minimumManifoldAngle function
uInd = 1:5; sInd = 6:10; ne = 10;
c = Matrix(1.0I, ne, ne);
@test minimumManifoldAngle(c, uInd, sInd) == pi/2 # orthogonal set of vectors
c[1, 1] = 1/sqrt(2); c[1, 2] = 1/sqrt(2);
c[2, 1] = 1/sqrt(2); c[2, 2] = 1/sqrt(2);
uInd = 1; sInd = 2:10;
c = Matrix(1.0I, ne, ne);
R(theta) = [cos(theta) -sin(theta); sin(theta) cos(theta)]
c[1:2, 1] = R(pi/4)*c[1:2, 1];
@test isapprox(minimumManifoldAngle(c, uInd, sInd), pi/4) # orthogonal set of vectors

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
