using LADS
using Test
using Statistics, LinearAlgebra, HDF5
import Random.seed!

@testset "clv_test" begin
beta = 0.0;
mu= 1.1;
K=0;
# 
L = 128;
ne = L;
nsps = 1;
xL = -0.01; xH = 0.02;
c = 0.005;
x0 = avginband(L, xL, xH, c);
C = L*c;
p = [mu K C beta];
tConverge = 10; # number of time units to look for convergence of QR values
delay = Int(tConverge/(nsps));
cdelay = Int(tConverge/(nsps));
tSample = 10; # 500000
ns = Int(tSample/(nsps));
ht = length(x0);
##------------------------------------------------------------------------------
##Code for testing long time function
(yS, QS, RS, CS, lypspecGS,
lypspecCLV, Qw, Cw, lambdaInst, Rw) = covariantLyapunovVectorsMap(LADS.tentCMap,
                        LADS.tentCJacobian, p, x0, delay, ns, ne, cdelay, nsps)
# check all the results
# check R matrix
@test all(mapslices(i->isapprox(abs.(i), mu*Matrix(1.0I, L, ne)), RS, dims=(1, 2)))
# check Q matrix
@test all(mapslices(i->isapprox(abs.(i), Matrix(1.0I, L, ne)), QS, dims=(1, 2)))
# check Lyapunov spectra
@test all(i -> isapprox(i, log(mu)), lypspecCLV)
@test all(i -> isapprox(i, log(mu)), lypspecGS)
end

@testset "lyapunov_spectrum" begin
beta = 0.0;
mu= 1.1;
K=0;
# 
L = 128;
ne = L;
nsps = 1;
xL = -0.01; xH = 0.02;
c = 0.005;
x0 = avginband(L, xL, xH, c);
C = L*c;
p = [mu K C beta];
tConverge = 10; # number of time units to look for convergence of QR values
delay = Int(tConverge/(nsps));
cdelay = Int(tConverge/(nsps));
tSample = 10; # 500000
ns = Int(tSample/(nsps));
ht = length(x0);
delta0 = Matrix(1.0I, length(x0), ne); # h5read(icfile, "delta");
tspec =     Int.(0:10:1E2);
##------------------------------------------------------------------------------
##Code for testing long time function
y, lypGS, delta, tout = lyapunovSpectrumGSMapDynamics(tentCMap, tentCJacobian, p, x0, delay, ns, ne, nsps, delta0, tout=tspec)
# check all the results
# check Q matrix
@test isapprox(abs.(delta), Matrix(1.0I, L, ne))
# check Lyapunov spectrum
@test all(i -> isapprox(i, log(mu)), lypGS[:, 2:end])
end