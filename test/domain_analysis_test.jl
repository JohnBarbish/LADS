@testset "pspec" begin

#------------------------------------------------------------------------------
# Unit testing for powerSpectrum function

# constant signal
N = 10;
x = ones(N);
krng, psdx = pspec(x, rms=true);
# check that only cos(k=0) is one, everything else is zero.
@test krng == 0:5;
nf = Int(N/2+1);
xhatm = zeros(nf); xhatm[1] = 1.0;
@test isapprox(psdx, xhatm, atol=eps(Float64))

# sine signal
N = 10;
t = 0:N-1;
x = sin.(2*pi*t/N);
krng, psdx = pspec(x, rms=true);
# check that only sin(k=1) is one, everything else is zero.
@test krng == 0:5;
nf = Int(N/2+1);
xhatm = zeros(nf); xhatm[2] = 0.5;
@test isapprox(psdx, xhatm, atol=eps(Float64))

# two sine signals
N = 10;
t = 0:N-1;
x = sin.(2*pi*t/N); # k=1
k2 = 3; m2 = 0.1;
x += m2*sin.(2*pi*k2*t/N); # k=3
krng, psdx = pspec(x, rms=true);
# check that only sin(k=1), sin(k=3) are nonzero, everything else is zero.
@test krng == 0:5;
nf = Int(N/2+1);
xhati = zeros(nf); xhati[2] = 0.5;
xhati[k2+1] = 0.5*m2;
@test isapprox(psdx, xhati, atol=eps(Float64))

# cos signal
N = 10;
t = 0:N-1;
x = cos.(2*pi*t/N);
krng, psdx = pspec(x, rms=true);
# check that only cos(k=1) is one, everything else is zero.
@test krng == 0:5;
nf = Int(N/2+1);
xhatm = zeros(nf); xhatm[2] = 0.5;
@test isapprox(psdx, xhatm, atol=eps(Float64))

# two cos signals
N = 10;
t = 0:N-1;
x = cos.(2*pi*t/N); # k=1
k2 = 3; m2 = 0.1;
x += m2*cos.(2*pi*k2*t/N); # k=3
krng, psdx = pspec(x, rms=true);
# check that only cos(k=1), cos(k=3) are nonzero, everything else is zero.
@test krng == 0:5;
nf = Int(N/2+1);
xhati = zeros(nf); xhati[2] = 0.5;
xhati[k2+1] = 0.5*m2;
@test isapprox(psdx, xhati, atol=eps(Float64))


# two cos signals, odd signal length
N = 11;
t = 0:N-1;
x = cos.(2*pi*t/N); # k=1
k2 = 3; m2 = 0.1;
x += m2*cos.(2*pi*k2*t/N); # k=3
krng, psdx = pspec(x, rms=true);
# check that only cos(k=1), cos(k=3) are nonzero, everything else is zero.
@test krng == 0:5;
nf = Int((N+1)/2);
xhati = zeros(nf); xhati[2] = 0.5;
xhati[k2+1] = 0.5*m2;
@test isapprox(psdx, xhati, atol=eps(Float64))


# two cos signals, odd signal length, longer sample
N = 101;
t = 0:N-1;
x = cos.(2*pi*t/N); # k=1
k2 = 3; m2 = 0.1;
x += m2*cos.(2*pi*k2*t/N); # k=3
krng, psdx = pspec(x, rms=true);
# check that only cos(k=1), cos(k=3) are nonzero, everything else is zero.
nf = Int((N+1)/2);
@test krng == 0:nf-1;
xhati = zeros(nf); xhati[2] = 0.5;
xhati[k2+1] = 0.5*m2;
@test isapprox(psdx, xhati, atol=eps(Float64))

end