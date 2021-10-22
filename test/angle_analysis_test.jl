@testset "angle analysis" begin

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
# Unit testing for angle function
v1 = [1.0, 0.0, 0.0, 0.0]; v2 = [0.0, 2.0, 0.0, 0.0];
@test isapprox(LADS.angle(v1, v2), pi/2)
v1 = [1.0, 0.0, 0.0, 0.0]; v2 = [10.0, 0.0, 0.0, 0.0];
@test isapprox(LADS.angle(v1, v2), 0)

#------------------------------------------------------------------------------
# Unit testing for allAngles function
v1 = [1.0, 0.0, 0.0, 0.0]; v2 = [10.0, 0.0, 0.0, 0.0];
v3 = [0.0, 0.0, 1.0, 0.0]; v4 = [0.0, 0.0, 1.0, 0.0];
c = hcat(v1, v2, v3, v4);
aa = allAngles(c, false)
@test issymmetric(aa) # angle between i and j is the same as j and i
# angle between vector and itself is always zero
@test all(i -> i==0, diag(aa))
# check all angles
aas = zeros(4, 4); aas[1:2, 3:4] .= pi/2; aas[3:4, 1:2] .= pi/2;
@test aas == aa

#------------------------------------------------------------------------------

end