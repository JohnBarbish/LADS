@testset "maps_flows_jacobians_2d" begin
#------------------------------------------------------------------------------
# Unit test for 2d wrap and unwrap function
# find coordinates (i, j) with n elements per row (i)
nx = 10;
i = 2; j = 3;
ind1d = 22;
# this should be 22
@test LADS.wrap2d(i, j, nx) == ind1d;
# check this unwraps back to i, J
@test LADS.unwrap2d(ind1d, nx) == (i, j)

#------------------------------------------------------------------------------
end