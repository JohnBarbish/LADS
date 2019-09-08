using LADS
using Test
using Statistics
@testset "LADS.jl" begin
# unit test for zero_index function
a = rand(100);
sort(a)
a[10] = 0;
@test isapprox(zero_index(a), 10) # 10.5 "zero_index Error: improper index found"
# println("function:\tzero_index\t-\tpassed")

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


#------------------------------------------------------------------------------
# Unit testing for C_creation function


#------------------------------------------------------------------------------
# Unit testing for forward_evolution function


#------------------------------------------------------------------------------
# Unit testing for sumu_rbc! function


#------------------------------------------------------------------------------
# Unit testing for sumu! function
y = rand(100); u = 1.0;
sumu!(y, u)
@test isapprox(mean(y), u)

#------------------------------------------------------------------------------
# Unit testing for pdf function


#------------------------------------------------------------------------------
# Unit testing for set_ϵ function


#------------------------------------------------------------------------------
# Unit testing for set_u₀ function


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
x = zeros(10); y = zeros(10); x[1] = 1; y[2] = 1;
@test isorthogonal(x, y) == true
y[1] = 1; y[2] = 0;
@test isorthogonal(x, y) == false

#------------------------------------------------------------------------------
# Unit testing for set_μ function


    #------------------------------------------------------------------------------
    # Unit testing for set_K function


    #------------------------------------------------------------------------------
    # Unit testing for set_b function


    #------------------------------------------------------------------------------
    # Unit testing for set_a functionend

end
