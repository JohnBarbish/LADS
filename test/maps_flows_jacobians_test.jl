@testset "maps_flows_jacobians" begin
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

@test norm(J - dfdx) < size(J, 1)*magδx*10

#------------------------------------------------------------------------------
# unit test for tentMapRBC and tentJacobianRBC functions
# Set parameters and test point
mu = 0.2; K = 1.3; C = 1.0; L = 128; beta = 1.0;
x0 = rand(L); # δx = zeros(3); δx[1] = 10^-10;
C = sum(x0);
p = [mu, K, C, beta];
J = tentCJacobianRBC(x0, p);
f0 = tentCMapRBC(x0, p);
dfdx = zeros(L, L);
magδx = 10^-8;
for j = 1:L-2
    δx = zeros(L); δx[j] = magδx;
    dfdx[:, j] = ((tentCMapRBC(x0+δx, p) - f0)/magδx)
end

@test norm(J - dfdx) < size(J, 1)*magδx*10

#------------------------------------------------------------------------------
end