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