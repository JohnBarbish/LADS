module LADS

using Statistics, Dates
using Random, LinearAlgebra
import JLD: jldopen, read, close
using HDF5
global a=0; global b=0; global K=0; global μ=0; global u₀=0; global ϵ=0;
global u=0;
export lyapunov_spectrum, printlog
export isturbulent, turbulentarray! # , autocorr, manualautocorr, coarsegrain!
export isorthogonal, calcanglebetween, principal_angle
export angle_between, angle_distribution, principal_angle_distribution
export forward_evolution, C_creation, CLV, DOS, DOS_violations, lyapunovSpectrumGSM

# set functions for model a parameters (a, b)
function set_a(newa)
  global a = newa
  println("New a value set to $a")
end

function set_b(newb)
  global b = newb
  println("New b value set to $b")
end

# set functions for tent map parameters (K, μ)
function set_K(newK)
  global K = newK
  println("New K value set to $K")
end

function set_μ(newμ)
  global μ = newμ
  println("New μ value set to $μ")
end

function set_u(newu)
  global u = newu
  println("New u value set to $u")
end

function set_u(newu, batchlog)
  global u = newu
  printlog(batchlog, "New u value set to $u")
end

function temppath(foldername="adflkafafda234")
  if foldername == "adflkafafda234"
    if Sys.iswindows()
      filepath = string("model a 1D results\\tmp\\batch-", Dates.format(now(),
                  "mm-dd-yyyy_T-HH-MM\\"))
    elseif Sys.islinux()
      filepath = string("model a 1D results/tmp/batch-", Dates.format(now(),
                        "mm-dd-yyyy_T-HH-MM/"))
    else
      println("unsupported operating system")
      @assert false
    end
  else
    if is_windows()
      filepath = string("model a 1D results\\tmp\\", foldername)
    elseif is_linux()
      filepath = string("model a 1D results/tmp/", foldername)
    else
      println("unsupported operating system")
      @assert false
    end
  end

  mkpath(filepath)
  return filepath
end

function isorthogonal(x::Array{Float64, 1}, y::Array{Float64, 1})
  @assert length(x)==length(y)
  sum = 0
  for i=1:length(x)
    sum += x[i]*y[i]
  end
  if isapprox(sum, 0)
    return true
  else
    return false
  end
end

function calcanglebetween(x::Array{Float64, 1}, y::Array{Float64, 1})
  @assert length(x)==length(y)
  sum = 0
  for i=1:length(x)
    sum += x[i]*y[i]
  end
  return acos(sum)
end

function coarsegrain!(coarsey::Array{Float64, 1}, y::Array{Float64, 1},
                      grainsize::Int)
  @assert size(y, 1) == grainsize*size(coarsey, 1)
  for i=1:size(coarsey, 1)
    coarsey[i] = mean(y[(i-1)*grainsize+1:i*grainsize])
  end
  return
end

function autocorr(g)
  g -= mean(g)
  return irfft(abs2(rfft(g)), size(g, 1))/sumabs2(g)
end
function manualautocorr(g)
  gcorr2 = zeros(length(g))
  g -= mean(g)
  for i=1:length(g)
    for j=1:length(g)-i
      gcorr2[i] += g[j]*g[j+i]
    end
    for j=(length(g)-i+1):length(g)
      gcorr2[i] += g[j]*g[j+i-length(g)]
    end
  end
  gcorr2 = prepend!(gcorr2/sumabs2(g), [1])
  return deleteat!(gcorr2, length(gcorr2))
end

function turbulentarray!(bita, y)
  h = size(y, 1)
  @assert size(bita, 1) == h
  for i = 2:h-1
    bita[i] = isturbulent(y[i-1], y[i], y[i+1])
  end
  bita[1] = isturbulent(y[h], y[1], y[2])
  bita[h] = isturbulent(y[h-1], y[h], y[1])
end

function isturbulent(uleft, umid, uright)
  Amid = abs(uleft - umid)/2
  Aright = abs(uright - umid)/2
  x = abs(Amid - Aright)/(abs(Amid + Aright))
  ϵ = 0.01
  if (2*Amid < ϵ && 2*Aright < ϵ)
    return false
  elseif (x < ϵ || isnan(x))
    return false
  else
    return true
  end
end
#-----------------------------------------------------------------------------#
# functions for 1d model a with broken conservation
function set_u₀(newu₀)
  global u₀ = newu₀
  println("New u₀ is set to $u₀")
end
function set_ϵ(newϵ)
  global ϵ = newϵ
  println("New ϵ is set to $ϵ")
end
# copy of f function for regular 1d model a
# function af(x)::Float64
#   z = mod(x, 1)
#   return a*x + b*z*(1-z)
# end
# copy of fprime function for regular 1d model a
# function afprime(x)::Float64
#   return a + b*(1 - 2*mod(x, 1))
# end

function ag(x::Float64)::Float64
  return (u₀ - x)^3
end

function agprime(x::Float64)::Float64
  return -3*(u₀ - x)^2
end

function model_a_1d_bc(x::Float64, xright::Float64, xleft::Float64)::Float64
  return x + ϵ*ag(x) + af(xleft) + af(xright) - 2*af(x)
end

function mod_a_timestep_bc!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = model_a_1d_bc(x[i], x[i+1], x[i-1])
  end
  # timestep first site
  y[1] = model_a_1d_bc(x[1], x[2], x[h])
  # timestep last site
  y[h] = model_a_1d_bc(x[h], x[1], x[h-1])
  # @assert all(i -> -1 <= i <= 1, y)
end

function mod_a_lin_timestep_bc!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = afprime(lattice[i-1])
    LT[i, i] = 1 + ϵ*agprime(lattice[i]) - 2*afprime(lattice[i])
    LT[i, i+1] = afprime(lattice[i+1])
  end
  LT[1, 1] = 1 + ϵ*agprime(lattice[1]) - 2*afprime(lattice[1])
  LT[1, h] = afprime(lattice[h])
  LT[1, 2] = afprime(lattice[2])
  LT[h, h] = 1 + ϵ*agprime(lattice[h]) - 2*afprime(lattice[h])
  LT[h, 1] = afprime(lattice[1])
  LT[h, h-1] = afprime(lattice[h-1])
end




#-----------------------------------------------------------------------------#
# functions for 1d tent map
function tentf(x)::Float64
  return 1- μ*abs(x)
end

function tent_map_1d(x::Float64, xright::Float64, xleft::Float64)::Float64
  return tentf(x) + K/2*(tentf(xright) - 2*tentf(x) + tentf(xleft))
end

function tent_timestep!(y::Array{Float64, 1})
  # timestep function for 1D lattice with periodic B.C.
  # tent map
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = tent_map_1d(x[i], x[i+1], x[i-1])
  end
  # timestep first site
  y[1] = tent_map_1d(x[1], x[2], x[h])
  # timestep last site
  y[h] = tent_map_1d(x[h], x[1], x[h-1])
end

function tentfprime(x)::Float64
  if x < 0
    return μ
  elseif x > 0
    return -μ
  else
    println("Wubba lubba dub dub!!")
    return 0
  end
end

function tent_lin_timestep!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = K/2*tentfprime(lattice[i-1])
    LT[i, i] = (1 - K)*tentfprime(lattice[i])
    LT[i, i+1] = K/2*tentfprime(lattice[i+1])
  end
  LT[1, 1] = (1 - K)*tentfprime(lattice[1])
  LT[1, h] = K/2*tentfprime(lattice[h])
  LT[1, 2] = K/2*tentfprime(lattice[2])
  LT[h, h] = (1 - K)*tentfprime(lattice[h])
  LT[h, 1] = K/2*tentfprime(lattice[1])
  LT[h, h-1] = K/2*tentfprime(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# functions for 1d model a map
function af(x)::Float64
  z = mod(x, 1)
  return a*x + b*z*(1-z)
end

function model_a_1d(x::Float64, xright::Float64, xleft::Float64)::Float64
  return x + af(xleft) + af(xright) - 2*af(x)
end

function mod_a_timestep!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = model_a_1d(x[i], x[i+1], x[i-1])
  end
  # timestep first site
  y[1] = model_a_1d(x[1], x[2], x[h])
  # timestep last site
  y[h] = model_a_1d(x[h], x[1], x[h-1])
  # @assert all(i -> -1 <= i <= 1, y)
end

"""
  afprime(x)

v07
  function description
"""
function afprime(x)::Float64
  return a + b*(1 - 2*mod(x,1))
end

"""
  mod_a_lin_timestep!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})

  function description.
"""
function mod_a_lin_timestep!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  # working towards evolving this into a generalized linear timestep function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = afprime(lattice[i-1])
    LT[i, i] = 1 - 2*afprime(lattice[i])
    LT[i, i+1] = afprime(lattice[i+1])
  end
  LT[1, 1] = 1 - 2*afprime(lattice[1])
  LT[1, h] = afprime(lattice[h])
  LT[1, 2] = afprime(lattice[2])
  LT[h, h] = 1 - 2*afprime(lattice[h])
  LT[h, 1] = afprime(lattice[1])
  LT[h, h-1] = afprime(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# functions for 1d model a map with rigid boundary conditions and
# broken conservation law if ϵ is nonzero

"""
  mod_a_timestep_rbc!(y::Array{Float64, 1})

  revised model a timestep function where the boundaries are set to zero
    and can have broken boundary conditions.
"""
function mod_a_timestep_rbc!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = model_a_1d_bc(x[i], x[i+1], x[i-1])
  end
  # timestep first site
  y[1] = u # sum(x)/h # model_a_1d_bc(x[1], x[2], x[h])
  # timestep last site
  y[h] = u # sum(x)/h # model_a_1d_bc(x[h], x[1], x[h-1])
  # @assert all(i -> -1 <= i <= 1, y)
end

"""
  mod_a_lin_timestep_rbc!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})

  Creates Jacobian to evolve the perturbations for model a with rigid boundary conditions.
"""
function mod_a_lin_timestep_rbc!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = afprime(lattice[i-1])
    LT[i, i] = 1 + ϵ*agprime(lattice[i]) - 2*afprime(lattice[i])
    LT[i, i+1] = afprime(lattice[i+1])
  end
  LT[1, 1] = 0 # 1 + ϵ*agprime(lattice[1]) - 2*afprime(lattice[1])
  LT[1, h] = 0 # afprime(lattice[h])
  LT[1, 2] = 0 # afprime(lattice[2])
  LT[h, h] = 0 # 1 + ϵ*agprime(lattice[h]) - 2*afprime(lattice[h])
  LT[h, 1] = 0 # afprime(lattice[1])
  LT[h, h-1] = 0 # afprime(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# functions for 1d model a map with parabolic coupling
# NOTE DO NOT USE THIS COUPLING FUNCTION AS IT IS HIGHLY UNSTABLE

"""
  f_par(x::Float64)::Float64

  parabolic function for nearest neighbor coupling. However, this is best used
    when around small x values.
"""
function f_par(x::Float64)::Float64
  # Best when used with small x.
  return 15*x^2
end

"""
  df_par(x::Float64)::Float64

  derivative of the parabolic function for nearest neighbor coupling. However,
    this is best used when around small x values.
"""
function df_par(x::Float64)::Float64
  # Best when used with small x.
  return 30*x
end

"""
  timestep_parabolic_coupling!(y::Array{Float64, 1})

  Timestep function with nearest neighbor coupling and uses a parabolic function
    for the coupling function f(x).
"""
function timestep_parabolic_coupling!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = x[i] + f_par(x[i-1]) - 2*f_par(x[i]) + f_par(x[i+1])
  end
  # timestep first site
  y[1] =  x[1] + f_par(x[h]) - 2*f_par(x[1]) + f_par(x[2])
  # timestep last site
  y[h] = x[h] + f_par(x[h-1]) - 2*f_par(x[h]) + f_par(x[1])
  # @assert all(i -> -1 <= i <= 1, y)
end

"""
  linearized_timestep_parabolic_coupling!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})

  Creates Jacobian to evolve the perturbations for model a with rigid boundary conditions.
"""
function linearized_timestep_parabolic_coupling!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = df_par(lattice[i-1])
    LT[i, i] = 1 - 2*df_par(lattice[i])
    LT[i, i+1] = df_par(lattice[i+1])
  end
  LT[1, 1] = 1 - 2*df_par(lattice[1])
  LT[1, h] = df_par(lattice[h])
  LT[1, 2] = df_par(lattice[2])
  LT[h, h] = 1 - 2*df_par(lattice[h])
  LT[h, 1] = df_par(lattice[1])
  LT[h, h-1] = df_par(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# functions for modified 1d model a with broken conservation and removed Modulo
function af_no_mod(x::Float64)::Float64
  return a*x+b*x*(1-x)
end
function afprime_no_mod(x::Float64)::Float64
  return a*x+b*(1-2*x)
end
function model_a_1d_bc_no_mod(x::Float64, xright::Float64, xleft::Float64)::Float64
  return x + ϵ*ag(x) + af_no_mod(xleft) + af_no_mod(xright) - 2*af_no_mod(x)
end

function mod_a_timestep_bc_no_mod!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = model_a_1d_bc_no_mod(x[i], x[i+1], x[i-1])
  end
  # timestep first site
  y[1] = model_a_1d_bc_no_mod(x[1], x[2], x[h])
  # timestep last site
  y[h] = model_a_1d_bc_no_mod(x[h], x[1], x[h-1])
  # @assert all(i -> -1 <= i <= 1, y)
end

function mod_a_lin_timestep_bc_no_mod!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = afprime_no_mod(lattice[i-1])
    LT[i, i] = 1 + ϵ*agprime(lattice[i]) - 2*afprime_no_mod(lattice[i])
    LT[i, i+1] = afprime_no_mod(lattice[i+1])
  end
  LT[1, 1] = 1 + ϵ*agprime(lattice[1]) - 2*afprime_no_mod(lattice[1])
  LT[1, h] = afprime_no_mod(lattice[h])
  LT[1, 2] = afprime_no_mod(lattice[2])
  LT[h, h] = 1 + ϵ*agprime(lattice[h]) - 2*afprime_no_mod(lattice[h])
  LT[h, 1] = afprime_no_mod(lattice[1])
  LT[h, h-1] = afprime_no_mod(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# functions for 1d model VT simple map with a linear and sin term

"""
  fls(x::Float64)::Float64

  parabolic function for nearest neighbor coupling. However, this is best used
    when around small x values.
"""
function fls(x::Float64)::Float64
  # the constant c1 was found by curve fitting this function with f(x) with mod
  return a*x + b*0.28*sin(pi*x)^2;
end

"""
  dfls(x::Float64)::Float64

  derivative of the parabolic function for nearest neighbor coupling. However,
    this is best used when around small x values.
"""
function dfls(x::Float64)::Float64
  # the constant c1 was found by curve fitting this function with f(x) with mod
  return a + b*2*0.28*sin(pi*x)*pi*cos(pi*x)
end

"""
  timestep_lin_sin_coupling!(y::Array{Float64, 1})

  Timestep function with nearest neighbor coupling and uses a parabolic function
    for the coupling function f(x).
"""
function timestep_lin_sin_coupling!(y::Array{Float64, 1})
  # general timestep function for 1D lattice with periodic B.C.
  # @assert all(i -> -1 <= i <= 1, y)
  h = size(y, 1) # number of rows
  x = copy(y)
  # timestep middle of lattice
  for i = 2:h-1
    y[i] = x[i] + fls(x[i-1]) - 2*fls(x[i]) + fls(x[i+1])
  end
  # timestep first site
  y[1] =  x[1] + fls(x[h]) - 2*fls(x[1]) + fls(x[2])
  # timestep last site
  y[h] = x[h] + fls(x[h-1]) - 2*fls(x[h]) + fls(x[1])
  # @assert all(i -> -1 <= i <= 1, y)
end

"""
  linearized_timestep_lin_sin_coupling!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})

  Creates Jacobian to evolve the perturbations for model a with rigid boundary conditions.
"""
function linearized_timestep_lin_sin_coupling!(LT::Array{Float64, 2}, lattice::Array{Float64, 1})
  # creates appropriate timestep matrix for delta's based on periodic B.C.
  # along with model_a_1d linearized function
  h = size(lattice, 1) # number of rows
  # construct evolution matrix
  for i in 2:h-1
    LT[i, i-1] = dfls(lattice[i-1])
    LT[i, i] = 1 - 2*dfls(lattice[i])
    LT[i, i+1] = dfls(lattice[i+1])
  end
  LT[1, 1] = 1 - 2*dfls(lattice[1])
  LT[1, h] = dfls(lattice[h])
  LT[1, 2] = dfls(lattice[2])
  LT[h, h] = 1 - 2*dfls(lattice[h])
  LT[h, 1] = dfls(lattice[1])
  LT[h, h-1] = dfls(lattice[h-1])
end

#-----------------------------------------------------------------------------#
# general purpose functions
"""
  printlog(stream1::IO, xs...)

  simple function to print statements to a stream along with typical outfile.
"""
function printlog(stream1::IO, xs...)
  # prints to stream1 and also to STDOUT
  println(stream1, xs...)
  println(xs...)
end

"""
  lyapunov_spectrum(lattice, timestep, lin_timestep!, mymap, linmap,
                    numsteps, delay, make_plots=false)

  Calculates lyapunov spectrum for the inputs.
"""
function lyapunov_spectrum(lattice, timestep, lin_timestep!, mymap,
                           linmap, numsteps, delay, make_plots=false)
  size = length(lattice)
  delta = Matrix(1.0I, size)
  # warms up the lattice
  for i in 1:delay
    timestep(lattice, mymap)
  end
  count = 0; count2 = 0
  lypspec = zeros(size)
  latp = Array([])
  LT = zeros(size, size)
  # evolve delta and lattice by numsteps
  while count < numsteps
    while count2 < 10
      count += 1; count2 += 1
      timestep(lattice, mymap)
      lin_timestep!(LT, lattice, linmap)
      delta = LT*delta
      append!(latp, lattice[div(size, 6)])
    end
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign(diag(r))))
    r = sgn*r
    delta = q*sgn
    # println(delta)
    lypspec = (log(diag(r)) + lypspec*(count-count2))/count
    count2 = 0
  end
  if make_plots
    # put code to make lyapunov spectrum plots here
  end
  return lypspec, latp
end

# take array of numbers in and return 'histogram' of array
"""
  pdf(data::Array{Float64, 1}, nbins::Int)

  returns a pdf version of the input data.  a primitive version of a histogram.
"""
function pdf(data::Array{Float64, 1}, nbins::Int)
  mx = maximum(data)
  mx /= nbins
  dist = zeros(nbins)
  for datum in data
    if !isapprox(datum, mx*nbins)
      dist[floor(Int, div(datum, mx)) + 1] += 1
    else
      dist[nbins] += 1
    end
  end
  dist /= length(data)
  return dist
end

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

"""
  sumu_rbc!(y, u)

  modifies y array so that the average lattice point value is u.
  Additionally, sets first and last lattice points to u as well.
"""
function sumu_rbc!(y, u)
  srand(10)
  h = size(y, 1)
  # Resets y to a random array
  for i=1:h
    y[i] = rand(Float64)
  end
  # Setting Boundary conditions for rigid boundary case
  y[1] = u; y[h] = u;
  s = sum(y);
  s = s - u*h;
  s = s/(h-2);
  for i=2:h-1
    y[i] -= s;
  end
  println("New u of y: ", mean(y))
  # y[h-1] = u*h - s
  # y[h] -= sum(y)
  assert(abs(sum(y)/h - u) < 0.1^10)
end

#-----------------------------------------------------------------------------#
# functions useful for calculating angles between vectors and subspaces

# using JLD: jldopen, read, close
# function for determining the angle between two vectors throughout time
#=
"""
  angle_between(C::Array{Float64, 3}, v1::Int, v2::Int)

  function description.
"""
function angle_between(C::Array{Float64, 3}, v1::Int, v2::Int)
  θ = zeros(size(C, 3))
  for t=1:size(C, 3)
    vec1 = C[:, v1, t]; vec2 = C[:, v2, t]
    # positive CLV's go into u1, all negative CLV's go into u2
    θ[t] = acos(round(dot(vec1, vec2)/(norm(vec1)*norm(vec2)), 14))
  end
  # return theta array containing minimum principal angle at each timestep
  return θ
end

# specialized function for determining pdf of angle between two vectors through
# time spread across many files in a specific way
"""
  angle_distribution(path::String, v1::Int, v2::Int, nbins=100)

  function description.
"""function angle_distribution(path::String,
                            v1::Int, v2::Int, nbins=100)
  # using JLD: jldopen, read, close
  mx = π
  mx /= nbins
  dist = zeros(nbins)
  if isfile(path)
    if ismatch(r"^chunk directory.csv", path)
      # do stuff with csv file
      for file in readcsv(path)
        file = String(file)
        println(file)
        f = jldopen(file, "r")
        try
	         C = read(f, "C")
	         θ = angle_between(C, v1, v2)
	          for datum in θ
	             if !isapprox(datum, mx*nbins)
	                dist[floor(Int, div(datum, mx)) + 1] += 1
	               else
	                  dist[nbins] += 1
	             end
	          end
	       finally
	        close(f)
	      end
	    end
      dist /= sum(dist)
      return linspace(0, π, nbins), dist

      # end
      else
        println("Error invalid name. must use chunk directory.csv")
        return zeros(nbins), zeros(nbins)
    end

    else
      for file in readdir(path)
	if ismatch(r"^chunk-\d*x\d*.jld", file)
	  # do stuff with data containing files
	  println(file)
	  f = jldopen(joinpath(path, file), "r")
	  try
	    C = read(f, "C")
	    θ = angle_between(C, v1, v2)
	    for datum in θ
	      if !isapprox(datum, mx*nbins)
		dist[floor(Int, div(datum, mx)) + 1] += 1
	      else
		dist[nbins] += 1
	      end
	    end
	  finally
	    close(f)
	  end

	end
      end
      dist /= sum(dist)
      return linspace(0, π, nbins), dist
    end
end

# function for determining the minimum angle between two subspaces
"""
  principal_angle(C::Array{Float64, 3}, lypspec::Array{Float64, 1}, cutoff=-1)

  function description.
"""function principal_angle(C::Array{Float64, 3}, lypspec::Array{Float64, 1}, cutoff=-1)
  # calculate minimum principal angle of specified subspaces in C
  # @assert length(lypspec) == size(C, 1)
  if cutoff < 1
      # first determine 'cutoff' by finding lyapunov exponent closest to zero
      # all CLV's above this are positive exponents, all below are negative exponents
      lypzero = indmin(abs.(lypspec))
    else
      # allows customly defining the stable and unstable manifolds
      lypzero = cutoff
  end
  # initializes theta matrix to contain the mimimum angle
  θ = zeros(size(C, 3))
  for t=1:size(C, 3)
    # positive CLV's go into u1, all negative CLV's go into u2
    u1 = C[:, 1:lypzero-1, t]; u2 = C[:, lypzero:size(C, 2), t]
    # then qr decomposition on u1 and u2 to form q1 and q2 (orthonormal basis for
    # space spanned by u1 and u2 respectively)
    q1, = qr(u1); q2, = qr(u2)
    # compute overlap matrix q2ᵀq1 (q2'q1)
    # obtain principal angles from svdvals of q2ᵀq1 and acos
    θ[t] = minimum(acos.(round.(svdvals(q2'*q1), 14)))
  end
  # return theta array containing minimum principal angle at each timestep
  return θ
end

# fucntion for determining the minimum angle between two subspaces over multiple
# files such that the data is contained as a pdf
"""
  principal_angle_distribution(path::String, lypspec::Array{Float64, 1},
                               nbins=100, cutoff=-1)

  function description.
"""function principal_angle_distribution(path::String, lypspec::Array{Float64, 1},
                                      nbins=100, cutoff=-1)
  # using JLD: jldopen, read, close
  if cutoff < 1
      # first determine 'cutoff' by finding lyapunov exponent closest to zero
      # all CLV's above this are positive exponents, all below are negative exponents
      lypzero = indmin(abs.(lypspec))
    else
      # allows customly defining the stable and unstable manifolds
      lypzero = cutoff
  end
  mx = π
  mx /= nbins
  dist = zeros(nbins)
  if isfile(path)
    if ismatch(r"^chunk directory.csv", path)
      for file in readcsv(path)
        file = String(file)
        println(file)
        f = jldopen(file, "r")
        try
          C = read(f, "C")
          θ = principal_angle(C, lypspec, lypzero)
          for datum in θ
            if !isapprox(datum, mx*nbins)
              dist[floor(Int, div(datum, mx)) + 1] += 1
            else
              dist[nbins] += 1
            end
          end
          finally
            close(f)
        end
      end
      dist /= sum(dist)
      return linspace(0, π, nbins), dist
     else
      println("Error: must use chunk directory.csv with this function")
      return zeros(nbins), zeros(nbins)
    end

  else
    for file in readdir(path)
      if ismatch(r"^chunk-\d*x\d*.jld", file)
        # do stuff with data containing files
        println(file)
        f = jldopen(joinpath(path, file), "r")
        try
          C = read(f, "C")
          θ = principal_angle(C, lypspec, lypzero)
          for datum in θ
            if !isapprox(datum, mx*nbins)
              dist[floor(Int, div(datum, mx)) + 1] += 1
            else
              dist[nbins] += 1
            end
          end
        finally
          close(f)
        end

      end
    end
    dist /= sum(dist)
    return linspace(0, π, nbins), dist
  end
end
=#
# end

#-----------------------------------------------------------------------------#
# functions useful for calculating CLVs
"""
  forward_evolution(timestep, lin_timestep!, datafile, lattice, delay,
                    ns, ne, gsdelay, cdelay, nsps=1)

  function description.
"""
function forward_evolution(timestep, lin_timestep!, datafile, batchlog, lattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)
  # forward evolution function goes here
  #= forward evolution consists of evolving the lattice and Q forwards in time
  the perscribed amount and then writing the contents to file.
  =#
  # creates HDF5 file to store data
  fid = h5open(datafile, "w")

  # initialize local variables
  ht = length(lattice)
  # srand(3)
  delta = zeros(ht, ne)
  for i=1:ne
    delta[i, i] = 1;
  end
  # delta = Matrix(1.0I, ht, ht) # rand(ht, ht)
  # the number of total steps for the system is the number of samples (ns)
  # * the number of steps per sample (nsps)
  numsteps = ns*nsps
  # create data handles for storing the data and allocate the necesary
  Chandle = d_create(fid, "C", datatype(Float64), dataspace(ne, ne, ns),
                     "chunk", (ne, ne, 1))
  Rhandle = d_create(fid, "R", datatype(Float64), dataspace(ne, ne, ns),
                     "chunk", (ne, ne, 1))
  Qhandle = d_create(fid, "Q", datatype(Float64), dataspace(ht, ne, ns),
                     "chunk", (ht, ne, 1))
  Jhandle = d_create(fid, "J", datatype(Float64), dataspace(ht, ht, ns),
                     "chunk", (ht, ht, 1))
  Yhandle = d_create(fid, "Y", datatype(Float64), dataspace(ht, ns*nsps),
                     "chunk", (ht, nsps))
  Rwhandle = d_create(fid, "Rw", datatype(Float64), dataspace(ne, ne, cdelay),
                  "chunk", (ne, ne, 1))
  # warm up the lattice
  for i in 1:delay
    timestep(lattice)
  end
  #= initialize  variables for evolution of delta and lattice, calculating Q
  and lypspec at all times =#
  count = 0; count2 = 0
  LT = zeros(ht, ht)
  LTJ = Matrix(1.0I, ht, ht)
  println("lattice warmed up.")
  # delay to warm up gram-schmidt (GS) vectors
  while count < gsdelay
    # LTJ = Matrix(1.0I, ht, ht)
     while count2 < nsps
       count += 1; count2 += 1
       # Yhandle[:, count] = lattice
       timestep(lattice)
       lin_timestep!(LT, lattice)
       delta = LT*delta
       LTJ = LT*LTJ
       # append!(latp, lattice[div(ht, 6)])
     end
     q, r = qr(delta)
     sgn = Matrix(Diagonal(sign.(diag(r))))
     r = sgn*r
     # Rhandle[:, :, div(count, nsps)] = r
     delta = q*sgn
     # Qhandle[:, :, div(count, nsps)] = delta
     # println(delta)
     # lypspec = (log.(diag(r)) + lypspec*(count-count2))/count
     # Jhandle[:, :, div(count, nsps)-1] = LTJ
     count2 = 0; LTJ = Matrix(1.0I, ht, ht)
   end
   count = 0; count2 = 0;
   println("GSV warmed up.")
  # create QR datapoints in middle
  lypspec = zeros(ne)
  # latp = Array([])
  LTJ = Matrix(1.0I, ht, ht)
  # evolve delta and lattice by numsteps in forward direction
  while count2 < nsps
    count += 1; count2 += 1;
    Yhandle[:, count] = lattice
    timestep(lattice)
    lin_timestep!(LT, lattice)
    delta = LT*delta
    LTJ = LT*LTJ
  end
  q, r = qr(delta)
  sgn = Matrix(Diagonal(sign.(diag(r))))
  r = sgn*r
  Rhandle[:, :, div(count, nsps)] = r
  delta = q*sgn
  Qhandle[:, :, div(count, nsps)] = delta
  # println(delta)
  lypspec = (log.(diag(r)) + lypspec*(count-count2))/count
  count2 = 0; LTJ = Matrix(1.0I, ht, ht)
  while count < numsteps
    # LTJ = Matrix(1.0I, ht, ht)
    while count2 < nsps
      count += 1; count2 += 1
      Yhandle[:, count] = lattice
      timestep(lattice)
      lin_timestep!(LT, lattice)
      delta = LT*delta
      LTJ = LT*LTJ
      # append!(latp, lattice[div(ht, 6)])
    end
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign.(diag(r))))
    r = sgn*r
    Rhandle[:, :, div(count, nsps)] = r
    delta = q*sgn
    Qhandle[:, :, div(count, nsps)] = delta
    # println(delta)
    lypspec = (log.(diag(r)) + lypspec*(count-count2))/count
    Jhandle[:, :, div(count, nsps)-1] = LTJ
    count2 = 0; LTJ = Matrix(1.0I, ht, ht)
  end
  println("collected QR data.")
  # create R data for warming up C
  count = 0; count2 = 0;
  while count < cdelay
    # LTJ = Matrix(1.0I, ht, ht)
    while count2 < nsps
      count += 1; count2 += 1
      # Yhandle[:, count] = lattice
      timestep(lattice)
      lin_timestep!(LT, lattice)
      delta = LT*delta
      LTJ = LT*LTJ
      # append!(latp, lattice[div(ht, 6)])
    end
    q, r = qr(delta)
    sgn = Matrix(Diagonal(sign.(diag(r))))
    r = sgn*r
    Rwhandle[:, :, div(count, nsps)] = r
    delta = q*sgn
    # Qhandle[:, :, div(count, nsps)] = delta
    count2 = 0; LTJ = Matrix(1.0I, ht, ht)
  end
  # finished forward evolution of lattice
  printlog(batchlog, "final u: ", sum(lattice)/size(lattice, 1))
  # write lyapunov spectrum as calculated by the R components to file
  fid["lypspec"] = lypspec
  println("the GSV Lyapunov Spectrum is:")
  println(lypspec)
  # close file to write changes
  close(fid) # to write to file
  println("finished forward evolution of CLVs")

end

"""
  C_creation(datafile, ns, nsps, cdelay)

  function description.
"""
function C_creation(datafile, batchlog, ns, nsps, cdelay)
  #to read data from file
  # using HDF5
  fid = h5open(datafile, "r+")
  Qhandle = fid["Q"];
  Rhandle = fid["R"];
  Jhandle = fid["J"];
  Yhandle = fid["Y"];
  Chandle = fid["C"];
  Rwhandle = fid["Rw"];

  # initialize items for backward evolution
  ht = size(Qhandle, 1)
  ne = size(Rhandle, 1)
  lypspecCLV = zeros(ne) # lypspec as calculated by CLV's
  converging = false
  # Vin = zeros(ht, ht); Vi = zeros(ht, ht)
  # V = zeros(R)
  # V[:, :, i] = Q[:, :, i]*Cm[:, :, i]
  # Vnorm = zeros(R)
  i2 = 0
  # warm up C vectors with backward transient Rw matrices
  C = Matrix(1.0I, ne, ne)
  idelay = cdelay
  while idelay > 1
    # renormalizes C for each vector
    for j=1:ne
      C[:, j] = C[:, j]/norm(reshape(C[:, j], ne))
    end
    # creates preceding C vector
    C = inv(reshape(Rwhandle[:, :, idelay], (ne, ne)))*C
    idelay -= 1
  end
  println("finished warming up C. Size is: ", size(C))
  # delete Rw matrices as they are no longer needed.
  o_delete(fid, "Rw")
  i = ns
  converging = false
  # Vin = zeros(ht, ht); Vi = zeros(ht, ht)
  Chandle[:, :, i] = C
  # V = zeros(R)
  # V[:, :, i] = Q[:, :, i]*Cm[:, :, i]
  # Vnorm = zeros(R)
  i2 = 0
  println("set IC for calculating V.")
  while !converging && i > 1
    if Int(rem(i, ns/10)) == 0
      println("percent completion: ", (ns-i)/ns*100);
    end
    # println("i is: ", i)
    # i2 += 1
    # renormalizes C for each vector
    for j=1:ne
      Chandle[:, j, i] = Chandle[:, j, i]/norm(reshape(Chandle[:, j, i], ne))
    end
    # Cn = C
    Chandle[:, :, i-1] = inv(reshape(Rhandle[:, :, i], (ne, ne))
                             )*reshape(Chandle[:, :, i], (ne, ne))

    # calculates the lyapunov spectrum using the CLV
    if i < ns*nsps-100
      i2 += 1
      for j=1:ne
        lypspecCLV[j] = (lypspecCLV[j]*(i2-1) +
      log(norm(reshape(Chandle[:, j, i], ne))/
          norm(reshape(Chandle[:, j, i-1], ne)))/nsps)/(i2)
      end
    end
    # Vin = Q[:, :, i]*Cn
    # Vi = Q[:, :, i-1]*C
    # if isapprox(Vin, J[:, :, i-1]*Vi)
    #   converging = true
    #   println("yay converging!")
    # end
    i -= 1
  end
  fid["lypspecCLV"] = lypspecCLV
  close(fid)
  close(batchlog)
end

"""
CLV(timestep, lin_timestep!, datafile, lattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)

Determines Covariant Lyapunov Vectors to datafile.
"""
function CLV(timestep, lin_timestep!, datafile, batchlog, lattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)
  forward_evolution(timestep, lin_timestep!, datafile, batchlog, lattice,
                             delay, ns, ne, gsdelay, cdelay, nsps)
  println("\nNow performing backward evoltion function.")
  println("------------------------------------------------------------------")
  C_creation(datafile, batchlog, ns, nsps, cdelay)
end

"""
clvQuick(timestep, lin_timestep!, datafile, batchlog, startLattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)

Determines Covariant Lyapunov Vectors, assuming computational objects are small enough to fit into memory.
"""
function clvQuick(timestep, lin_timestep!, datafile, batchlog, startLattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)
  forward_evolution(timestep, lin_timestep!, datafile, batchlog, startLattice,
                             delay, ns, ne, gsdelay, cdelay, nsps)
  println("\nNow performing backward evoltion function.")
  println("------------------------------------------------------------------")
  C_creation(datafile, batchlog, ns, nsps, cdelay)
end


"""
lyapunovSpectrumGSM(timestep, lin_timestep!, startLattice,
                           delay, ns, ne, gsdelay, cdelay, nsps=1)

Returns the Lyapunov Spectrum using the Gram-Schmidt Method for computing the
exponents, which should always fit in memory except for very large lattice sizes.

"""
function lyapunovSpectrumGSM(timestep, lin_timestep!, startLattice,
                           delay, ns, ne, gsdelay, nsps=1)

# initialize local variables
ht = length(startLattice)
lattice = copy(startLattice)
lypspec = zeros(ne) # will contain lyapunov spectrum to be returned
delta = zeros(ht, ne) # matrix of the number of exponents (ne) to determine
# initializes CLVs to orthonormal set of vectors
for i=1:ne
  delta[i, i] = 1;
end

# the number of total steps for the system is the number of samples (ns)
# times the number of steps per sample (nsps)
numsteps = ns*nsps
# warm up the lattice
for i in 1:delay
  timestep(lattice)
end
#= initialize  variables for evolution of delta and lattice, calculating Q
and lypspec at all times =#
count = 0; count2 = 0
LT = zeros(ht, ht)
println("lattice warmed up.")
# delay to warm up gram-schmidt (GS) vectors
while count < gsdelay
   while count2 < nsps
     count += 1; count2 += 1
     timestep(lattice)
     lin_timestep!(LT, lattice)
     delta = LT*delta
   end
   q, r = qr(delta)
   sgn = Matrix(Diagonal(sign.(diag(r))))
   r = sgn*r
   delta = q*sgn
   count2 = 0;
 end
 count = 0; count2 = 0;
 println("GSV warmed up.")
# calculate Lyapunov Spectrum for the given number samples (ns) and
# given spacing (nsps)
while count < numsteps
  while count2 < nsps
    count += 1; count2 += 1
    timestep(lattice)
    lin_timestep!(LT, lattice)
    delta = LT*delta
  end
  q, r = qr(delta)
  sgn = Matrix(Diagonal(sign.(diag(r))))
  r = sgn*r
  delta = q*sgn
  lypspec = (log.(diag(r)) + lypspec*(count-count2))/count
  count2 = 0
end
# do not need R data for warming up C, since we are doing Gram-Schmidt Method
# finished evolution of lattice
println("final u: ", sum(lattice)/size(lattice, 1))
println("the GSV Lyapunov Spectrum is:")
println(lypspec)
return lypspec
end



#-----------------------------------------------------------------------------#
# functions useful for calculating Domination of Osledet Splitting
"""
  DOS(fid::HDF5.HDF5File, window::Int64)

Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(fid, window::Int64) # ::HDF5.HDF5File) problem with HDF5 type currently, need to fix
  # fid = h5open(datafile, "r")
  Rhandle = fid["R"];
  Chandle = fid["C"];
  ns = size(Rhandle, 3);
  ne = size(Chandle, 1);
  nu = zeros(ne, ne);
  nutemp = zeros(ne, ne);
  CLV_growth = zeros(ne);
  @assert(window < ns && window > 0) # checks that window is not too large
  for i=1:ns-window-1
    if Int(rem(i, ns/10)) == 0
      println("percent completion: ", i/ns*100);
    end
    # added for loop to go have the instantaneous CLV growth averaged over window
    CLV_growth = zeros(ne);
    for j=0:window-1
      C1 = reshape(Chandle[:, :, i+j], (ne, ne))
      R2 = reshape(Rhandle[:, :, i+1+j], (ne, ne))
      CLV_growth += CLV_instant_growth(C1, R2);
    end
    CLV_growth /= window;
    nutemp = DOS_violations(CLV_growth)
    nu += nutemp;
  end
  nu /= (ns-1)
  return nu
end
"""
  DOS(datafile::String, window::Int64)

Determines the Domination of Osledet Splitting based on recorded C and R matrices.
"""
function DOS(datafile::String, window::Int64)
  fid = h5open(datafile, "r")
  nu = DOS(fid, window)
  close(fid)
  return nu
end
"""
  DOS\\_violations(CLV_growth::Array{Float64, 1})

Determines nu matrix of total violations from list of instantaneous growth.
"""
function DOS_violations(CLV_growth::Array{Float64, 1})
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

"""
  CLV_instant_growth(C1::Array{Float64, 2}, R2::Array{Float64, 2})

Determines Lyapunov Exponent growth as a result of one timestep.
"""
function CLV_instant_growth(C1::Array{Float64, 2}, R2::Array{Float64, 2})
  C2 = R2*C1
  ne = size(C1, 1)
  CLV_growth = zeros(ne)
  for i=1:ne
    CLV_growth[i] = (norm(reshape(C2[:, i], ne))/
    norm(reshape(C1[:, i], ne)))
  end
  return CLV_growth #::Array{Float64, 1} size(C, 1)
end

"""
  Kaplan_York_Dimension(lyapunov_spectrum::Array{Float64, 1})

  Determines the Kaplan York Dimesion (KYD) based on an input lyapunov spectrum.
"""
function Kaplan_York_Dimension(lyapunov_spectrum::Array{Float64, 1})
    ne = size(lyapunov_spectrum, 1);
    tempsum = 0.0;
    KYD = -1.0;
    i = 1;
    KYD_bool = true;
    while ( i <= ne )
      tempsum += lyapunov_spectrum[i];
      if tempsum < 0
        # linear interpolation to determine fractional dimension
        # y = mx + b
        # 0 = (lypspec[i] - lypspec[i-1])/1*dim -(i-1)
        KYD = (i-1) + sum(lyapunov_spectrum[1:(i-1)])/abs(lyapunov_spectrum[i]);
        break
      end
      i += 1;
    end

    if KYD == -1.0
      println("Kaplan York Dimension Does not exist.")
    end

    return KYD
end


"""
  clean_log(a::Array{Float64, 2})

  Computes the log of each element of the array and clean -Inf to small number.
"""
function clean_log(a::Array{Float64, 2}, base)
    n1 = size(a, 1);
    n2 = size(a, 2);
    nu = log.(base, a);
    min0 = 1.0;
    # finds the minimum which isn't -Inf
    for i in 1:n1
      for j in 1:n2
        if nu[i, j] < min0 && nu[i, j] != -Inf
          min0 = nu[i, j]
        end
      end
    end

    for i in 1:n1
      for j in 1:n2
        if nu[i, j] == -Inf
          nu[i, j] = min0
        end
      end
    end
    return nu
end

#-----------------------------------------------------------------------------#
# functions for calculating angle between CLVs along with subspaces in the case
# that the data is stored as HDF5 files
"""
  principal_angle(C::Array{Float64, 3}, lypspec::Array{Float64, 1}, cutoff=-1)

  Computes the principal angle for each timestep.
"""
function principal_angle(C::Array{Float64, 3}, lypspec::Array{Float64, 1}, cutoff=-1)
  # calculate minimum principal angle of specified subspaces in C
  # @assert length(lypspec) == size(C, 1)
  if cutoff < 1
      # first determine 'cutoff' by finding lyapunov exponent closest to zero
      # all CLV's above this are positive exponents, all below are negative exponents
      lypzero = indmin(abs.(lypspec))
    else
      # allows customly defining the stable and unstable manifolds
      lypzero = cutoff
  end
  # initializes theta matrix to contain the mimimum angle
  θ = zeros(size(C, 3))
  for t=1:size(C, 3)
    # positive CLV's go into u1, all negative CLV's go into u2
    u1 = C[:, 1:lypzero-1, t]; u2 = C[:, lypzero:size(C, 2), t]
    # then qr decomposition on u1 and u2 to form q1 and q2 (orthonormal basis for
    # space spanned by u1 and u2 respectively)
    q1, = qr(u1); q2, = qr(u2)
    # compute overlap matrix q2ᵀq1 (q2'q1)
    # obtain principal angles from svdvals of q2ᵀq1 and acos
    θ[t] = minimum(acos.(round.(svdvals(q2'*q1), 14)))
  end
  # return theta array containing minimum principal angle at each timestep
  return θ
end

"""
  principal_angle(C::Array{Float64, 2}, lypspec::Array{Float64, 1}, cutoff=-1)

  Computes the principal angle for C matrix unless cutoff is specified.
"""
function principal_angle(C::Array{Float64, 2}, lypspec::Array{Float64, 1}, cutoff=-1)
    if cutoff < 1
        lypzero = indmin(abs.(lypspec))
    else
        lypzero = cutoff
    end
    theta = 1000
    u1 = C[:, 1:lypzero-1]; u2 = C[:, lypzero:size(C, 2)]
    q1, = qr(u1); q2, = qr(u2)
    theta = minimum(acos.(round.(svdvals(q2'*q1), 14)))
    return theta
end

"""
  angle_between(C::Array{Float64, 3}, v1::Int, v2::Int)

  Computes the angle between the vectors v1 and v2 in C for each timestep.
"""
function angle_between(C::Array{Float64, 3}, v1::Int, v2::Int)
  θ = zeros(size(C, 3))
  for t=1:size(C, 3)
    vec1 = C[:, v1, t]; vec2 = C[:, v2, t]
    # positive CLV's go into u1, all negative CLV's go into u2
    θ[t] = acos(round(dot(vec1, vec2)/(norm(vec1)*norm(vec2)), 14))
  end
  # return theta array containing minimum principal angle at each timestep
  return θ
end

"""
  angle_distribution(C::Array{Float64, 3}, v1::Int, v2::Int, nbins=100)

  Computes the distribution of angle between the vectors v1 and v2 in C.
"""
function angle_distribution(C::Array{Float64, 3}, v1::Int, v2::Int, nbins=100)
  # Works with current C matrix in memory
  mx = π
  mx /= nbins
  dist = zeros(nbins)
  lengthV = size(C, 3)
  θ = angle_between(C, v1, v2)
  indexstart = 1
  indexend = lengthV
  for index in indexstart:indexend
    if !isapprox(θ[index], mx*nbins)
      dist[floor(Int, div(θ[index], mx)) + 1] += 1
    else
      dist[nbins] += 1
    end
  end
  dist /= sum(dist)*pi/nbins
  return linspace(0, π, nbins), dist
end

"""
  angle_distribution(fid::HDF5.HDF5File, v1::Int, v2::Int, nbins=100)

  Computes the distribution of angle between the vectors v1 and v2 in C in
  file fid.
"""
function angle_distribution(fid, v1::Int, v2::Int, nbins=100)
  #fid::HDF5.HDF5File - need to incorporate in future version
  # with file already open with fid
  mx = π
  mx /= nbins
  dist = zeros(nbins)
  # V's from 10% to 90%
  # Qhandle = fid["Q"]
  Chandle = fid["C"]
  lengthV = size(Chandle, 3)
  ht = size(Chandle, 2)
  indexstart = Int(round(lengthV*0.1))
  indexend = Int(round(lengthV*0.9))
  for index in indexstart:indexend
    V = reshape(Chandle[:, :, index], (ht, ht))
    vec1 = V[:, v1]; vec2 = V[:, v2]
    θ = acos(round(dot(vec1, vec2)/(norm(vec1)*norm(vec2)), 14))
    if !isapprox(θ, mx*nbins)
      dist[floor(Int, div(θ, mx)) + 1] += 1
    else
      dist[nbins] += 1
    end
  end
  dist /= sum(dist)*pi/nbins
  return linspace(0, π, nbins), dist
end

"""
  principal_angle_distribution(fid::HDF5.HDF5File,
                                      lypspec::Array{Float64, 1},
                                      nbins=100, cutoff=-1)

  Computes the distribution of principal angles in the file fid.
"""
function principal_angle_distribution(fid, # ::HDF5.HDF5File,
                                      lypspec::Array{Float64, 1},
                                      nbins=100, cutoff=-1)
  # using JLD: jldopen, read, close
  if cutoff < 1
      # first determine 'cutoff' by finding lyapunov exponent closest to zero
      # all CLV's above this are positive exponents, all below are negative exponents
      lypzero = indmin(abs.(lypspec))
    else
      # allows customly defining the stable and unstable manifolds
      lypzero = cutoff
  end
  mx = π
  mx /= nbins
  dist = zeros(nbins)
  # V's from 10% to 90%
  Qhandle = fid["Q"]
  Chandle = fid["C"]
  lengthV = size(Chandle, 3)
  ht = size(Chandle, 2)
  indexstart = Int(round(lengthV*0.1))
  indexend = Int(round(lengthV*0.9))
  for index in indexstart:indexend
    V = reshape(Qhandle[:, :, index], (ht, ht))*reshape(Chandle[:, :, index], (ht, ht))
    θ = principal_angle(V, lypspec, lypzero)
    if !isapprox(θ, mx*nbins)
      dist[floor(Int, div(θ, mx)) + 1] += 1
      else
      dist[nbins] += 1
    end
  end
  dist /= sum(dist)
  return linspace(0, π, nbins), dist
end

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

"""
  zero_index(lypspec::Array{Float64, 1})

  Returns index at which lyapunov spectrum crosses zero.
"""
function zero_index(lypspec::Array{Float64, 1})
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

end # module
