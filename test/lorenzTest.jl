# Define Runge-Kutta 4 timestepping function
function rk4(v,X,h,n)
    # RK4   Runge-Kutta scheme of order 4
    #   performs n steps of the scheme for the vector field v
    #   using stepsize h on each row of the matrix X
    #   v maps an (m x d)-matrix to an (m x d)-matrix

    for i = 1:n
        k1 = v(X);
        k2 = v(X + h/2*k1);
        k3 = v(X + h/2*k2);
        k4 = v(X + h*k3);
        X = X + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
    return X
end


using Plots
# Get the Lorenz System running
sigma = 10; rho = 28; beta = 8/3;
v = x -> [sigma*(x[2]-x[1]),              # the Lorenz system
          rho*x[1]-x[2]-x[1].*x[3],
          x[1].*x[2]-beta*x[3]];
f = x -> rk4(v,x,0.01,1);                       # f is the time-0.01-map
ns = 2000;
y = zeros(ns, 3);
y[1,1:3] = rand(3);                             # random initial point

# initialize a 3D plot with 1 empty series
plt = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz Attractor", marker = 2)

# build an animated gif by pushing new points to the plot, saving every 10th frame
@gif for k=1:ns-1
    y[k+1,:] = f(y[k,:]);
    push!(plt, y[k+1, 1], y[k+1, 2], y[k+1, 3])
end every 10
