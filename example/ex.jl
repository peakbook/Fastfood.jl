using Fastfood
using Gadfly

# generate training dataset (two spirals)
data = zeros(2, 97)
for i in 0:96
    angle = i * pi / 16.0
    radius = 6.5 * (104.0 - i) / 104.0
    data[:, i+1] = radius * [sin(angle), cos(angle)]
end
X_train = hcat(data, -data)
Y_train = hcat(ones(1,size(data,2)), -ones(1,size(data,2)))

# train 
sgm = 3.0
N = 1000
param = FastfoodParam(N, size(X_train,1))
Kern = FastfoodKernel(X_train, param, sgm=sgm)
W = Y_train * pinv(Kern)  # least squares method

# generate test dataset
xrange = yrange = -6:0.1:6.0
X = repmat(xrange', length(yrange), 1)
Y = repmat(yrange, 1, length(xrange))
X_test = hcat(X[:], Y[:])'

# test
Y_test = W * FastfoodKernel(X_test, param, sgm=sgm)
Y_test = reshape(Y_test, length(xrange), length(yrange))'

# visualize
p = plot(
         layer(x=xrange, y=yrange, z=Y_test, Geom.contour(levels=[0.0])),
         layer(x=data[1, :], y=data[2,:], Geom.point, Theme(default_color=colorant"skyblue")),
         layer(x=-data[1, :], y=-data[2,:], Geom.point, Theme(default_color=colorant"salmon")),
         Coord.cartesian(xmin=-6, xmax=6, ymin=-6, ymax=6),
)
draw(SVG("out.svg", 4inch, 4inch), p)

