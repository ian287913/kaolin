# cubic hermite spline implemented in PyTorch, make it available to be optimized by torch autograd
# from https://gist.github.com/chausies/c453d561310317e7eda598e229aea537
import torch as T

def h_poly_helper(tt):
  A = T.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

# m is tangent
def interp_func_with_tangent(x, y, m, xs):
  "Returns integral of interpolating function"
  if len(y)==1: # in the case of 1 point, treat as constant function
    return y[0] + T.zeros_like(xs)
  I = T.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  hh[0] = hh[0].cuda()
  hh[1] = hh[1].cuda()
  hh[2] = hh[2].cuda()
  hh[3] = hh[3].cuda()

  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def test_interpolation():
  # define keys
  x = T.linspace(0, 1, 2)
  y = T.linspace(0, 1, 2)
  m = T.linspace(0, 0, 2)
  # define ts
  xs = T.linspace(0, 1, 5)

  ys = interp_func_with_tangent(x, y, m, xs)

  for yss in ys:
    print(f"{yss}")

# # Example
# # See https://i.stack.imgur.com/zgA0s.png for resulting image
# if __name__ == "__main__":
#   test_interpolation()
#   return
#   import matplotlib.pylab as P # for plotting
#   x = T.linspace(0, 6, 7)
#   y = x.sin()
#   xs = T.linspace(0, 6, 101)
#   ys = interp(x, y, xs)
#   Ys = integ(x, y, xs)
#   P.scatter(x, y, label='Samples', color='purple')
#   P.plot(xs, ys, label='Interpolated curve')
#   P.plot(xs, xs.sin(), '--', label='True Curve')
#   P.plot(xs, Ys, label='Spline Integral')
#   P.plot(xs, 1-xs.cos(), '--', label='True Integral')
#   P.legend()
#   P.show()