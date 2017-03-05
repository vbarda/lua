local gradient_descent = {}

t = require('torch')

local function cost_function(X, y, b)
  -- take X, y and b (betas) and calculate the value of cost function
  m = X:size(1)
  errors = X * b - y
  return errors:pow(2):sum() / 2 / m
end

local function step(X, y, b, alpha)
  -- calculate gradient descent step (update betas and return cost)
  m = X:size(1)
  b = b - (alpha / m) * (X:transpose(1, 2) * (X * b - y))
  return b, cost_function(X, y, b)
end

function gradient_descent.gradient_descent(X, y, b, alpha, precision,
                                           max_iterations)
  -- iterate until either precision is met or max_iterations is reached
  local costs = {}
  n = X:size(2)
  if not b then
    b = t.zeros(n, 1)
  end
  b, cost = step(X, y, b, alpha)
  i = 1
  costs[i] = cost
  while true do
    b, cost = step(X, y, b, alpha)
    chg = cost - unpack(costs, #costs, #costs)
    i = i + 1
    if (i == max_iterations) or (math.abs(chg) <= precision) then break end
    costs[i] = cost
  end
  return {b=b, costs=costs}
end

return gradient_descent
