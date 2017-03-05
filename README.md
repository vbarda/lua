### Experiments in lua

Current functionality:

 * loading `csv` into X, y tensors

 ```
 h = require('helpers')
 X, y = h.read_csv_into_X_y(path_to_csv, 'y', true)
 ```

 * simple gradient descent

 ```
 g = require('gradient_descent')
 results = g.gradient_descent(X, y, nil, .005, .00001, 1000)
 ```
