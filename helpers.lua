local helpers = {}

require('csvigo')
t = require('torch')

local function column_to_tensor( column )
  -- take a column from csvigo output and convert it to Tensor
  local _table = {}
  for _, x in ipairs(column) do
    table.insert(_table, tonumber(x))
  end
  return t.Tensor(_table):reshape(#_table, 1)
end

local function add_intercept_column( tensor )
  m = tensor:size(1)
  return t.ones(m, 1):cat(tensor)
end

function helpers.read_csv_into_X_y( path, y_col, add_intercept )
  --[[ load the csv into X and y. if y_col is not specified,
    X will be the first n-1 columns, and y - nth
  --]]
  local col_names = {}; X_tensors = {}
  -- load the raw csv
  d = csvigo.load(path)
  -- create table of column names
  for col_name, col in pairs(d) do
    table.insert(col_names, col_name)
  end

  if not y_col then
    y_col = unpack(col_names, #col_names, #col_names)
  end
  -- concat X_cols into one Tensor
  for _, col_name in ipairs(col_names) do
    if col_name ~= y_col then
      table.insert(X_tensors, column_to_tensor(d[col_name]))
    end
  end
  -- concatenate X columns into a single Tensor
  X = t.cat(X_tensors)
  -- add column of ones if needed
  if add_intercept then
    X = add_intercept_column(X)
  end
  y = column_to_tensor(d[y_col])
  return X, y
end

return helpers
