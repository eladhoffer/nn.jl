type Sequential <: Layer
  layers::Vector
  output::Array
  Sequential(layers) = new(layers)
end

function updateOutput(m::Sequential, x)
    output = x
    for i = 1:length(m.layers)
      output = updateOutput(m.layers[i], output)
    end
    m.output = output;
    return m.output
end

