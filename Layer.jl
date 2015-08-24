abstract Layer{T}

function forward(m::Layer, input)
  return updateOutput(m, input)
end

function backward(m::Layer, input, gradOutput)
  updateGradInput(m, input, gradOutput)
  if method_exists(accGradParameters, (typeof(m),Any))
    accGradParameters(m, input, gradOutput)
  end
  return m.gradInput
end
