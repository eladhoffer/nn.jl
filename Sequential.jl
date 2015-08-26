type Sequential <: Layer
  layers::Vector
  output::Array
  gradInput::Array
  Sequential(layers) = new(layers)
end

function updateOutput(m::Sequential, input)
    currentOutput = input
    for i = 1:length(m.layers)
      currentOutput = updateOutput(m.layers[i], currentOutput)
    end
    m.output = currentOutput;
    return m.output
end

function updateParameters(m::Sequential, updateFunc)
  for i = 1:length(m.layers)
    updateParameters(m.layers[i], updateFunc)
  end
end

function zeroGradParameters(m::Sequential)
  for i = 1:length(m.layers)
    zeroGradParameters(m.layers[i])
  end
end

function updateGradInput(m::Sequential, input, gradOutput)
   currentGradOutput = gradOutput
   currentLayer = m.layers[length(m.layers)]
   for i = (length(m.layers)-1):-1:1
      previousLayer = m.layers[i]
      currentGradOutput = updateGradInput(currentLayer, previousLayer.output, currentGradOutput)
      currentLayer = previousLayer
   end
   currentGradOutput = updateGradInput(currentLayer, input, currentGradOutput)
   m.gradInput = currentGradOutput
   return m.gradInput
end

function accGradParameters(m::Sequential, input, gradOutput, scale)
   currentGradOutput = gradOutput
   currentLayer = m.layers[length(m.layers)]
   for i = (length(m.layers)-1):-1:1
      previousLayer = m.layers[i]
      accGradParameters(currentLayer, previousLayer.output, currentGradOutput)
      currentGradOutput = currentLayer.gradInput
      currentLayer = previousLayer
   end
   accGradParameters(currentLayer, input, currentGradOutput)
end

function backward(m::Sequential, input, gradOutput, scale)
   currentGradOutput = gradOutput
   currentLayer = m.layers[length(m.layers)]
   for i = (length(m.layers)-1):-1:1
      previousLayer = m.layers[i]
      currentGradOutput = backward(currentLayer, previousLayer.output, currentGradOutput)
      currentLayer.gradInput = currentGradOutput
      currentLayer = previousLayer
   end
   currentGradOutput = backward(currentLayer, input, currentGradOutput)
   m.gradInput = currentGradOutput
   return m.gradInput
end
