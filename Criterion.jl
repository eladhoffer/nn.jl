abstract Criterion{T} <: Layer{T}

type ClassNLLCriterion{T} <: Layer{T}
  output::Array{T}
  gradInput::Array{T}
end

function updateOutput(m::ClassNLLCriterion, input, target)
  m.output = Array{typeof(x[1]),size(x,1)}
  for i=1:size(input,1)
    m.output[i] = -input[i][target[i]]
  end
  return m.output
end
function updateGradInput(m::ClassNLLCriterion, input, gradOutput)
  m.gradInput = zeros(gradOutput)
  for i=1:size(gradOutput,1)
    m.gradInput[i][target] = -1
  end
  m.gradInput .*= gradOutput
  return gradInput
end
