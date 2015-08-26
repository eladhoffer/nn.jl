abstract Criterion{T} <: Layer{T}

type ClassNLLCriterion{T} <: Layer{T}
  output
  gradInput::Array{T}
  ClassNLLCriterion() = new()
end

function forward(m::ClassNLLCriterion, input, target)
  m.output = 0.0
  for i=1:size(input,1)
    m.output += -input[i,target[i]]
  end
  m.output /= size(input,1)
  return m.output
end
function backward(m::ClassNLLCriterion, input, target)
  m.gradInput = zeros(input)
  for i=1:size(input, 1)
    m.gradInput[i, target[i]] = -1
  end
  return m.gradInput
end

ClassNLLCriterion() = ClassNLLCriterion{Float32}()
