type Linear{T} <: Layer{T}
  nInputDims
  nOutputDims
  weight::Array{T}
  bias::Array{T}
  output::Array{T}
  gradInput::Array{T}
  gradBias::Array{T}
  gradWeight::Array{T}
  Linear(nInputDims,nOutputDims) =
  (
  x = new(nInputDims,nOutputDims);
  x.weight = rand(T, nInputDims,nOutputDims);
  x.bias = rand(T, nOutputDims);
  x.gradWeight = zero(x.weight);
  x.gradBias = zero(x.bias);
  reset!(x);
  x
  )
end

function updateOutput(m::Linear, input)
  m.output = input * m.weight .+ m.bias';
  return m.output
end


function updateGradInput(m::Linear, input, gradOutput)
  m.gradInput = gradOutput * m.weight';
  return m.gradInput
end

function accGradParameters(m::Linear, input, gradOutput)
  BLAS.gemm!('N','N', 1, gradOutput, weight, m.gradInput)
  m.gradBias += sum(gradOutput, 2)
end

accGradParameters

function reset!(m::Linear)
  stdv = 1/sqrt(size(m.weight,2))
  reset!(m::Linear, stdv)
end

function reset!(m::Linear, stdv)
  stdv *= sqrt(3)
  rand!(m.weight)
  rand!(m.bias)
  m.weight = m.weight*stdv - stdv
  m.bias = m.bias*stdv - stdv
end

function updateParameters(m::Linear, updateFunc)
  updateFunc(m.weight, m.bias, m.gradWeight, m.gradBias)
end
function zeroGradParameters(m::Linear)
  fill!(m.gradWeight, 0)
  fill!(m.gradBias, 0)
end
Linear(nInputDims,nOutputDims) = Linear{Float32}(nInputDims,nOutputDims)
