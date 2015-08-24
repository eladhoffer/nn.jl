abstract Pointwise{T} <: Layer{T}

type ReLU{T} <: Pointwise{T}
  inplace::Bool
  pForward
  pBackward
  output::Array{T}
  gradInput::Array{T}
  ReLU(inplace::Bool = false) =
  (
  m = new(inplace);
  m.pForward = x -> x > 0 ? x : 0;
  m.pBackward = x-> x > 0 ? 1 : 0;
  m
  )
end

function updateOutput(m::Pointwise, input)
  m.inplace ? m.output = map!(m.pForward, x) : m.output = map(m.pForward, input);
  return m.output
end

function updateGradInput(m::Pointwise, input, gradOutput)
  m.gradInput = gradOutput .* map(m.pBackward, input);
  return m.gradInput
end

ReLU(inplace) = ReLU{Float32}(inplace)
