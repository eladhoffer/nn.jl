type LogSoftMax{T} <: Layer{T}
  output::Array{T}
  gradInput::Array{T}
  LogSoftMax() = new()
end

function updateOutput(m::LogSoftMax, input)
    m.output = input .- log(sum(exp(input),2))
    return m.output
end
function updateGradInput(m::LogSoftMax, input, gradOutput)
    m.gradInput = gradOutput - exp(m.output).*sum(gradOutput, 2)
    return m.gradInput
end

LogSoftMax() = LogSoftMax{Float32}()
