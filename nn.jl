module nn
export Layer,
       Sequential,
       Linear,
       Pointwise,
       ReLU,
       LogSoftMax,
       Criterion,
       ClassNLLCriterion

export forward,
       backward,
       reset!,
       updateOutput,
       updateParameters,
       updateGradInput,
       zeroGradParameters


include("Layer.jl")
include("Linear.jl")
include("Pointwise.jl")
include("Sequential.jl")
include("LogSoftMax.jl")
include("Criterion.jl")
end
