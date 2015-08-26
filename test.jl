using nn
using MNIST
#linear = Linear(3,5)
#x = rand(Float16, (128,3))
#y = forward(linear, x)


model =Sequential({
                    Linear(28*28,128),
                    #ReLU(false),
                    Linear(128,10),
                    LogSoftMax()
                    })

criterion = ClassNLLCriterion()


#y = forward(model, rand(Float32,128,10))
#z = backward(model, rand(Float32,128,10), y)

function naiveSGD(lr)
  updateFunc = function(weight::Array, bias::Array, gradWeight::Array, gradBias::Array)
    weight += lr * gradWeight
    bias += lr * gradBias
  end
  return updateFunc
end

#updateParameters(model, naiveSGD(0.01))
trainX, trainY = traindata()
testX, testY = testdata()

trainX = transpose(trainX)
testX = transpose(testX)
trainY += 1
testY += 1

batchSize = 128
EPOCHES = 10

for epoch=1:EPOCHES
  loss = 0.0
  println("Epoch $epoch")
  for i=1:batchSize:(size(trainX,1)-batchSize)
    x = trainX[i:i+batchSize-1, :]
    yt = trainY[i:i+batchSize-1]
    y = forward(model, x)
    loss += forward(criterion, y, yt)
    dE_dy = backward(criterion, y, yt)
    backward(model, x, dE_dy)
    #println(mean(model.layers[2].gradInput))
    updateParameters(model, naiveSGD(0.001))
    zeroGradParameters(model)
  end
  println(loss / floor(size(trainX,1)/batchSize))
end
