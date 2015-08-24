using nn

linear = Linear(3,5)
x = rand(Float16, (128,3))
y = forward(linear, x)


model =Sequential({
                    Linear(10,5),
                    ReLU(false),
                    Linear(5,2),
                    LogSoftMax()
                    })
size(model.layers)


y = forward(model, rand(Float32,128,10))
println(y)

z = backward(model, rand(Float32,128,10), y)
