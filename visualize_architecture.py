from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model

model = LeNet.build(28, 28, 1, 10)
print(model.summary())
plot_model(model, to_file="lenet.png", show_shapes=True)