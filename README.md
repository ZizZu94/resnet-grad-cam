# ResNet18 Grad-Cam

In this notebook we will try to generate the Grad-Cam images of a pretrianed ResNet-18 model.

Grad-Cam provides us with a way to look into what particular parts of the image influenced the whole model’s decision for a specifically assigned label. It is particularly useful in analyzing wrongly classified samples. The Grad-CAM algorithm is very intuitive and reasonably simple to implement.

The intuition behind the algorithm is based upon the fact that the model must have seen some pixels (or regions of the image) and decided on what object is present in the image. Influence in the mathematical terms can be described with a gradient. On the high-level, that is what the algorithm does. It starts with finding the gradient of the most dominant logit with respect to the latest activation map in the model. We can interpret this as some encoded features that ended up activated in the final activation map persuaded the model as a whole to choose that particular logit (subsequently the corresponding class). The gradients are then pooled channel-wise, and the activation channels are weighted with the corresponding gradients, yielding the collection of weighted activation channels. By inspecting these channels, we can tell which ones played the most significant role in the decision of the class.

## References
[Implementing Grad-CAM in PyTorch](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
[Transfer Learning with PyTorch CNNs](https://python.plainenglish.io/part-2-transfer-learning-with-pytorch-cnns-for-beginners-2b5d247ac311)
