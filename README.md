## Implementation of [**ResNet**](https://arxiv.org/abs/1512.03385) in PyTorch. 

Currently trainer supports only [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

### **Requirements**
- PyTorch v0.4.0
- torchvision
- numpy
- scipy
- tqdm


### **Performance**:
| Models                      | Top-1 Accuracy |
| :---                        | ---:           |
| ResNet-10 without Identity  | 94.2% |
| ResNet-10 with Identity     | 94.4% |
| ResNet-18 without Identity  | 94.8% |
| ResNet-18 with Identity     | 95.0% |
| ResNet-34 without Identity  | 93.5% |
| ResNet-34 with Identity     | 95.5% |


### **Comments and Implementation Details**:
- In most ResNet implementations that I encountered on the web, there were additional convolutional layers used to represent the identity mappings. Even some included batch normalization layers. I believe this contradicts with the idea that deeper neural nets struggle representing identity mappings. Therefore here, I use average pooling for downsampling and add extra feature maps which contain all-zeros to increase the number of feature planes.
- When I work on the validation set, I noticed that identity mappings in residual blocks stabilizes the training in the long run. Yet, all trainning & validation & test accuracies tend to converge for ResNet-10 and ResNet-18. Performance gap becomes noticable when depth increases, i.e., ~2% on ResNet-34.
- Default settings start with a learning rate of 0.1 and the learning rate is multiplied by 0.1 after every 100 epochs.
- You can benefit from `tensorboard` if you install `tensorboard_logger` via pip (skip this if you already use Tensorflow) and set the *tensorboard* argument to *True*.
- To run on CPU or GPU supporting cuda set, *device* argument to *'cpu'* or *'cuda'* respectively.
- Don't forget to arange folders for `exp_dir` and `log_dir`.

### **What the learned features look like**:
To examine the representations learned by a ResNet on the Cifar-10:
1. I extracted the features of the test set from the ResNet-34, which yield 95% test set accuracy.
2. For each feature:
   - I sorted all the features based on their magnitudes.
   - I took the least and the most relevant 10 images, and formed the below (big!) image. The left and the right halves contain the least and the most relevant ones, respectively.

For the most rows, the most relevant ones seem to be related conceptually. But still, I believe such a visual inspection cannot be well-grounded.

![alt text](images/cifar10_images_selected_by_features.png "Cifar-10 images retrieved based on their feature magnites.")

### **TODO**:
- [ ] Add Cifar100 support
- [ ] Share results for ResNet-34 and ResNet-50
