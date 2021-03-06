# EECS738 Project3 Says One Neuron To Another
EECS738 Machine Learning course project3. Two deep neural networks have been implemented: a Multilayer Perceptron (MLP) with multiple linear layers and a typical Convolutional Neural Network (CNN). Two image datasets are selected: **MNIST** handwriting characters and **GTSRB** traffic sign recognition. Only third-party packages used in all implementations are **numpy**, **pandas**, **python-mnist** and **Pillow**.

## Function Checklist
These functions are implemented manually in this project in a detailed manner:
* Fully-Connected layers.
* Convolutional layers.
* Batch Normalization 2d & 1d.
* Dropout.
* Maxpooling.
* Activation functions: Sigmoid and ReLU.
* Data loading and data prepossessing (normalization).
* Forwarding and backpropagation using BGD.
* Cross-entropy loss.


## Dataset
* [Modified National Institute of Standards and Technology database (MNIST)](http://yann.lecun.com/exdb/mnist/). This is possibly one of the most commonly-used and simplest dataset in image tasks in machine learning, which combines with different images of hand-written digits. The sizes of training dataset and testing dataset are 60,000 and 10,000 each. Though the attached webpage link seems not to be working, the dataset can be directly accessed by package [python-mnist](https://github.com/sorki/python-mnist). Images and grey-scale with a size of 28\*28 each. Ten labels in total range from 0, 1, 2... to 9. 
* [The German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html). This is a more challenging dataset with images of traffic signs captured in real street views. Each image contains exactly one traffic sign, usually located right at the middle. There are 43 labels in total, representing different types of traffic signs, and more than 50,000 samples. Recognition of these signs is much more difficult and practically significant, as many of the signs have similar shapes and colors. 


## Ideas and Thinking
* **DNN**: In my opinion, deep neural networks are basically a complex but repetitive variant of weighted function with non-linear operators added. What deep learning does is to tweak the weights gradually by iterating over a large (usually) number of examples. If taking a task as a process of building sand castles, the precise solution will be like building by crafting structures and shapes, and the deep learning will be like flapping and rubbing by frequently looking at samples. Outcome of deep learning are set of parameters to mimic and map certain features, known as saliency, to high confidence scores in exact dimension. This 'learning' process imitates the process of human learning (recognition or understanding), setting up correlations between 'name/meaning' and 'features' through repetitively getting exposed to examples and strengthening neuron connections.  
* **Activation Functions**: An important component of deep learning is activation function, which significantly accounts for deep learning's success. The non-linear properties of deep learning are largely contributions of activations. What activations really does is to differentiate neurons to respond to unique features. After activation, some of the neurons will be silent for specific features, while others will not. In this way, the deep neural networks are no longer an inflexible combination of weighted values, but a instead dynamic system where each unit has a 'switch'. This is done mainly by the cut-off (e.g. ReLU) or downgrade (e.g. Sigmoid) in gradients in backpropagation. Once a neuron get silent for certain feature distribution, little changes will be witnessed in its parameters when updating. In forwarding cases, different parts of neurons will 'switch on' for different feature distribution, which greatly increase the fitting ability of the function in limited parameter space.
* **Backpropagation**: Backpropagation is a rather simple but important process. By calculating the partial gradient of the model output over parameters, the parameters are updated from layer to layer to increase highly-activated neurons' contribution to current label, and decrease noises from unrelated neurons.
* **Testing**: After iterations over training samples, the model parameters are tuned to distribution of the task domain, which expects the ability to guide new examples from the same domain into explainable outputs. Different from training, some operators, e.g. dropout and batch normalization, tend to stay static in testing phase to expect stable behaviors, e.g. to eliminate quantity influence and random errors.

## Setup
### Environment
* Python 3.9
* MacOS or Linux

### Package
* Recommend setting up a virtual environment. Different versions of packages may cause unexpected failures.
* Install packages in **requirements.txt**.
```bash
pip install -r requirements.txt
``` 
* I've updated my environment to python3.9 and there is inconsistency between my previous course projects and this one. Sorry for any possible troubles.

## Usage
### Positional & Optional Parameters
* **data**: Dataset name (task) to choose from 'mnist' and 'gtsrb'.
* **-p**: Whether to load pre-trained model parameters. The parameters for MNIST model is saved in ['./paras'](https://github.com/liuzey/EECS738_Project3/tree/main/paras). The parameters for GTSRB model is saved in ['./paras_save'](https://github.com/liuzey/EECS738_Project3/tree/main/paras_save). (Default: False).
* **-s**: Whether to save the trained model parameters (Default: False).

### Example
```bash
python main.py 'gtsrb' -s 1 -p 1
```
* 'gtsrb': GTSRB recognition task.
* -p: Model parameters are pre-trained and loaded.
* -s: Trained parameters are saved periodically.


## Result Analysis 
Training records and results are saved in ['./records'](https://github.com/liuzey/EECS738_Project3/tree/main/records), which shows indeed a decrease in loss. This indicates the training is working in some sense. However, the accuracy doesn't show great improvements. This is a rough implementation of neural networks, which has much space for improvements, especially in gradient calculation. I remain these as my future work.

### MLP
The structure of the MLP model used in MNIST task is 

| MLP  |
| :-------------: |
| FC1(1\*28\*28, 100)  |
| FC2(100, 10)  |
All linear layers have biases.

### CNN
For this convolutional neural network in this project, the convolutional kernels is set at (3,3). The structure of the CNN model used in GTSRB task is 

| CNN  |
| :-------------: |
| Conv1(channels=32, stride=(1, 1), padding=(0, 0)), BatchNorm(32), ReLU |
| Conv2(channels=64, stride=(2, 2), padding=(1, 1)) |
| Conv3(channels=64, stride=(1, 1), padding=(0, 0)), BatchNorm(64), ReLU, Dropout |
| Conv4(channels=128, stride=(2, 2), padding=(1, 1)) |
| Conv5(channels=128, stride=(1, 1), padding=(0, 0)), BatchNorm(128), ReLU |
| FC1(128\*5\*5, 100), ReLU |
| FC2(100, 43) |

## Technical Details
### Activations
[Sigmoid](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/MLP.py#L94):

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/Sigmoid-function.JPG)

[ReLU](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/MLP.py#L49) returns the non-negative value of inputs.

### Dropout
[Dropout](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/MLP.py#L80) turns values into 0 at a probability of 50% in training phase, and multiplies value by 0.5 in testing phase.

### Batch Normalization
[BatchNorm](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/CNN.py#L136) follows the similar procedure as normalization over batch, but the final inputs is weighted by gamma and added bias beta. The gamma and beta are adjusted in the training phase. In the testing phase, the outputs are calculated with remembered global values.

### Convolutional Layer
[Forwarding](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/layers.py#L15):
Overview function:

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/05.gif)

Convolutionalk kernel operation:

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/06.gif)

[Backwarding](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/layers.py#L34):

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/07.gif)

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/08.gif)

### Fully-connected Layer
[Forwarding](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/layers.py#L57):

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/02.gif)

[Backwarding](https://github.com/liuzey/EECS738_Project3/blob/e65024acd5599e13549a8742eb563814eb168b1a/layers.py#L64):

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/01.gif)

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/03.gif)

![](https://github.com/liuzey/EECS738_Project3/blob/main/formula/04.gif)


## Notes
* Cross entropy loss is only used for printing in a perceptual manner in MLP, but does not participate in actual backpropagation. For CNN records, the loss is the sum of gradient in the last layer.
* Training the CNN for GTSRB can be highly time-consuming, which needs my further improvements.
* I've tried different MLP structures in MNIST task and different learning rates.
* I find it difficult to choose initial parameters, especially for linear layers in adopted rough propagation/gradient strategy. The large dimensions will often make the results big and shrink difference between features, losing expressive ability in sigmoid activations and causing troubles in exponential calculations in softmax. 
* In the final version, maxpooling and BatchNormaliztion1d are eliminated for simplification. For maxpooling, the backpropagation requires more work. For batch1d, it makes values deviated largely and result in minus values, which requires more work. Maxpooling should be implemented in a very similar way as Dropout function.

## Schedule
- [x] Set up a new git repository in your GitHub account
- [x] Pick two datasets from https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
- [x] Choose a programming language (Python, C/C++, Java) **Python**
- [x] Formulate ideas on how neural networks can be used to accomplish the task for the specific dataset
- [x] Build a neural network to model the prediction process programmatically
- [x] Document your process and results
- [x] Commit your source code, documentation and other supporting files to the git repository in GitHub

## Reference
* MNIST - Wikipedia. https://en.wikipedia.org/wiki/MNIST_database
* GTSRB Dataset - Institut f??r Neuroinformatik. https://benchmark.ini.rub.de/gtsrb_dataset.html
* Python-mnist - Github. https://github.com/sorki/python-mnist
* Simple-Convolutional-Neural-Network - Github. https://github.com/zhangjh915/Simple-Convolutional-Neural-Network
* Numpy.random - Numpy API. https://numpy.org/doc/stable/reference/random/
* A Gentle Introduction to Cross-Entropy for Machine Learning. https://machinelearningmastery.com/cross-entropy-for-machine-learning/
* Softmax Activation Function with Python. https://machinelearningmastery.com/softmax-activation-function-with-python/
* BatchNorm2D - Pytorch Docs. https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
* Torch.optim - Pytorch Docs. https://pytorch.org/docs/stable/optim.html
* batch-norm - Github. https://github.com/anuvindbhat/batch-norm
* Python PIL | Image.resize() method - GeeksforGeeks. https://www.geeksforgeeks.org/python-pil-image-resize-method/

