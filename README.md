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
* [Modified National Institute of Standards and Technology database (MNIST)](http://yann.lecun.com/exdb/mnist/). This is possibly one of the most commonly-used and simplest datasetin image tasks in machine learning, which combines with different images of hand-written digits. The sizes of training dataset and testing dataset are 60,000 and 10,000 each. Though the attached webpage link seems not to be working, the dataset can be directly accessed by package [python-mnist](https://github.com/sorki/python-mnist). Images and grey-scale with a size of 28\*28 each. Ten labels in total ranges from 0, 1, 2... to 9. 
* [The German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html). This is a more chanllenging dataset with images of traffic signs captured in real street views. Each image contains exactly one traffic sign, usually located right at the middle. There are 43 labels in total, representing different types of traffic signs, and more than 50,000 samples. Recognition of these signs is much more difficult and practically significant, as many of the signs have similar shapes and colors. 


## Ideas and Thinking
* **DNN**: In my opinion, deep neural networks are basically a complex but repetitive variant of weighted function with non-linear operators added. What deep learning does is to tweak the weights gradually by iterating over a large (usually) number of examples. If taking a task as a process of building sand castles, the precise solution will be like building by crafting structures and shapes, and the deep learning will be like flapping and rubbing by frequently looking at samples. Outcome of deep learning are set of parameters to mimic and map certain features, known as saliency, to high confidence scores in exact dimension. This 'learning' process imitates the process of human learning (recognition or understanding), setting up correlations between 'name/meaning' and 'features' thourgh repetitive getting exposed to examples and strengthening neuron connections.  
* **Activation Functions**: An important component of deep learning is activation function, which significantly accounts for deep learning's success. The non-linear properties of deep learning are largely contributions of activations. What activations really does is to differentiate neurons to respond to unique features. After activation, some of the neurons will be silent for specific features, while others will not. In this way, the deep neural networks are no longer an inflexible combination of weighted values, but a instead dynamic system where each unit has a 'switch'. This is done mainly by the cut-off (e.g. ReLU) or downgrade (e.g. Sigmoid) in gradients in backpropagation. Once a neuron get silent for certain feature distribution, little changes will be witnessed in its parameters when updating. In forwarding cases, different parts of neurons will 'switch on' for different feature distribution, which greatly increase the fitting ability of the function in limited parameter space.
* **Backpropagation**: Backpropagation is a rather simple but important process. By calculating the partial gradient of the neuron output over model parameters, the paraemters are updated from layer to layer to increase highly-activated neurons' contribution to current label, and decrease noises from unrelated neurons.
* **Testing**: After iterations over training samples, the model parameters are tuned to distribution of the task domain, which expects the ability to guide new examples from the same domain into explainable outputs. Different from training, some operators tend to stay static in testing phase to expect stable behaviors in different situation, e.g. dropout and batch normalization.

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


## Result Analysis 
Training records adn results are saved in ['./records'](https://github.com/liuzey/EECS738_Project3/tree/main/records), which shows indeed a decrease in loss. This indicates the training is working in some sense. However, the accuracy doesn't show great improvements.

## Notes
* I find it difficult to choose initial parameters, especially for linear layers in adopted rough propagation/gradient strategy. The large dimensions will often make the results big and shrink difference between features, losing expressive ability in sigmoid activations and causing troubles in exponential calculations in softmax. 
* In the final version, maxpooling and BatchNormaliztion1d are eliminated for simplification. For maxpooling, the backpropagation requires more work. For batch1d, it makes values deviated largely and result in minus values, which requires more 

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
* GTSRB Dataset - Institut für Neuroinformatik. https://benchmark.ini.rub.de/gtsrb_dataset.html
* Python-mnist - Github. https://github.com/sorki/python-mnist

