A Pytorch implementation of the CNN+RNN architecture on the MS-COCO dataset

## Background 
This project is about combining CNN and RNN networks to build a deep learning model that produces captions given an input image.
Image captioning requires a complex deep learning model with two components: 
1) a CNN that transforms an input image into a set of features, and 
2) an RNN that turns those features into rich, descriptive language. 

## Approach
recurrent neural networks learn from ordered sequences of data.
* use pre-trained (VGG-19) model for object detection and classification
* combine pre-trained CNNs and RNNs to build a complex image captioning
* Implement an LSTM for sequential text (image caption) generation.
* Train a model to predict captions and understand a visual scene.

### DataLoader
*get_loader function in data_loader.py

### CNN encoder 
![Encoder-CNN](encoder.png)
* Transform to Tensor: pre-process the test images
### RNN decoder
![Decoder-LSTM](decoder.png)
## Training: Hyperparameter Tunning 
### Model Parameter 
* vocab_threshold - the minimum word count threshold. Note that a larger threshold will result in a smaller vocabulary, whereas a smaller threshold will include rarer words and result in a larger vocabulary.
* vocab_from_file - a Boolean that decides whether to load the vocabulary from file.
* embed_size - the dimensionality of the image and word embeddings.
* hidden_size - the number of features in the hidden state of the RNN decoder.


### Training Parameter
* num_epochs - the number of epochs to train the model
* learn_rate 
* batch_size - the batch size of each training batch. It is the number of image-caption pairs used to amend the model weights in each training step.

## Reference 
* [Show and Tell: Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
* [Show and Tell : Lessons Learned from the 2015] (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7505636)
* [Show and Tell with Attention](https://arxiv.org/pdf/1502.03044.pdf)

