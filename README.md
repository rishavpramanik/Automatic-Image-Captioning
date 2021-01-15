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
## Reference 
* [Show and Tell: Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

