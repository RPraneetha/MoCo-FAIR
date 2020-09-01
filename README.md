# Reproduction of Momentum Contrast For Unsupervised Visual Representation Learning

Debadeep Basu - 5089107
Ramya Praneetha Ghantasala - 5014212

## Introduction

Summary - we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning.

Dictionaries

Dictionaries in supervised and unsupervised learning

Unsupervised representation learning is gaining traction in the fields where the input can be broken down into tokenized items, like words in NLP. When there are tokenized representations available, unsupervised learning can be applied to achieve good results. Tokenized dictionaries like word2vec provide efficient search spaces

 On the other hand, in the field of Computer Vision, supervised learning is still dominantly used. A possible reason for this could be that the input stream for a computer vision task cannot be broken down into tokenized items as it is a continuous, high-dimensional input. 

 Recent studies have shown that a dynamic dictionary can be built for unsupervised representation learning using contrastive loss. The keys in the dynamic dictionaries are represented by an encoder network and are sampled from data like an image or a patch. The principle of contrastive loss is that when performing a look-up in the dictionary, the matching key should be similar to the encoded query and dissimilar to others. The more similar the key is to the query, the lesser is the contrastive loss. 


Therefore, it is desirable to build dynamic dictionaries that are large and consistent as they improve during training. The dictionary needs to be large enough to hold the high-dimensional space visual input and should be consistent in the sense that the keys in the dictionary are represented by same or similar encoders to quantitatively compare each of the keys to the query, and such that these comparisons are consistent for all keys.

The paper proposes Momentum Contrast or MoCo as a way to build these kind of dictionaries for unsupervised learning with a contrastive loss. The unique perspective the paper offers is to build the dictionary as a queue of data samples, where the encoded representations of the current mini-batch are enqueued and the oldest are dequeued.

Pretext tasks - Point of MoCo is a query matches a key if they are emcoded views of the same image.

In the task of unsupervised visual learning, the metwork is pre-trained on representations that can be transferred to downstream tasks by fine-tuning. The performance of the MoCo network in comparison to current SoTA methods is elaborated in the results section.


## Datasets Used

Datasets used(attempted):

ImageNet-1M,
Cifar-10
MNIST
EMNIST (900k images)    
ImageNet-mini

## Architecture

## Methodology
Contrastive loss

## Challenges Faced
Primary challenge: environment with multiple GPU
We tried working on a single GPU using Colab and Kaggle. But as defined in the paper(see Shuffling BN)
batch norm leads to information leakage between batches(which we have seen in a result... Loss not changing after 5 epochs)

Secondary challenge: Adapting the Shuffling code to adapt to TPU with 8 cores

Lower support of Xla by torch compared to distributed GPU backend - torch distributed does not contain any support for "tpu" or "Xla" backend yet, and is also not part of their roadmap at this point. This leads to a number of integral functions such as "torch.distributed.broadcast" to be unavailable to be adapted for the tpu backend - Currently torch.distributed has support for, and recommends only "gloo"(for CPU), and "nccl"(GPU) backends for distributed processing of tensors

Experiments: due to lower processing power, we made some changes to the suggested code from the paper. For instance, the transformation method using ImageNet for Resnet50, expects the image crops to be random crops of 224x224.   We had to use 32x32 to prevent our resources from running out of memory or terminating due to overuse
We changed to a Resnet18 for EMNIST

## Results

## Conclusion

The unique MoCo method proposed in this paper performs comparitively with supervised representation learning and puts forward unsupervised representation learning as a strong contender to traditional supervised learning methods in the field of Computer Vision.
