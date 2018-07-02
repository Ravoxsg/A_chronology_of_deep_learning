# A chronology of deep learning

Hey everyone who is reading this! 

So what is the hook with deep learning? Why is everyone talking about it? What happened? Well, in the last three decades, a lot of awesome ideas came out, leading to exceptional breakthroughs on general benchmark tasks to evaluate AI systems performance, like image classification, voice recognition, etc. To get the bigger picture, this repository tries to list in chronological order the main papers about deep learning. The number of citations is given according to Google Scholar stats. 

## Before the 1980s

* The concept of *perceptron* traces back to 1957-58:\
[The perceptron: a probabilistic model for information storage and organization in the brain](http://www2.fiit.stuba.sk/~cernans/nn/nn_texts/neuronove_siete_priesvitky_02_Q.pdf), Rosenblatt, 1958, Psychological Review, **7784 citations**

## 1980s

* *Recurrent connections* were invented at the beginning of the 1980s:\
[Neural networks and physical systems with emergent collective computational abilities](http://www.pnas.org/content/pnas/79/8/2554.full.pdf), Hopfield, 1982, NAS, **19632 citations (!)**
* Deep learning would not exist as it exists now if it was not for Geoffrey Hinton, David Rumelhart and Ronald Williams invention of the *Backpropagation* algorithm: \
[Learning representations by back-propagating erros](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf), Rumelhart et al., 1986, Nature, **14500 citations** (!)\
From this point, it was possible to train arbitrarily deep feed-forward neural networks with *error backpropagation*. 
* A few years after, training deep *recurrent* nets was detailed in this paper: \
[A learning algorithm for continually running fully recurrent neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9724&rep=rep1&type=pdf), Williams et al., 1989, Neural Computation, **2913 citations**
* A single hidden layer plain network has been shown to be a universal function approximator - justifying the interest in neural nets:\
[Approximation by superpositions of a sigmoidal function](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf), Cibenko, 1989, Mathematics of control, signal and system, **10168 citations (!)**

## 1990s

Despite promising breakthroughs in the late 1980s, in the 1990s AI entered a new *Winter era*, during the which there were few developments (especially compared to what happened in the 2010s). Deep learning approaches were discredited because of their average performance, mostly because of a lack of training data and computational power.

* Bengio's team was the first to exhibit how hard it can be to learn patterns *over a long time depth*:\
[Learning long-term dependencies with gradient is difficult](http://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf), Bengio et al., 1994, IEEE, **2418 citations**
* The *wake-sleep algorithm* inspired the autoencoder type of neural networks: \
[The wake-sleep algorithm for unsupervised neural networks](http://www.cs.toronto.edu/~fritz/absps/ws.pdf), Hinton et al., 1995, Science, **942 citations**
* Convolutional neural networks (CNNs) were developed in the early 1990s, mostly by Yann LeCun, and their broad application was described here: \
[Convolutional neural networks for images, speech and time-series](https://www.researchgate.net/profile/Yann_Lecun/publication/2453996_Convolutional_Networks_for_Images_Speech_and_Time-Series/links/0deec519dfa2325502000000.pdf), Yann LeCun & Yoshua Bengio, 1995, The Handbook of Brain Theory and Neural Networks, **1550 citations**
* LSTMs, still widely used today for sequence modeling, are actually quite an old invention: \
[Long short-term memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf), Hochreiter et al., 1997, Neural Computation, **9811 citations**
* Roughly around the same time as LSTMs came the idea of training RNNs in *both directions*, meaning that hidden states have access to input elements from the past *and the future*: \
[Bidirectional recurrent neural networks](https://www.researchgate.net/profile/Mike_Schuster/publication/3316656_Bidirectional_recurrent_neural_networks/links/56861d4008ae19758395f85c.pdf), Schuster et al., 1997, IEEE Transactions on Neural Processing, **1167 citations**
* At the end of the 1990s, Yoshua Bengio and Yann LeCun, regarded today as two of the godfathers in deep learning, generalized document recognition via neural networks trained by gradient desent, and introduced *Graph Transformer Networks* : \
[Gradient-based learning applied to document recognition](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf), LeCun et al., 1998, IEEE, **12546 citations (!)**

## 2000s

This *AI Winter* continued until roughly 2006, when research in deep learning started to flourish again. 

* We have known for a very long time that neural nets are efficient at language modeling:\
[A neural probabilistic language model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), Bengio et al., 2003, Journal of Machine Learning Research, **3504 citations**
* *Belief networks* were pretty popular in the 2000s:\
[A fast learning algorithm for deep belief nets](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf), Hinton et al., 2006, Neural Computation, **8362 citations**
* This *AI winter* probably ended with the groundbreaking (yet fairly simple) invention of *autoencoders*:\
[Reducing the dimensionality of data with neural networks](https://pdfs.semanticscholar.org/7d76/b71b700846901ac4ac119403aa737a285e36.pdf), Hinton et al., 2006, Science, **7066 citations**
* *Pre-training* can avoid training getting stuck because of a poor initialization:\
[Greedy layer-wise training of deep networks](http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf), Bengio et al., 2007, NIPS, **3022 citations**
* The end of the 2000's saw improvements in deep generative models:\
[Convolutional deep belief networks for scalable unsupervised learning and hierarchical representations](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf), Lee et al., 2009, ICML, **1832 citations**
* Applying neural nets in multiple stages:\
[What is the best multi-stage architecture for object recognition?](https://ieeexplore.ieee.org/abstract/document/5459469/?part=1), Jarrett et al., 2009, ICCV, **1349 citations**


## 2010s

### 2010-2011

* You can pile up autoencoders to improve feature extraction:\
[Stacked denoising autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), Vincent et al., 2010, Journal of Machine Learning Research, **2485 citations**
* Although an older idea (dating back to 2000), Nair and Hinton popularized the use of *ReLU* as an activation function in deep learning:\
[Rectified Linear Units Improve Restricted Boltzmann Machines](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf), Nair et al., 2010, ICML, **3379 citations**
* Glorot and Bengio showed how hard it can be to optimize deep networks, and introduced the *softsign* activation function:\
[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi), Glorot et al., 2010, Journal of Machine Learning Research, **3059 citations**
* Andrew Ng's team showed that focusing on simple approach is once again of key relevance:\
[An analysis of single-layer networks in unsupervised feature learning](http://proceedings.mlr.press/v15/coates11a/coates11a.pdf), Coates et al., 2011, Journal of Machine Learning Research, **1329 citations**

### 2012

* Deep learning became mainstream for real when Hinton's lab destroyed the previous state-of-the-art on *ImageNet* (an annual visual recognition challenge) by using deep Conv Nets in 2012:\
[Imagenet classification using deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), Krizhevsky et al., 2012, NIPS, **24405 citations (!!)**
* The *Adadelta* popular adaptive learning rate dates back from 2012:\
[Adadelta: an adaptive learning rate method](https://arxiv.org/pdf/1212.5701.pdf), Zeiler, 2012, arxiv.org, **1812 citations**
* *Dropout* first paper:\
[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf), Hinton et al., 2012, arxiv.org, **2773 citations**
* *Bayesian optimization* can enhance the performance of most machine learning algorithms:\
[Practical Bayesian optimization of machine learning algorithms](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf), Snoek et al., 2012, NIPS, **1371 citations**
* Perhaps surprisingly, *random search* is more efficient than grid search for hyper-parameter tuning:\
[Random search for hyper-parameter optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), Bergstra et al, 2012, Journal of Machine Learning Research, **1541 citations**

### 2013

* A team from Singapore introduced a new kind of CNNs, emphasizing interest in convolutions with 1 by 1 receptive fields:\
[Network-in-network](https://arxiv.org/pdf/1312.4400.pdf), Lin et al., arxiv.org, **1439 citations**
* In 2013 Kingma designed a new approach to autoencoders, in a Bayesian framework called the *Variational Autoencoder (VAE)*:\
[Auto-encoding variational Bayes](https://arxiv.org/pdf/1312.6114.pdf), Kingma et al, 2013, ICLR, **2223 citations**\
This paper would have considerable consequences in the deep learning community.
* Here, an integrated framework (that deals with recognition, localization and detection altogether !) using CNNs is presented:\
[Overfeat: integrated recognition, localization and detection using convolution networks](https://arxiv.org/pdf/1312.6229.pdf), Sermanet et al, 2013, arxiv.org, **2209 citations**\
Nowadays this framework is outdated and object detection is mostly done via algorithms like YOLO. Nevertheless, this paper was groundbreaking when it came out. 
* Unsupervised learning with deep nets:\
[Building high-level features using large scale unsupervised learning](https://arxiv.org/pdf/1112.6209.pdf&amp), Le et al., 2013, ICASSP, **1596 citations**
* Deep learning started to prove very efficient for speech recognition too:\
[Speech recognition with deep recurrrent neural networks](https://arxiv.org/pdf/1303.5778.pdf)?), Graves et al., 2013, Acoustics, speech and signal processing, **2296 citations**
* Scene labeling with deep CNNs:\
[Learning hierarchical features for scence labeling](https://hal-upec-upem.archives-ouvertes.fr/file/index/docid/742077/filename/farabet-pami-13.pdf), Farabet et al., 2013, IEEE Pattern Analysis and Machine Intelligence, **1451 citations**

### 2014

2014 was really a seminal year for deep learning, with major contributions from a broad variety of groups. 

* The *VGG-16 and VGG-19* architectures popularized the use of 3 by 3 convolution filters as well as going very deep in terms of number of layers:\
[Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556/), Simonyan et al., 2014, ICLR, **12292 citations (!)**
* *Gated Recurrent Units*, an improvement on LSTMs, were introduced in 2014:\
[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Cho et al., 2014, arxiv.org, **2647 citations**
* Their performance for sequence modelling is evaluated here:\
[Empirical evaluation of gated recurrent networks for sequence modelling](https://arxiv.org/pdf/1412.3555.pdf), Chung et al., 2014, arxiv.org, **1300 citations**
* *Adam*, probably the most used optimizer, was described by Durk Kingma (him again !) and Jimmy Ba that year:\
[Adam: a method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf), Kingma and Ba, 2014, ICLR, **9562  citations**
* The popular framework *Caffe* was released that year:\
[Caffe: convolutional architecture for fast feature embedding](https://arxiv.org/pdf/1408.5093.pdf), Jia et al., 2014, ACM, **7895 citations**
* *Neural Turing Machines* are neural nets extended with an auxiliary memory:\
[Neural Turing machines](https://arxiv.org/pdf/1410.5401.pdf), Graves et al., 2014, arxiv, **654 citations**
* Semi-supervised  learning revisited with VAEs:\
[Semi-supervised learning with deep generative models](http://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf), Kingma et al, 2014, NIPS, **533 citations**
* Rigorous benchmark of CNNs architecture:\
[Return of the devil in details: delving deep into convolutional nets](https://arxiv.org/pdf/1405.3531.pdf), 2014, Chatfield et al., 2014, ICCV (2015), **1636 citations**
* *Stochastic backpropagation* allows to train nets which weights are represented by a mean and a variance:\
[Stochastic backpropagation and approximate inference in deep generative models](https://arxiv.org/pdf/1401.4082.pdf), Rezende et al., 2014, ICML, **867 citations**
* *Sequence-to-sequence* learning with neural nets revolutionized machine translation in 2014:\
[Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), Sutskever et al., 2014, NIPS, **3747 citations**
* *Dropout* is now widely used for regularization. This is the official paper\
[Dropout: a simple way to prevent neural networks from overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer), Srivastava et al., 2014, Journal of Machine Learning Research, **8792 citations**\
Dropout was first described by Srivastava during his Master's thesis, and the paper was rejected to NIPS 2012. 
* Google described a more efficient way to generate image captions:\
[Show and tell: a neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2A_101.pdf), Vinyals et al., 2014, CVPR (2015), **1652 citations**
* Ian Goodfellow, while a PhD student in Bengio's group, discovered *Generative Adversarial Networks* - a completely new type of neural networks, that actually involves two networks:\
[Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al., 2014, NIPS, **3501 citations**\
Yann LeCun said that GANs *"were the most exciting idea in machine learning in the last ten years"*

### 2015

* ImageNet has been a catalyst of powerful ideas in deep learning. Few new architectures have been as widely adopted as *Residual connections*, introduced by Microsoft's Beijing team:\
[Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf), He et al., 2015, CVPR, **9309 citations**
* *Batch-normalization* acts as a training booster and regularizer, by normalizing each mini-batch:\
[Batch normalization: accelerating deep network training by reducing internal covariate shift](https://arxiv.org/pdf/1502.03167.pdf的paper适合想深入了解原理，这个视频很清楚的讲了bn起到的作用。), Ioffe et al., 2015, ICML, **5048 citations**
* Major improvement in object detection:\
[Fast R-CNN](http://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf), Girshick, 2015, ICCV, **2736 citations**
* Building on the previous paper, more efficiently:\
[Faster R-CNN: towards real-time object detection with region-proposal networks](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf), Ren et al., 2015, NIPS, **3651 citations**
* Merging sequence modelling with CNNs:\
[Convolutional LSTM](http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf), Shi et al., 2015, NIPS, **339 citations**
* Another approach of combining CNNs and LSTMs:\
[Long-term recurrent convolutional networks for visual recognition and description](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf), Donahue et al., 2015, CVPR, **1673 citations**
* A new way to go deeper with neural nets:\
[Highway networks](https://arxiv.org/pdf/1505.00387.pdf), Srivastava et al., 2015, ICML, **454 citations**
* The interest of highway networks is further developed here:\
[Training very deep networks](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf), Srivastava et al., 2015, NIPS, **405 citations**
* *Attention* was one of the most important improvements in deep learning in the last few years:\
[Show, attend and tell: neural image caption generation with visual attention](http://proceedings.mlr.press/v37/xuc15.pdf), Xu et al., 2015, ICML, **1778 citations**
* Efficient end-to-end image segmentation with low data:\
[U-Net: convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1611.09326.pdf), Ronneberger et al., 2015, MICCAI, **1993 citations**
* Unsupervised learning with CNNs became of new interest with GANs:\
[Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/pdf/1511.06434.pdfï¼‰), Radford et al., 2015, ICLR (2016), **1713 citations**
* Combining GANs with VAEs can give interesting results:\
[Adversarial autoencoders](https://arxiv.org/pdf/1511.05644.pdf), Makhzani et al., 2015, ICLR (2016), **333 citations**

### 2016

* In 2016 Google officially released *Tensorflow*:\
[Tensorflow: a system for large-scale machine learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf), Abadi et al., 2016, USENIX, **3671 citations**
* *YOLO* represents today the state-of-the-art in object detection, both in terms of accuracy and speed:\
[You only look once: unified, real-time object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf), Redmon et al., 2016, CVPR, **1612 citations**
* OpenAI brought a few major tricks to improve GAN training:\
[Improved techniques for training GANs](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf), Salimans et al., 2016, NIPS, **811 citations**
* Probabilistic generative models with attention:\
[Attend, infer, repeat](https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models.pdf), Ali Eslami et al., 2016, NIPS, **67 citations**
* Factorizing convolution filters allows better results with less parameters:\
[Rethinking the inception architecture for computer vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf), Szegedy et al., 2016, CVPR, **1595 citations**

### 2017

* One major problem with GANs is training unstability, where the decoder learns much faster than the generator, and training gets stuck. W-GANs introduce a new loss function in GANs, that helps tackling this problem:\
[Wasserstein GANs](https://arxiv.org/pdf/1701.07875.pdf), Arjovski et al., 2017, ICML, **720 citations**
* A team from Google introduced the *Transformer*, an architecture solely relying on attention, that yields state-of-the-art results on machine translation tasks:\
[Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), Vaswani et al., 2017, NIPS, **348 citations**
* One application of GANs is to improve image resolution, and in 2017 they were able to generate photo-like images (!):\
[Photo-Realistic single image super-resolution using a Generative Adversarial Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf), Ledig et al., 2017, CVPR, **598 citations**
* The impact of residual connections in image recognition was once again showed in this paper by Google:\
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf), Szegedy et al, 2017, AAAI, **864 citations**
* *Cycle-GANs* allow amazing image-to-image translation, like stylizing a horse into a zebra for instance:\
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf), Zhu et al., 2017, ICCV, **501 citations**
* In the Fall 2017, Hinton's Google Brain team published a completely new kind of neural net architecture, the *capsule network*, that Hinton is credited to have described as *finally something that seems to work*. Results on MNIST are very competitive.\
[Dynamic routing between capsules](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf), Sabour et al., 2017, NIPS, **135 citations**
* 2017 has seen a regain of interest for gradient estimators, that are (preferably unbiased and ideally low-variance) substitutions in cases where it's impossible to get the true gradient:\
[REBAR: low-variance, unbiased gradient estimates for discrete latent-variable models](http://papers.nips.cc/paper/6856-rebar-low-variance-unbiased-gradient-estimates-for-discrete-latent-variable-models.pdf), Tucker et al., 2017, NIPS, **28 citations**
