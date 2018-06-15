# A chronology of deep learning

Hey everyone who is reading this! 

So what is the hook with deep learning? Why is everyone talking about it? What happened? Well, in the last three decades, a lot of awesome ideas came out, leading to exceptional breakthroughs on general benchmark tasks to evaluate AI systems performance, like image classification, voice recognition, etc. To get the bigger picture, this repository tries to list in chronological order the main papers about deep learning. The number of citations is given according to Google Scholar stats. 

## Before the 1980s

* The concepton of *perceptron* traces back to 1957-58:\
[The perceptron: a probabilistic model for information storage and organization in the brain](http://www2.fiit.stuba.sk/~cernans/nn/nn_texts/neuronove_siete_priesvitky_02_Q.pdf), Rosenblatt, 1958, Psychological Review, **7784 citations**

## 1980s

* *Recurrent connections* were invented at the beginning of the 1980s:\
[Neural networks and physical systems with emergent collective computational abilities](http://www.pnas.org/content/pnas/79/8/2554.full.pdf), Hopfield, 1982, NAS, **19632 citations (!)**

* Deep learning would exist as it exists now if it was not for Geoffrey Hinton, David Rumelhart and Ronald Williams invention of the *Backpropagation* algorithm: \
[Learning representations by back-propagating erros](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf), Rumelhart et al., 1986, Nature, **14500 citations** (!)\
From this point, it was possible to train arbitrarily deep feed-forward neural networks. 

* A few years after, training deep recurrent nets was detailed in this paper: \
[A learning algorithm for continually running fully recurrent neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9724&rep=rep1&type=pdf), Williams et al., 1989, Neural Computation, **2913 citations**

## 1990s

Despite promising breakthroughs in the late 1980s, in the 1990s, AI entered a new *Winter era*, during the which there were few developments (especially compared to what happened in the 2010s). Deep learning approaches were discredited because of their average performance, mostly because of lack of training data and computational power.

* Bengio's team was the first to exhibit how hard it can be to learn patterns *over a long time depth*:\
[Learning long-term dependencies with gradient is difficult](http://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf), Bengio et al., 1994, IEEE, **2418 citations**

* The *wake-sleep algorithm* inspired the autoencoder type of neural networks: \
[The wake-sleep algorithm for unsupervised neural networks](http://www.cs.toronto.edu/~fritz/absps/ws.pdf), Hinton et al., 1995, Science, **942 citations**

* Convolutional neural networks (CNNs) were developed in the early 1990s, mostly by Yann LeCun, and their broad application was described here: \
[Convolutional neural networks for images, speech and time-series](https://www.researchgate.net/profile/Yann_Lecun/publication/2453996_Convolutional_Networks_for_Images_Speech_and_Time-Series/links/0deec519dfa2325502000000.pdf), Yann LeCun & Yoshua Bengio, 1995, The Handbook of Brain Theory and Neural Networks, **1550 citations**

* LSTMs, still widely used today for sequence modeling, are actually quite an old invention: \
[Long short-term memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf), Hochreiter et al., 1997, Neural Computation, **9811 citations**

* Roughly around the same time as LSTMs came the idea of training RNNs in *both directions*, meaning that hiddent states have access to input elements from the future: \
[Bidirectional recurrent neural networks](https://www.researchgate.net/profile/Mike_Schuster/publication/3316656_Bidirectional_recurrent_neural_networks/links/56861d4008ae19758395f85c.pdf), Schuster et al., 1997, IEEE Transactions on Neural Processing, **1167 citations**

* At the end of the 1990s, Yoshua Bengio and Yann LeCun, regarded today as two of the godfathers in deep learning, generalized document recognition via neural networks trained by gradient desent, and introduced Graph Transformer Networks : \
[Gradient-based learning applied to document recognition](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf), LeCun et al., 1998, IEEE, **12546 citations (!)**

## 2000s

This *AI Winter* continued until roughly 2006, when research in deep learning started to flourish again. 

* This *AI winter* probably ended with the groundbreaking (yet fairly simple) invention of *autoencoders*:\
[Reducing the dimensionality of data with neural networks](https://pdfs.semanticscholar.org/7d76/b71b700846901ac4ac119403aa737a285e36.pdf), Hinton et al., 2006, Science, **7066 citations**

## 2010s

### 2010-2011

### 2012

* Deep learning became mainstream for real when Hinton's lab destroyed the previous state-of-the-art on ImageNet by using deep Conv Nets in 2012:\
[Imagenet classification using deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), Krizhevsky et al., 2012, NIPS, **24405 citations (!!)**

### 2013

* A team from Singapore introduced a new kind of CNNs, emphasizing interest in convolutions with 1 by 1 receptive fields:\
[Network-in-network](https://arxiv.org/pdf/1312.4400.pdf), Lin et al., ?, **1439 citations**

* In 2013 Kingma designed a new approach to autoencoders, in a Bayesian framework called the *Variational Autoencoder (VAE)*:\
[Auto-encoding variational Bayes](https://arxiv.org/pdf/1312.6114.pdf), Kingma et al, 2013, ICLR, **2223 citations**\
This paper would have considerable consequences in the deep learning community.

### 2014

2014 was really a seminal year for deep learning, with major contributions from a broad variety of groups. 

* *Gated Recurrent Units* were introduced in 2014:\
[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Cho et al., 2014, ?, **2647 citations**

* Ian Goodfellow, while a PhD student in Bengio's group, discovered *Generative Adversarial Networks* - a completely new type of neural networks, that actually involves two networks:\
[Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), Goodfellow et al., 2014, NIPS, **3501 citations**\
Yann LeCun said that GANs *"were the most exciting idea in machine learning in the last ten years"*

### 2015

* ImageNet has been a catalyst of powerful ideas in deep learning. Few new architectures have been as widely adopted as *Residual connections*, introduced by Microsoft's Beijing team:\
[Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf), He et al., 2015, CVPR, **9309 citations**

### 2016

* *YOLO* represents today the state-of-the-art in object detection, both in terms of accuracy and speed:\
[You only look once: unified, real-time object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf), Redmon et al., 2016, CVPR, **1612 citations**

* OpenAI brought a few major tricks to improve GAN training:\
[Improved techniques for training GANs](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf), Salimans et al., 2016, **811 citations**

### 2017

* One application of GANs is to improve image resolution, and in 2017 they were able to generate photo-like images:\
[Photo-Realistic single image super-resolution using a Generative Adversarial Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf), Ledig et al., 2017, CVPR, **598 citations**

* The impact of residual connections in image recognition was once again showed in this paper by Google:\
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf), Szegedy et al, 2017, AAAI, **864 citations**

* *Cycle-GANs* allow amazing image-to-image translation, like stylizing a horse into a zebra for instance:\
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf), Zhu et al., 2017, ICCV, **501 citations**
