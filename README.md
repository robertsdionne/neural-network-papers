# neural-network-papers

## Table of Contents
1. [Surveys](#surveys)
2. [Datasets](#datasets)
3. [Programming Frameworks](#programming-frameworks)
4. [Learning to Compute](#learning-to-compute)
5. [Natural Language Processing](#natural-language-processing)
6. [Convolutional Neural Networks](#convolutional-neural-networks)
7. [Recurrent Neural Networks](#recurrent-neural-networks)
8. [Autoencoders](#autoencoders)
9. [Restricted Boltzmann Machines](#restricted-boltzmann-machines)
10. [Biologically Plausible Learning](#biologically-plausible-learning)
11. [Unsupervised Learning](#unsupervised-learning)
12. [Reinforcement Learning](#reinforcement-learning)
13. [Theory](#theory)
14. [Quantum Computing](#quantum-computing)
15. [Training Innovations](#training-innovations)
16. [Numerical Optimization](#numerical-optimization)
17. [Numerical Precision](#numerical-precision)
18. [Cognitive Architectures](#cognitive-architectures)
19. [Motion Planning](#motion-planning)
20. [Computational Creativity](#computational-creativity)
21. [Cryptography](#cryptography)
22. [Distributed Computing](#distributed-computing)
23. [Clustering](#clustering)

## Surveys
* [Deep Learning](http://rdcu.be/cW4c "Yann LeCunn, Yoshua Bengio, Geoffrey Hinton")
* [Deep Learning in Neural Networks: An Overview](http://arxiv.org/abs/1404.7828 "Juergen Schmidhuber")

## Datasets
* [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/abs/1502.05698 "Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush")
* [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909 "Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau")

## Programming Frameworks
* [Caffe: Convolutional Architecture for Fast Feature Embedding](http://arxiv.org/abs/1408.5093 "Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, Trevor Darrell")
  * [Improving Caffe: Some Refactoring](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-improving.pdf "Yangqing Jia")
* [Theano: A CPU and GPU Math Compiler in Python](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf "James Bergstra, Olivier Breuleux, Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, Guillaume Desjardins, Joseph Turian, David Warde-Farley, Yoshua Bengio")
  * [Theano: new features and speed improvements](http://arxiv.org/abs/1211.5590 "Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, James Bergstra, Ian Goodfellow, Arnaud Bergeron, Nicolas Bouchard, David Warde-Farley, Yoshua Bengio")
  * [Blocks and Fuel: Frameworks for deep learning](http://arxiv.org/abs/1506.00619 "Bart van Merriënboer, Dzmitry Bahdanau, Vincent Dumoulin, Dmitriy Serdyuk, David Warde-Farley, Jan Chorowski, Yoshua Bengio")
  * [Announcing Computation Graph Toolkit](http://joschu.github.io/index.html#Announcing CGT "John Schulman")
* [Torch7: A Matlab-like Environment for Machine Learning](http://ronan.collobert.com/pub/matos/2011_torch7_nipsw.pdf "Ronan Collobert, Koray Kavukcuoglu, Clément Farabet")
* [cuDNN: Efficient Primitives for Deep Learning](http://arxiv.org/abs/1410.0759 "Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, Evan Shelhamer")
* [Fast Convolutional Nets With fbfft: A GPU Performance Evaluation](http://arxiv.org/abs/1412.7580 "Nicolas Vasilache, Jeff Johnson, Michael Mathieu, Soumith Chintala, Serkan Piantino, Yann LeCun")

## Learning to Compute
* [Neural Turing Machines](http://arxiv.org/abs/1410.5401 "Alex Graves, Greg Wayne, Ivo Danihelka")
  * [Reinforcement Learning Neural Turing Machines](http://arxiv.org/abs/1505.00521 "Wojciech Zaremba, Ilya Sutskever")
* [Memory Networks](http://arxiv.org/abs/1410.3916 "Jason Weston, Sumit Chopra, Antoine Bordes")
  * [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895 "Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus")
* [Learning to Transduce with Unbounded Memory](http://arxiv.org/abs/1506.02516 "Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, Phil Blunsom")
* [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/abs/1503.01007 "Armand Joulin, Tomas Mikolov")
* [Pointer Networks](http://arxiv.org/abs/1506.03134 "Oriol Vinyals, Meire Fortunato, Navdeep Jaitly")
* [Learning to Execute](http://arxiv.org/abs/1410.4615 "Wojciech Zaremba, Ilya Sutskever")
* [Grammar as a Foreign Language](http://arxiv.org/abs/1412.7449 "Oriol Vinyals, Lukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, Geoffrey Hinton")

## Natural Language Processing
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/ "Christopher Olah")

### Word Vectors
* [Natural Language Processing (almost) from Scratch](http://arxiv.org/abs/1103.0398 "Ronan Collobert, Jason Weston, Leon Bottou, Michael Karlen, Koray Kavukcuoglu, Pavel Kuksa")
* [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781 "Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean")
* [Learning to Understand Phrases by Embedding the Dictionary](http://arxiv.org/abs/1504.00548 "Felix Hill, Kyunghyun Cho, Anna Korhonen, Yoshua Bengio")
* [Inverted indexing for cross-lingual NLP](http://cst.dk/anders/inverted.pdf "Anders Søgaard, Željko Agić, Héctor Martínez Alonso, Barbara Plank, Bernd Bohnet, Anders Johannsen")

### Sentence and Paragraph Vectors
* [Distributed Representations of Sentences and Documents](http://arxiv.org/abs/1405.4053 "Quoc V. Le, Tomas Mikolov")
* [A Fixed-Size Encoding Method for Variable-Length Sequences with its Application to Neural Network Language Models](http://arxiv.org/abs/1505.01504 "Shiliang Zhang, Hui Jiang, Mingbin Xu, Junfeng Hou, Lirong Dai")
* [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726 "Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler")

### Character Vectors
* [Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation](http://arxiv.org/abs/1508.02096 "Wang Ling, Tiago Luís, Luís Marujo, Ramón Fernandez Astudillo, Silvio Amir, Chris Dyer, Alan W. Black, Isabel Trancoso")
* [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615 "Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush")
* [Modeling Order in Neural Word Embeddings at Scale](http://arxiv.org/abs/1506.02338 "Andrew Trask, David Gilmore, Matthew Russell")
* [Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs](http://arxiv.org/abs/1508.00657 "Miguel Ballesteros, Chris Dyer, Noah A. Smith")

### Sequence-to-Sequence Learning
* [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215 "Ilya Sutskever, Oriol Vinyals, Quoc V. Le")
* [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/abs/1506.07285 "Ankit Kumar, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Mohit Iyyer, Ishaan Gulrajani, Richard Socher")
* [Neural Transformation Machine: A New Architecture for Sequence-to-Sequence Learning](http://arxiv.org/abs/1506.06442 "Fandong Meng, Zhengdong Lu, Zhaopeng Tu, Hang Li, Qun Liu")

### Language Understanding
* [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340 "Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom")
* [Investigation of Recurrent-Neural-Network Architectures and Learning Methods for Spoken Language Understanding](http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf "Grégoire Mesnil, Xiaodong He, Li Deng, Yoshua Bengio")
* [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://arxiv.org/abs/1506.08941 "Karthik Narasimhan, Tejas Kulkarni, Regina Barzilay")

### Question Answering, and Conversing
* [Deep Learning for Answer Sentence Selection](http://arxiv.org/abs/1412.1632 "Lei Yu, Karl Moritz Hermann, Phil Blunsom, Stephen Pulman")
* [Neural Responding Machine for Short-Text Conversation](http://arxiv.org/abs/1503.02364 "Lifeng Shang, Zhengdong Lu, Hang Li")
* [A Neural Conversational Model](http://arxiv.org/abs/1506.05869 "Oriol Vinyals, Quoc Le")

### Convolutional
* [A Convolutional Neural Network for Modelling Sentences](http://arxiv.org/abs/1404.2188 "Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom")
* [Text Understanding from Scratch](http://arxiv.org/abs/1502.01710 "Xiang Zhang, Yann LeCun")
* [DeepWriterID: An End-to-end Online Text-independent Writer Identification System](http://arxiv.org/abs/1508.04945 "Weixin Yang, Lianwen Jin, Manfei Liu")

### Recurrent
* [Long Short-Term Memory Over Tree Structures](http://arxiv.org/abs/1503.04881 "Xiaodan Zhu, Parinaz Sobhani, Hongyu Guo")
* [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075 "Kai Sheng Tai, Richard Socher, Christopher D. Manning")

## Convolutional Neural Networks
* [Spatial Transformer Networks](http://arxiv.org/abs/1506.02025 "Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu")
* [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806 "Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller")
* [Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556 "Karen Simonyan, Andrew Zisserman")
* [Network In Network](http://arxiv.org/abs/1312.4400 "Min Lin, Qiang Chen, Shuicheng Yan")
* [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842 "Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich")
* [Learning to Generate Chairs with Convolutional Neural Networks](http://arxiv.org/abs/1411.5928 "Alexey Dosovitskiy, Jost Tobias Springenberg, Thomas Brox")
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://arxiv.org/abs/1411.4389 "Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell")
* [A Machine Learning Approach for Filtering Monte Carlo Noise](http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2015_LBF "Nima Khademi Kalantari, Steve Bako, Pradeep Sen")
* [Image Super-Resolution Using Deep Convolutional Networks](http://arxiv.org/abs/1501.00092 "Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang")
* [Learning to Deblur](http://arxiv.org/abs/1406.7444 "Christian J. Schuler, Michael Hirsch, Stefan Harmeling, Bernhard Schölkopf")
* [Monocular Object Instance Segmentation and Depth Ordering with CNNs](http://arxiv.org/abs/1505.03159 "Ziyu Zhang, Alexander G. Schwing, Sanja Fidler, Raquel Urtasun")
* [FlowNet: Learning Optical Flow with Convolutional Networks](http://arxiv.org/abs/1504.06852 "Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox")
* [DeepStereo: Learning to Predict New Views from the World's Imagery](http://arxiv.org/abs/1506.06825 "John Flynn, Ivan Neulander, James Philbin, Noah Snavely")
* [Deep convolutional filter banks for texture recognition and segmentation](http://arxiv.org/abs/1411.6836 "Mircea Cimpoi, Subhransu Maji, Andrea Vedaldi")
* [Deep Karaoke: Extracting Vocals from Musical Mixtures Using a Convolutional Deep Neural Network](http://arxiv.org/abs/1504.04658 "Andrew J.R. Simpson, Gerard Roma, Mark D. Plumbley")
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852 "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun")
* [Rotation-invariant convolutional neural networks for galaxy morphology prediction](http://arxiv.org/abs/1503.07077 "Sander Dieleman, Kyle W. Willett, Joni Dambre")
* [Deep Fried Convnets](http://arxiv.org/abs/1412.7149 "Zichao Yang, Marcin Moczulski, Misha Denil, Nando de Freitas, Alex Smola, Le Song, Ziyu Wang")
* [Fractional Max-Pooling](http://arxiv.org/abs/1412.6071 "Benjamin Graham")
* [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html "Sander Dieleman")
* [Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/ "Christopher Olah")

## Recurrent Neural Networks
* [Training recurrent networks online without backtracking](http://arxiv.org/abs/1507.07680 "Yann Ollivier, Guillaume Charpiat")
* [Modeling sequential data using higher-order relational features and predictive training](http://arxiv.org/abs/1402.2333 "Vincent Michalski, Roland Memisevic, Kishore Konda")
* [Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329 "Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals")
* [Grid Long Short-Term Memory](http://arxiv.org/abs/1507.01526 "Nal Kalchbrenner, Ivo Danihelka, Alex Graves")
* [Long Short-Term Memory](http://people.idsia.ch/~juergen/rnn.html "Sepp Hochreiter Jürgen Schmidhuber") (ftp://ftp.idsia.ch/pub/juergen/lstm.pdf)
  * [LSTM: A Search Space Odyssey](http://arxiv.org/abs/1503.04069 "Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, Jürgen Schmidhuber")
* [Learning Longer Memory in Recurrent Neural Networks](http://arxiv.org/abs/1412.7753 "Tomas Mikolov, Armand Joulin, Sumit Chopra, Michael Mathieu, Marc'Aurelio Ranzato")
* [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](http://arxiv.org/abs/1504.00941 "Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton")
* [A Clockwork RNN](http://arxiv.org/abs/1402.3511 "Jan Koutník, Klaus Greff, Faustino Gomez, Jürgen Schmidhuber")
* [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/abs/1502.04623 "Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra")
* [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367 "Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio")
* [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473 "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio")
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/ "Christopher Olah")
* [A Recurrent Latent Variable Model for Sequential Data](http://arxiv.org/abs/1506.02216 "Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, Yoshua Bengio")
* [ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](http://arxiv.org/abs/1505.00393 "Francesco Visin, Kyle Kastner, Kyunghyun Cho, Matteo Matteucci, Aaron Courville, Yoshua Bengio")
* [Translating Videos to Natural Language Using Deep Recurrent Neural Networks](http://arxiv.org/abs/1412.4729 "Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko")
* [Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/abs/1502.04681 "Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov")

## Autoencoders
* [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114 "Diederik P Kingma, Max Welling")
* [Analyzing noise in autoencoders and deep networks](http://arxiv.org/abs/1406.1831 "Ben Poole, Jascha Sohl-Dickstein, Surya Ganguli")
* [k-Sparse Autoencoders](http://arxiv.org/abs/1312.5663 "Alireza Makhzani, Brendan Frey")
* [Generalized Denoising Auto-Encoders as Generative Models](http://arxiv.org/abs/1305.6663 "Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent")
* [Marginalized Denoising Auto-encoders for Nonlinear Representations](http://www.cse.wustl.edu/~mchen/papers/deepmsda.pdf "Minmin Chen, Kilian Weinberger, Fei Sha, Yoshua Bengio")
  * [Marginalized Denoising Autoencoders for Domain Adaptation](http://arxiv.org/abs/1206.4683 "Minmin Chen, Zhixiang Xu, Kilian Weinberger, Fei Sha")
* [Real-time Hebbian Learning from Autoencoder Features for Control Tasks](http://mitpress.mit.edu/sites/default/files/titles/content/alife14/ch034.html "Justin K. Pugh, Andrea Soltoggio, Kenneth O. Stanley")
* [Is Joint Training Better for Deep Auto-Encoders?](http://arxiv.org/abs/1405.1380 "Yingbo Zhou, Devansh Arpit, Ifeoma Nwogu, Venu Govindaraju")
* [Towards universal neural nets: Gibbs machines and ACE](http://arxiv.org/abs/1508.06585 "Galin Georgiev")
* [Transforming Auto-encoders](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf "G. E. Hinton, A. Krizhevsky, S. D. Wang")

## Restricted Boltzmann Machines
* [The wake-sleep algorithm for unsupervised neural networks](https://www.cs.toronto.edu/~hinton/absps/ws.pdf "Geoffrey E Hinton, Peter Dayan, Brendan J Frey, Radford M Neals")
  * [A simple algorithm that discovers efficient perceptual codes](https://www.cs.toronto.edu/~hinton/absps/percepts.pdf "Brendan J. Frey, Peter Dayan, Geoffrey E. Hinton")
  * [Reweighted Wake-Sleep](http://arxiv.org/abs/1406.2751 "Jörg Bornschein, Yoshua Bengio")
* [An Infinite Restricted Boltzmann Machine](http://arxiv.org/abs/1502.02476 "Marc-Alexandre Côté, Hugo Larochelle")
* [Quantum Deep Learning](http://arxiv.org/abs/1412.3489 "Nathan Wiebe, Ashish Kapoor, Krysta M. Svore")
* [Quantum Inspired Training for Boltzmann Machines](http://arxiv.org/abs/1507.02642 "Nathan Wiebe, Ashish Kapoor, Christopher Granade, Krysta M Svore")

## Biologically Plausible Learning
* [How Auto-Encoders Could Provide Credit Assignment in Deep Networks via Target Propagation](http://arxiv.org/abs/1407.7906 "Yoshua Bengio")
* [Towards Biologically Plausible Deep Learning](http://arxiv.org/abs/1502.04156 "Yoshua Bengio, Dong-Hyun Lee, Jorg Bornschein, Zhouhan Lin")
* [Random feedback weights support learning in deep neural networks](http://arxiv.org/abs/1411.0247 "Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, Colin J. Akerman")

## Unsupervised Learning
* [Index-learning of unsupervised low dimensional embedding](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/indexlearning.pdf "Ben Graham")
* [An Analysis of Unsupervised Pre-training in Light of Recent Advances](http://arxiv.org/abs/1412.6597 "Tom Le Paine, Pooya Khorrami, Wei Han, Thomas S. Huang")
* [Is Joint Training Better for Deep Auto-Encoders?](http://arxiv.org/abs/1405.1380 "Yingbo Zhou, Devansh Arpit, Ifeoma Nwogu, Venu Govindaraju")
* [Rectified Factor Networks](http://arxiv.org/abs/1502.06464 "Djork-Arné Clevert, Andreas Mayr, Thomas Unterthiner, Sepp Hochreiter")

## Reinforcement Learning
* [Human-level control through deep reinforcement learning](http://rdcu.be/cdlg "Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei Rusu, Joel Veness, Marc Bellemare, Alex Graves, Martin Riedmiller, Andreas Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis")
* [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602 "Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller")
* [Universal Value Function Approximators](http://jmlr.org/proceedings/papers/v37/schaul15.html "Tom Schaul, Daniel Horgan, Karol Gregor, David Silver")

## Theory
* [On the saddle point problem for non-convex optimization](http://arxiv.org/abs/1405.4604 "Razvan Pascanu, Yann N. Dauphin, Surya Ganguli, Yoshua Bengio")
* [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](http://arxiv.org/abs/1406.2572 "Yann Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio")
* [The Loss Surfaces of Multilayer Networks](http://arxiv.org/abs/1412.0233 "Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, Yann LeCun")
* [On the Number of Linear Regions of Deep Neural Networks](http://arxiv.org/abs/1402.1869 "Guido Montúfar, Razvan Pascanu, Kyunghyun Cho, Yoshua Bengio")
* [An exact mapping between the Variational Renormalization Group and Deep Learning](http://arxiv.org/abs/1410.3831 "Pankaj Mehta, David J. Schwab")
* [Why does Deep Learning work? - A perspective from Group Theory](http://arxiv.org/abs/1412.6621 "Arnab Paul, Suresh Venkatasubramanian")
* [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120 "Andrew M. Saxe, James L. McClelland, Surya Ganguli")
* [On the Stability of Deep Networks](http://arxiv.org/abs/1412.5896 "Raja Giryes, Guillermo Sapiro, Alex M. Bronstein")
* [Over-Sampling in a Deep Neural Network](http://arxiv.org/abs/1502.03648 "Andrew J.R. Simpson")
* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/ "Christopher Olah")
* [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/ "Christopher Olah")
* [Groups & Group Convolutions](http://colah.github.io/posts/2014-12-Groups-Convolution/ "Christopher Olah")
* [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ "Christopher Olah")
* [Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/ "Christopher Olah")
* [Causal Entropic Forces](http://www.alexwg.org/publications/PhysRevLett_110-168702.pdf "A. D. Wissner-Gross, C. E. Freer")
* [Physics, Topology, Logic and Computation: A Rosetta Stone](http://arxiv.org/abs/0903.0340 "John C. Baez, Mike Stay")

## Quantum Computing
* [Analyzing Big Data with Dynamic Quantum Clustering](http://arxiv.org/abs/1310.2700 "M. Weinstein, F. Meirer, A. Hume, Ph. Sciau, G. Shaked, R. Hofstetter, E. Persi, A. Mehta, D. Horn")
* [Quantum algorithms for supervised and unsupervised machine learning](http://arxiv.org/abs/1307.0411 "Seth Lloyd, Masoud Mohseni, Patrick Rebentrost")
* [Entanglement-Based Machine Learning on a Quantum Computer](http://arxiv.org/abs/1409.7770 "X.-D. Cai, D. Wu, Z.-E. Su, M.-C. Chen, X.-L. Wang, L. Li, N.-L. Liu, C.-Y. Lu, J.-W. Pan")
* [A quantum speedup in machine learning: Finding a N-bit Boolean function for a classification](http://arxiv.org/abs/1303.6055 "Seokwon Yoo, Jeongho Bang, Changhyoup Lee, Jinhyoung Lee")

## Training Innovations
* [The Effects of Hyperparameters on SGD Training of Neural Networks](http://arxiv.org/abs/1508.02788 "Thomas M. Breuel")
* [Gradient-based Hyperparameter Optimization through Reversible Learning](http://arxiv.org/abs/1502.03492 "Dougal Maclaurin, David Duvenaud, Ryan P. Adams")
* [Accelerating Stochastic Gradient Descent via Online Learning to Sample](http://arxiv.org/abs/1506.09016 "Guillaume Bouchard, Théo Trouillon, Julien Perez, Adrien Gaidon")
* [Weight Uncertainty in Neural Networks](http://arxiv.org/abs/1505.05424 "Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra")
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167 "Sergey Ioffe, Christian Szegedy")
* [Highway Networks](http://arxiv.org/abs/1505.00387 "Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber")
  * [Training Very Deep Networks](http://arxiv.org/abs/1507.06228 "Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber")
* [Improving neural networks by preventing co-adaptation of feature detectors](http://arxiv.org/abs/1207.0580 "Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov")
  * [Efficient batchwise dropout training using submatrices](http://arxiv.org/abs/1502.02478 "Ben Graham, Jeremy Reizenstein, Leigh Robinson")
  * [Dropout Training for Support Vector Machines](http://arxiv.org/abs/1404.4171 "Ning Chen, Jun Zhu, Jianfei Chen, Bo Zhang")
* [Regularization of Neural Networks using DropConnect](http://jmlr.org/proceedings/papers/v28/wan13.html "Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, Rob Fergus")
* [Distilling the Knowledge in a Neural Network](http://arxiv.org/abs/1503.02531 "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")
* [Random Walk Initialization for Training Very Deep Feedforward Networks](http://arxiv.org/abs/1412.6558 "David Sussillo, L.F. Abbott")
* [Domain-Adversarial Neural Networks](http://arxiv.org/abs/1412.4446 "Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand")
* [Compressing Neural Networks with the Hashing Trick](http://jmlr.org/proceedings/papers/v37/chenc15.html "Wenlin Chen, James Wilson, Stephen Tyree, Kilian Weinberger, Yixin Chen")

## Numerical Optimization
* [Recursive Decomposition for Nonconvex Optimization](http://homes.cs.washington.edu/~pedrod/papers/ijcai15.pdf "Abram L. Friesen, Pedro Domingos")
  * [Recursive Decomposition for Nonconvex Optimization: Supplementary Material](http://homes.cs.washington.edu/~afriesen/papers/ijcai2015sp.pdf "Abram L. Friesen, Pedro Domingos")
* [Beating the Perils of Non-Convexity: Guaranteed Training of Neural Networks using Tensor Methods](http://arxiv.org/abs/1506.08473 "Majid Janzamin, Hanie Sedghi, Anima Anandkumar")
* [Graphical Newton](http://arxiv.org/abs/1508.00952 "Akshay Srinivasan, Emanuel Todorov")
* [Gradient Estimation Using Stochastic Computation Graphs](http://arxiv.org/abs/1506.05254 "John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel")
* [Equilibrated adaptive learning rates for non-convex optimization](http://arxiv.org/abs/1502.04390 "Yann N. Dauphin, Harm de Vries, Yoshua Bengio")
* [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf "John Duchi, Elad Hazan, Yoram Singer")
* [ADADELTA: An Adaptive Learning Rate Method](http://arxiv.org/abs/1212.5701 "Matthew D. Zeiler")
* [ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient](http://arxiv.org/abs/1412.7419 "Caglar Gulcehre, Yoshua Bengio")
* [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980 "Diederik Kingma, Jimmy Ba")
* [A sufficient and necessary condition for global optimization](http://www.sciencedirect.com/science/article/pii/S0893965909002869 "Dong-Hua Wu, Wu-Yang Yu, Quan Zheng")
* [Unit Tests for Stochastic Optimization](http://arxiv.org/abs/1312.6055 "Tom Schaul, Ioannis Antonoglou, David Silver")
* [A* Sampling](http://arxiv.org/abs/1411.0030 "Chris J. Maddison, Daniel Tarlow, Tom Minka")
* [Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems](http://arxiv.org/abs/1505.05114 "Yuxin Chen, Emmanuel J. Candes")
* [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767 "Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind")

## Numerical Precision
* [Deep Learning with Limited Numerical Precision](http://arxiv.org/abs/1502.02551 "Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, Pritish Narayanan")
* [Low precision storage for deep learning](http://arxiv.org/abs/1412.7024 "Matthieu Courbariaux, Yoshua Bengio, Jean-Pierre David")
* [1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs](http://research.microsoft.com/apps/pubs/?id=230137 "Frank Seide, Hao Fu, Jasha Droppo, Gang Li, Dong Yu")

## Cognitive Architectures
* [Derivation of a novel efficient supervised learning algorithm from cortical-subcortical loops](http://journal.frontiersin.org/article/10.3389/fncom.2011.00050/full "Ashok Chandrashekar, Richard Granger")
* [A Minimal Architecture for General Cognition](http://arxiv.org/abs/1508.00019 "Michael S. Gashler, Zachariah Kindle, Michael R. Smith")

## Motion Planning
* [Continuous Character Control with Low-Dimensional Embeddings](https://graphics.stanford.edu/projects/ccclde/ "Sergey Levine,	Jack M. Wang,	Alexis Haraux,	Zoran Popović,	Vladlen Koltun")
* [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702 "Sergey Levine, Chelsea Finn, Trevor Darrell, Pieter Abbeel") ([youtu.be/Q4bMcUk6pcw](http://youtu.be/Q4bMcUk6pcw))
* [Sampling-based Algorithms for Optimal Motion Planning](http://arxiv.org/abs/1105.1186 "Sertac Karaman, Emilio Frazzoli") ([youtu.be/r34XWEZ41HA](http://youtu.be/r34XWEZ41HA))
  * [Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic](http://arxiv.org/abs/1404.2334 "Jonathan D. Gammell, Siddhartha S. Srinivasa, Timothy D. Barfoot") ([youtu.be/nsl-5MZfwu4](http://youtu.be/nsl-5MZfwu4))
  * [Batch Informed Trees (BIT*): Sampling-based Optimal Planning via the Heuristically Guided Search of Implicit Random Geometric Graphs](http://arxiv.org/abs/1405.5848 "Jonathan D. Gammell, Siddhartha S. Srinivasa, Timothy D. Barfoot") ([youtu.be/TQIoCC48gp4](http://youtu.be/TQIoCC48gp4))

## Computational Creativity
* [Inceptionism: Going Deeper into Neural Networks](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html "Alexander Mordvintsev, Christopher Olah, Mike Tyka")
  * [DeepDream - a code example for visualizing Neural Networks](http://googleresearch.blogspot.com/2015/07/deepdream-code-example-for-visualizing.html "Alexander Mordvintsev, Christopher Olah, Mike Tyka")
* [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576 "Leon A. Gatys, Alexander S. Ecker, Matthias Bethge")
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "Andrej Karpathy")
* [GRUV: Algorithmic Music Generation using Recurrent Neural Networks](http://cs224d.stanford.edu/reports/NayebiAran.pdf "Aran Nayebi, Matt Vitelli")
* [Composing Music With Recurrent Neural Networks](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/ "Daniel Johnson")

## Cryptography
* [Crypto-Nets: Neural Networks over Encrypted Data](http://arxiv.org/abs/1412.6181 "Pengtao Xie, Misha Bilenko, Tom Finley, Ran Gilad-Bachrach, Kristin Lauter, Michael Naehrig")

## Distributed Computing
* [Dimension Independent Similarity Computation](http://arxiv.org/abs/1206.2082 "Reza Bosagh Zadeh, Ashish Goel")
  * [Dimension Independent Matrix Square using MapReduce](http://arxiv.org/abs/1304.1467 "Reza Bosagh Zadeh, Gunnar Carlsson")
  * [All-pairs similarity via DIMSUM](https://blog.twitter.com/2014/all-pairs-similarity-via-dimsum "Reza Zadeh")
* [A Fast, Minimal Memory, Consistent Hash Algorithm](http://arxiv.org/abs/1406.2294 "John Lamping, Eric Veach")

## Clustering
* [Clustering by fast search and find of density peaks](https://dl.dropboxusercontent.com/u/182368464/2014-rodriguez.pdf "Alex Rodriguez, Alessandro Laio")
