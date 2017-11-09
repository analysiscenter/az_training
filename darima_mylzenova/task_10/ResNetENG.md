ResNet

1. Задумывалась для классификации, детекции
2. Позволила строить глубокие сети, решила проблему затухания градиентов и уменьшило число параметров в слоях.
3. Может иметь разное количество резблоков и соответственно разную глубину
	- В каждом блоке скипконекшн, который ...
	- Боттлнек, который ...
4. - Чем глубже, тем дольше. Можно короче, но шире с тем же качестом (wide resnet). 
- Фишка: почти все равно, макспул или конв со страйдом.
5. Вывод: улучшило качество глубоких сетей благодаря предложенным конструкциям скипконекшн, боттлнек.



0. Hi! Today I'm gonna talk about ResNet architecture
<!-- which was originally designed for classification and segmentation task.  -->
which won the 1st place in 2015 on the ImageNet classification challenge.

2. It was a breaktrough because of the proposed residual learning framework that eased the training process and allowed to 
train substantially deeper networks than those used previously. Remarkably although the depth significantly increased, 152 layers ResNet still had lower complexity than VGG-19.

3. ResNets are modularized architectures that stack building blocks of the same connecting shape. слои
<!-- which means that you can stack building blocks of the same connecting shape and make the net as deep as you want.
 -->

	- The original building block consists of 2 3x3 convolutional layers and is defined as 
		y = F(x, {W_i}) + x
			where x and y are the input and output of the layers considered, W_i={W_{i, k} | 1 <= k <= K} is a set of weights and biases in the block, K is the number of layers in the block.
			<!-- and F(x, {W_i}) represents the mapping to be learned.
 -->	
 
<!--  		This is the central idea of resnet: consider y = H(x) a mapping to be fit by a few stacked layers. But instead of approximating H(x) by attaching an identity skip connection we let these layers approximate a residual function F(x) = H(x) - x. That is why it is called residual learning. 
 -->
 		This simple idea creates a 'direct' path for propagating information in a backpropagation mode which means that the gradients of a layer does not vanish even when the weights are arbitrarily small.

 		Let's look at the gradients formula to explain why:
 		
 		(formulas will be on the slide)
 		If x_{l+1} = x_l + F(xl, Wl) then recursively
 		x_{l+2} = x_{l+1} + F(x_{l+1}, W_{l+1}) = x_l + F(xl, Wl) + F(x_{l+1}, W_{l+1}, etc we will have 
 		x_K = x_l + \sum_{i=1}^{K-1} F(x_i, W_i)
 		for any deeper unit K and any shallower unit l.
 		Denoting the loss function as E from the chain rule of backpropagation we have:
 		dE/dxl = dE/dxK * dxK/dxl = dE/dxK (1 + d/dxl \sum_{i=1}^{K-1} F(xi, Wi))
 		The additive term of dE/dxK ensures that information is directly propagated to any shallower unit l. 

 	Обратно про сети и графики 
 	иерархическая структура - 
 	тезисы
 	ссылка на исследования батчнорм и релу

 	- The construction of the residual block with a skipconnction made optimization of a very deep networks possible. But making networks deeper one faces an increased computational costs. Therefore in ResNets with more than a hundred layers authors replaced two 3x3 convolutional layers with a stack of 1x1 ,3x3 and 1x1 convolutional layers. 1x1 layers are responsible for reducing and then restoring
 	dimensions, leaving the 3x3 layer a bottleneck with smaller input/output dimensions and fewer parameters to optimize.

 	4. Authors of the ResNet stated that a network depth is a key to success of modern deep learning but in proposed later WideResNet architecture they argued that the effect of depth may be supplementary. A 50 layers WideResNet outperforms 152 layer original ResNet on ImageNet and takes several times less time to train, thus showing that the main power of residual networks is in residual blocks and not in an extreme depth as claimed earlier.

 	5. So to sum it up: today we discussed the ResNet architectures which made it possible to train a very deep networks due to an idea of skipconnections and bottleneck. Additional shorter connections between layers turned out to be a very powerful technique which is used in many modern architectures.





 		through the entire network 

 		a mapping done by the layers ib the block  instead of


 because of the proposed consept of skipconnections

