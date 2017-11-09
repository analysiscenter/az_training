
1. Hi! Today I'm gonna talk about ResNet architecture which won the 1st place in 2015 on the ImageNet classification challenge.

2. It was a breaktrough because of the proposed residual learning framework that eased the training process and allowed to 
train substantially deeper networks than those used previously. Remarkably although the depth significantly increased, 152 layers ResNet still had lower complexity than VGG-19.

So before ResNet there was a problem with training deeper networks: adding more layers to suitably deep model led to higher training error at the same number of iterations. (pic) 

3. The authors of ResNet managed to solve this problem introducing a 34, 50, 101 and 152 layered architecture with every deeper network having lower test error.

But you can make as many layers in your ResNet as you need by stacking "residual blocks" of the same connecting shape.
	- The residual block is the main power of ResNet architectures:
		The original one consists of 2 3x3 convolutional layers and is defined as 	
		y = F(x, {W_i}) + x
			where x and y are the input and output of the layers considered, W_i={W_{i, k} | 1 <= k <= K} is a set of weights and biases in the block, K is the number of layers in the block.
 		This addition of the input and output is also called skipconnection and it is the central idea of resnet.
		Skipconnection creates a 'direct' path for propagating information between shallow and deeper layers. In a backpropagation mode that means the gradients of a layer do not vanish even when the weights are arbitrarily small.

 		Let's have a look at the gradients formula to explain why:
 		
 		(formulas will be on the slide)
 		If x_{l+1} = x_l + F(xl, Wl) then recursively
 		x_{l+2} = x_{l+1} + F(x_{l+1}, W_{l+1}) = x_l + F(xl, Wl) + F(x_{l+1}, W_{l+1}, etc we will have 
 		x_K = x_l + \sum_{i=1}^{K-1} F(x_i, W_i)
 		for any deeper block K and any shallower block l.
 		Denoting the loss function as E from the chain rule of backpropagation we have:
 		dE/dxl = dE/dxK * dxK/dxl = dE/dxK (1 + d/dxl \sum_{i=1}^{K-1} F(xi, Wi))
 		The additive term of dE/dxK ensures that information is directly propagated to any shallower block l. 

		Actually usually there is also Relu activation applied to the result of addition but the construction above also works if you apply relu only to the input of the the convolution layer. For more information go to the original paper with experiments. I leave the link bellow.  

	So this construction of the residual block with a skipconnection made optimization of a very deep networks possible. (pic)
 	
 	- But making networks deeper one faces an increased computational costs. Therefore in ResNets with more than a hundred layers authors replaced two 3x3 convolutional layers with a stack of 1x1 ,3x3 and 1x1 convolutional layers. 1x1 layers are responsible for reducing and then restoring
 	dimensions, leaving the 3x3 layer a bottleneck with smaller input/output dimensions and fewer parameters to optimize.

 	4. Authors of the ResNet stated that a network depth is a key to success of modern deep learning but in proposed later WideResNet architecture they argued that the effect of depth may be supplementary. A 50 layers WideResNet outperforms 152 layer original ResNet on ImageNet and takes several times less time to train, thus showing that the main power of residual networks is in residual blocks and not in an extreme depth as claimed earlier.

 	5. So to sum it up: today we discussed the ResNet architectures which made it possible to train a very deep networks due to an idea of skipconnections and bottleneck. Additional shorter connections between layers turned out to be a very powerful technique which is used in many modern architectures.