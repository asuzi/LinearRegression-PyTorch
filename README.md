# LinearRegression-PyTorch
Here I have created a simple dataset using the formula for linear regression. (<b> y = w * X + b </b>) with known variables for weight and bias. How ever the model is unaware of these variables and is trying to learn the correct variables to predict the Testing data <i>(green dots)</i>. 
<br><br>
<b>Dataset</b>
There is fifty (50), samples in total created which has been split into training and testing data, making the total samples for training data: forty (40) and testing data: ten (10)
<br><br>
<b>Model</b>
The model itself is using only one linear layer with one (1) input and one (1) output.
<br><br>
<b>Training</b>
For the training of the model, I have implemented the <i>nn.L1Loss()</i> function to calculate the loss and <i>torch.optim.SGD()</i> optimizer.
With hyperparameters, <b>Learning Rate</b> = 0.01 and <b>Epoch</b> = 1000
<br><br>
The image below is taken from the models first initial guess (totally random)
![image](https://github.com/asuzi/LinearRegression-PyTorch/assets/61744031/b2b80b2c-b7db-474d-9198-8f285bb078a2)

We can see that after one thousand (1000) training rounds the model is able to predict the Testing data, quite accurately.<br>
The best traing results for the loss were <b>0.0022</b><br>
And in the image below we can see that the loss for testing was <b>0.0086</b>
![image](https://github.com/asuzi/LinearRegression-PyTorch/assets/61744031/3ba059f9-e921-4183-a2f0-dd9ca9ed8ae5)
