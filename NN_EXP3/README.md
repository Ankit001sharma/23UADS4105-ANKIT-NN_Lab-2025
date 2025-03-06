<h1>Object 3</h1>
<h4>Program to implement a three-layer neural network using Tensor flow library (only, no
keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation
of feed-forward and back-propagation approaches.</h4>
<hr>

<h3>Description of the Model </h3>

<b>Data Preprocessing:</b>

The MNIST dataset is loaded and normalized (pixel values scaled between 0 and 1).
Images are flattened from 28×28 to a 1D array of size 784.
Labels are one-hot encoded for classification.
<hr>

<b>Model Architecture:</b>
<ul>
<li>Input Layer: 784 neurons (one for each pixel in the image).</li>
<li>Hidden Layer 1: 128 neurons with sigmoid activation.</li>
<li>Hidden Layer 2: 64 neurons with sigmoid activation.</li>
<li>Output Layer: 10 neurons (one for each digit 0-9) with softmax activation.</li>
</ul>
<hr>

<b>Training Mechanism:</b>

Uses Adam optimizer with a learning rate of 0.01.
Cross-entropy loss function is used to measure classification error.
Model is trained in mini-batches of size 64 over 10 epochs.
Loss is calculated and updated at each step using gradient descent.
<hr>
<b>Performance Evaluation:</b>

After training, the model is tested on the test dataset.
Accuracy is computed by comparing predicted labels with actual labels.
A loss curve is plotted to visualize how the model's loss decreases over epochs.
<hr>
<b>Visualization:</b>

The model plots the loss curve to analyze training progress.
Sample predictions are displayed to see how well the model recognizes handwritten digits

<hr>
<h3>Code description  :- </h3>
<ol>
<li><b>Preprocessing the MNIST Dataset</b><br>
<ul>
<li>Normalization: Pixel values are scaled from 0-255 to 0-1 for faster convergence during training.</li>
<li>Flattening Images: The 28×28 grayscale images are converted into 1D arrays of 784 elements.</li>
<li>One-hot Encoding: Integer labels (0-9) are converted into one-hot vectors of size 10.</li>
</ul></li><hr>
<li>
<b>Defining the Neural Network</b><br>
<ul>
<li>Input Layer: 784 neurons.</li>
<li> Hidden Layer 1: 128 neurons, initialized with random weights (W1) and biases (b1).</li>
<li>Hidden Layer 2: 64 neurons with (W2, b2).</li> 
<li>Output Layer: 10 neurons for classification (W3, b3).</li>
<li>Matrix multiplication (tf.matmul()) applies weights and biases.</li>
<li>Sigmoid activation is used for hidden layers.</li>
<li>Softmax activation is used at the output layer for probability distribution over 10 classes.</li>

</ul></li><hr>

<li><b>Defining the Loss Function and Optimizer</b>

<ul>
<li>Loss Function: Computes cross-entropy loss for classification.</li>
<li>Optimizer: Uses Adam optimizer with a learning rate of 0.01 for updating weights.</li></ul></li><hr>

<li><b>Defining the Training Step</b>
<ul>
<li>Uses TensorFlow's GradientTape to compute gradients for backpropagation.</li>
<li>The weights and biases are updated using the computed gradients.</li> 
</ul>
</li><hr><li>

<b>Evaluating the Model on Test Datasets</b><br>
Innference is performed on the test dataset and accuracy is calculated.
</li><hr><li>
<b>Training Loop</b> 
<ul>
<li>Training is carried out for 10 epochs.</li>
<li>Batch size is set to 64.</li>
<li>Stores loss values for visualization.</li>
<li>Training is carried out in mini-batches:<br>
a. Loss is computed and weights are updated for each batch .<br>
b. The progress bar is updated with current loss.<br>
c. Loss per epoch is stored for visualisation.</li>
</ul>
</li><hr><li>
<b> Evaluating the Model After Training</b>
<ul><li>The final test accuracy after training is calculated.</li></ul> 
</li><hr><li>
<b>Visualisation</b>
<ul><li>The loss curve is plotted to track training performance.</li>
<li>The first 5 images are selected from the test set.</li>
<li>Each image is displayed along with the model's predicted class and actual class.</li></ul>
</ol>
<hr></li>

<h3>My Comments</h3>

<ul>
<li>The model may take a lot of time to train for e.g. A 6gb vram graphics card <i>(Nvidia RTX 4050 Laptop GPU)</i> takes around 2 minutes for 10 epochs and batch size 64. In order to improve the training time the hardware should be equipped with the robust specifications. 
</li>

<li>Instead of using the sigmoid function as activation function, rectified linear unit (ReLu) function can be used which is comparatively simple and hence improves training time.</li>

<li>More layer can be added to learn complicated patterns and improve accuracy. </li>
</ul><hr>






