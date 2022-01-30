$\hat{y} = f(\sum_{i=1}^{d} WX + b)$

# Perceptron

**Single Layer Perceptron**은 최초로 제안된 신경 모델이다. 뉴런의 로컬 메모리는 뉴런에 존재하는 벡터의 weight로 구성된다. Single Layer Perceptron의 각 **계산**은 각각의 weight의 벡터에 해당 element를 곱합 값을 가진 입력 벡터의 총합을 통해 수행된다. 출력에 표시되는 값은 활성화 기능의 입력이 된다.

![1](https://github.com/junsu9637/Artificial-Intelligence/blob/main/Cheat%20Sheet/Artificial%20Intelligence%20Cheat%20Sheet/Image/Perceptron_01.jpg?raw=true)

이미지 분류 문제를 Single Layer Perceptron으로 구현해보자. Single Layer Perceptron은 다음과 같은 **Logistic Regression**을 바탕으로 표현된다.

![2](https://github.com/junsu9637/Artificial-Intelligence/blob/main/Cheat%20Sheet/Artificial%20Intelligence%20Cheat%20Sheet/Image/Perceptron_02.jpg?raw=true)

로지스틱 회귀 학습을 위한 과정은 다음과 같다.
```markdown
1. Weight는 학습을 시작할 때 임의의 값으로 초기화된다.
2. Error은 training set의 각 element에 대해 원하는 출력과 실제 출력 간의 차이로 계산된다. 계산된 Error은 Weight를 조정하는데 사용한다.
3. 전체 training set에서 발생하는 Error가 지정된 임계값 이상이 될 때까지 지정된 반복 횟수만큼 반복한다.
```

로지스틱 회귀 분석을 위한 전체 코드는 다음과 같다.

```Python
# Import MINST data 
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) 

import tensorflow as tf 
import matplotlib.pyplot as plt 

# Parameters 
learning_rate = 0.01 
training_epochs = 25 
batch_size = 100 
display_step = 1 

# tf Graph Input 
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28 = 784 
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes 

# Create model 
# Set model weights 
W = tf.Variable(tf.zeros([784, 10])) 
b = tf.Variable(tf.zeros([10])) 

# Construct model 
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax 

# Minimize error using cross entropy 
cross_entropy = y*tf.log(activation) 
cost = tf.reduce_mean\ (-tf.reduce_sum\ (cross_entropy,reduction_indices = 1)) 

optimizer = tf.train.\ GradientDescentOptimizer(learning_rate).minimize(cost) 

#Plot settings 
avg_set = [] 
epoch_set = [] 

# Initializing the variables init = tf.initialize_all_variables()
# Launch the graph 
with tf.Session() as sess:
   sess.run(init)
   
   # Training cycle
   for epoch in range(training_epochs):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples/batch_size)
      
      # Loop over all batches
      for i in range(total_batch):
         batch_xs, batch_ys = \ mnist.train.next_batch(batch_size)
         # Fit training using batch data sess.run(optimizer, \ feed_dict = {
            x: batch_xs, y: batch_ys}) 
         # Compute average loss avg_cost += sess.run(cost, \ feed_dict = {
            x: batch_xs, \ y: batch_ys})/total_batch
      # Display logs per epoch step
      if epoch % display_step == 0:
         print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            avg_set.append(avg_cost) epoch_set.append(epoch+1)
   print ("Training phase finished")
    
   plt.plot(epoch_set,avg_set, 'o', label = 'Logistic Regression Training phase') 
   plt.ylabel('cost') 
   plt.xlabel('epoch') 
   plt.legend() 
   plt.show() 
    
   # Test model 
   correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1)) 
   
   # Calculate accuracy 
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) print 
      ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```
