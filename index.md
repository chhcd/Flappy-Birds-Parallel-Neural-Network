## TITLE

“Training A Neural Network to Play Flappy Bird” 

Diego Garza Rodriguez

Nihar Pol 

### SUMMARY

We are going to train a parallel reinforcement neural network to play flappy bird game on NVIDIA GPGPUs. 

### BACKGROUND

We want to implement a parallel reinforcement learning with neural networks to play the game “Flappy bird”. The idea is to have a different state according to the relative position of the bird with respect to the next pipe and to have two different possible actions (pressing the screen to make the bird go up or not pressing the screen—it will fall down due to gravity). The passive reinforcement learning allow us to assign positive and negative rewards to the environment to learn through each iteration the best action on each of the states. We want to parallelize this application in a way such that, for example, each core can run a single life in the game, acquire knowledge, and after all the cores have finished, combine the data with convolution in a single set of values and start over again. Consequently the computer will become better and better at playing the game after more and more iterations pass. The main goal is to speedup the process of learning in this game to achieve a high score as soon as possible. 
	
The compute intensive application we are trying to accelerate is the training and evaluation of the neural network involved in playing the game. As discussed in lecture, the simplest implementation of training a neural network is time consuming due to the many nested for loops. These processes can benefit from much parallelism. If we want to speedup even more the application, we would analyze the graphics ourselves to obtain the relative distance of the bird from the ground and the next pipe, operation that is being handle in a serial approach and is time consuming

### THE CHALLENGE

The challenge is to speed up this neural network training as much as possible. Although there do exist many libraries for training neural networks on the NVIDIA GPGPUs, we really want to focus on this specific game and see how fast we can make it by “hand-optimizing”. What makes this problem particularly challenging is that in the simple nested for loop implementation, there is a large frequency of cache misses and a large amount of data that is created and processed. 

One of the first design constraints we have identified is that it takes considerable time to ship data to and from the GPUs. Because training and evaluating a neural network involves a lot of data creation, ideally we want to keep that data on the GPU and not have to pass it back and forth between the host device.

If we reach our main goal, we will analyze the graphic in parallel to obtain helper functions of the relative distance to make our decision on each state, this is a challenging problem because it will involve computer vision analysis and dividing the work within thread in the cuda block to obtain a relative distance that could be divided across multiple threads.
  
### RESOURCES

We are planning on starting with two sets of code bases.

The first code base we will use is Assignment 3 from Diego’s AI class (http://www.cs.cmu.edu/~15381-f17/). We have already obtained permission from Dr. Tai Sing Lee to use this code as starter code in our project. The implementation for that homework is the serial implementation of the code that we want to use with 3 extra helper functions that makes things easier. An additional reward for being at the altitude of the pipe hole, horizontal distance from the bird to the next pipe and vertical distance from the bird to the pipe hole. The additional reward speeds the serial version from requiring 40 minutes to learn up to 5 minutes. The helper functions are performed with a computer vision optimization algorithm, but Dr. Lee suggest to do it ourselves in parallel if we have time left because is a good problem for parallelization.

The second code base we will use is the circles renderer code from Assignment II. We especially will take advantage of the data management portions of that code that handle reading from files. Also, the code launches a visual window that renders the circles and we want to use that application to display images in real time. 
The device we are planning on using is the NVIDIA GPGPUs. We don’t expect there to be any issues obtaining access or using the device unless it is close to the final deadline when everything is due. However, we should make it clear that this is the proposed device we are planning on using because we currently think that there is sufficient independence and parallelism in the calculations that makes this the best choice. We may, however, find out along the way that the device is too slow for what we want to do and that perhaps the Xeon Phis are better suited, with either OpenMP or OpenMPI. 

### GOALS AND DELIVERABLES
#### PLAN TO ACHIEVE 

We will begin with the assumption that the game is given to us as a precoded series of pipes and altitudes/distances. These will be given to us in an array. We will randomly generate this array and use this as the input. We will then train the neural network and evaluate it as the game is playing on the NVIDIA device itself. We will also need to develop the infrastructure of the game (calculating score, etc.) which does not seem to be too hard. The last component is that we want to see how our algorithm does in real time, so we want to actually render the game as the AI algorithm is playing it. We will demo the serial version of the code side by side with the optimized parallel version. We expect the serial version to take around 5 minutes to learn and achieve a score of 300 pipes in a row while the parallel version achieve that in 30 seconds. We believe that having multiple parallel birds playing the game, will increase the overall knowledge when combining each bird knowledge into one general knowledge data set that will be used for the next iteration.

#### HOPE TO ACHIEVE

If we achieve our goals specified in PLAN TO ACHIEVE, we would actually like to first change the input to our program. Rather than assume that the game is given to us as an array with a series of pipe distances and altitudes, we would like our program to actually process the image of the game in real time and thus (try to) determine the locations and altitudes of the pipes as they are coming. This will involve training and evaluating an entirely different neural network that is more related to image processing. The challenge here will be training and evaluating two different neural networks simultaneously on the same device. It may be possible to train the image processing network before it is deployed on the device by creating our own library of pipes and training it on that. 

We plan on having an interactive demo at the poster session that shows the program learning to the play in real-time with the output visible in the renderer application. We want to also contrast it with a CPU implementation that tries to learn to play the same game as fast as it can. We want to be able to have the audience visually see the speedup in parallelizing the code on the GPU. 

Here are some questions we want to answer over the course of our project: 
- Is the NVIDIA GPU the best device on which to train and evaluate a neural network? Although the GPU is best for matrix multiplication, it may not be as suited for large memory movement and storage. Because training and evaluating neural networks involves significant amounts of both of these aspects, what is the best hardware for this? 
- How is training and evaluating a neural network to play Flappy Bird different or similar to training and evaluating a neural network to process the images of the game Flappy Bird? 


### PLATFORM CHOICE

In lecture we learned that training and evaluating a neural network involves a significant amount of computation. Much of this computation, however, can be done in parallel because they are independent operations. If the computation is performed as a giant matrix multiply, for example, then the GPUs will be best suited for that. Also, most of the computations will involve floating point operations, and GPUs are optimized for that, too. 

### SCHEDULE 

Monday, November 6th
Do enough research to understand what computations are involved in training a neural network. 
Write pseudocode for training the neural network. 
Decide what parameters of the game to modify (altitude of pipes, distance between them, etc.) 
Decide how to encode these parameters as input and how to measure output (score of game) 
Read and understand assignment handout from Diego’s AI class. 

Monday, November 13th
Have a working serial implementation that runs on CPUs. 
Write render code that will render the game in real-time on GPUs. Speed of the renderer will be based on how much time it takes to train the network. 
Write render code that will render the game in real-time on CPUs. Speed of the renderer will be based on how much time it takes to train the network. 

Monday, November 20th
Have a working parallel implementation that runs on the GPUs.
Determine whether GPUs are the best device that we want to be using for this project. 
Compare performance on GPUs to performance on CPUs. Hope to have speedup of at least 5x. This is an arbitrary number that serves just as a ballpark number. 

Monday, November 27th
Further optimize parallel implementation that runs on the GPUs. Specifically, we want to have optimized the matrix multiplication aspect of the program. 
Start optimizing the data storage and management portion of the program. Can the amount of data that is created by reduced using (lossy) compression? 

Monday, December 4th
Keep optimizing parallel implementation that runs on GPUs using various methods, including data compression and approximation. We suspect this will take time to implement and debug. 

Tuesday, December 12th
Finish optimizing GPU implementation, possibly using various innovative, creative methods that were alluded to or weren’t discussed in lecture. We want to hand-tune the performance of our implementation so that it accomplishes the specific task of learning to play Flappy Bird as fast as possible. 
Record performance statistics and document them. Discuss our results in a document and reflect on the various things we tried, analyzing their outcomes. 
Get ready for final presentation and demo. 
