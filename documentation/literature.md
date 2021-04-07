# Reading List

## Uncertainty 
### [The Hidden Uncertainty in a Neural Networks Activations](https://arxiv.org/abs/2012.03082)
Differentiate between Epistemic (due to choise of model/parameters) and aleatoric (due to noise in the data) uncertanities
- Epistemic uncertanity: Example OOD data or informative datapoints that can be used in active learning
- Aleatoric uncertainty: Inherent in the data such as a 3 that is similar to an 8.

Use density of features in hidden layers of the network to estimate the aleatoric uncertainty via entropy. 
Only shown for classification and regression, no CNNs.

In order to do this, they model the probability density of the features in a shallow layer using:
- Reduce dimensionality using PCA
- Generative modeling (GMM) to learn `p(features|label)` and count labels on training set to get `p(label)`

Model uncertanity defined as suprisal to see the feature representation `-log(p(feature)) = -log(\int p(feature|y)\cdotp(y) dy` and the aleatoric uncertainty
with the expected suprisal of `h(y|feature) = - \int p(y|feature) \cdot log(p_\theta)(y|feature) dy`


### [Uncertainty Estimation Using a Single Deep Deterministic Neural Network](https://arxiv.org/pdf/2003.02037.pdf)
Use the idea of RBF networks. Measure uncertainty as distance between model output and closest centroid.
They use ResNet as feature extractor and remove softmax layer. They then create one centroid for each class and assign the distance 
between model and output by calculating `rbf(Wc*feature - centroid)`, where Wc is a weight matrix for each centroid.

- Loss : Cross entropy loss between distance to centroid and one hot label

- Feature collapse: NN tend to map different inputs to the same feature to discard noise etc. This can be disadventage if we try
to find OOD inputs, which might be collapsed to known features. To overcome this, they regularise the feature maps using gradient penalty.
  
DUQ captures aleatoric (close to multiple centroids) and epistemic (far away from any centroid) uncertainty.
  

## Path Planning
### [An Efficient Sampling-based Method for Online Informative Path Planning in Unknown Environments](https://arxiv.org/pdf/1909.09548.pdf)
Introduce a sample based path planning algorithm similar to RRT*. Once a new node is executed, they do not discard tree but rewire 
the cut off subtrees. Additionaly they introduce a new value formulation for nodes to incoorporate custom goals as for instance tsdf reconstruction.

#### Algorithm
- Each node of tree consists of trajectory, gain, cost and value.
  - Gain can be any function that only depends on the final pose
  - Cost is usually distance
  - Value is a fusion of cost and gain into an utility value
    
Tree expansion:
- If there are not enough viewpoints available, sample next viewpoint close otherwise sample one globally exploring unmapped space.
Updating tree:
- Set current node to root node
- Rewire old branches that are not connected due to root node change if possible





## RL for Active Learning
### [Semantic Curiosity for Active   Learning](https://arxiv.org/pdf/2006.09367.pdf)
Use inconsistentency in detector as reward. Objects should obtain the same label for different viewpoints.
They measure inconsistencies by measuring the temporal entropy of prediction- if an object is labeled with different classes as the viewpoint changes, it will have 
high temporal entropy.
Trajectories with high temporal entropies are then labeled via an oracle and used to retrain.

Implementation
Use exploration policy which is used to sample N trajectories which should then be labeled.

1) Create a semantic mapping from RGB-D images and store it in a 3-dimensional tensor (categories, width, length). 
   The height component was removed/reduced. This map is used to associate predictions while the agent is moving.
   Goal is to obtain a map that has many different assignement for each point.
   Then train reinforcement algorithm.
   
## Baselines
- Random
- Prediction Error Curiosity: [https://arxiv.org/abs/1705.05363](https://arxiv.org/abs/1705.05363)
- Object Exploration: Use detected objects as reward
- Coverage Exploration: Maximize explored area [https://openreview.net/forum?id=SyMWn05F7](https://openreview.net/forum?id=SyMWn05F7)
- Active Neural Slam: Maximize explored area [https://openreview.net/forum?id=HklXn1BKDH](https://openreview.net/forum?id=HklXn1BKDH))


### [Embodied Visual Active Learning for Semantic Segmentation](https://arxiv.org/pdf/2012.09503.pdf)
We study the task of embodied visual active learning, where an agent is set to explore a 3d environment with the goal to acquire visual scene understanding by actively selecting views for which to request annotation.

The agents are equipped with a semantic segmentation network and seek to acquire informative views, move and explore in order to propagate annotations in the neighbourhood of those views, then refine the underlying segmentation network by online retraining.

### Evaluation:
To assess how successful an agent is on the task, we test how accurate the perception module is on multiple random viewpoints selected uniformly in the area explored by the agent.

The goal is to maximize the mIoU and mean accuracy of the segmentation network on the views in the area explored by the agent.

### Task:
The agent begins each episode randomly positioned and rotated in a 3d environment, with a randomly initialized semantic segmentation network. The ground truth segmentation mask for the first view is given for the initial training of the segmentation network. The agent can choose movement actions (MoveForward, MoveLeft, MoveRight, RotateLeft, RotateRight with 25 cm movements and 15 degree rotations), or perception actions (Annotate, Collect)
If the agent moves or rotates, the ground truth mask is propagated using optical flow

### RL:
- Actions

- State
    - Given by: InputImage x SemanticSegmentationMask x PropagatedAnnotation x FeatureMap

- Reward
    - Use final mIoU improvement of final segmentation.

- Baselines
    - Random actions 
    - Only rotate left
    - Bounce: Walk forward until hits wall than sample new random direction
    - Frontier Exploration: Build a map online
    - Space filler: Follow a shortest space filling curve within view radius r. 


### [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf)
We formulate curiosity as the error in an agent's ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model

Train a model to encode state into a feature vector $\phi(s_t)$. Train a second network 
to predict action that was taken for two given states.

Additionally, train an inverse network that predicts $s_t+1$ given $s_t$ and action $a$.

Intrinsic reward than is calculated as difference between predicted next state (feature) and true next state.
