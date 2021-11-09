# Reading List<br>
## Uncertainty 
### [The Hidden Uncertainty in a Neural Networks Activations](https://arxiv.org/abs/2012.03082)
Differentiate between Epistemic (due to choise of model/parameters) and aleatoric (due to noise in the data) uncertanities
- Epistemic uncertanity: Example OOD data or informative datapoints that can be used in active learning
- Aleatoric uncertainty: Inherent in the data such as a 3 that is similar to an 8.<br>
Use density of features in hidden layers of the network to estimate the aleatoric uncertainty via entropy. 
Only shown for classification and regression, no CNNs.<br>
In order to do this, they model the probability density of the features in a shallow layer using:
- Reduce dimensionality using PCA
- Generative modeling (GMM) to learn `p(features|label)` and count labels on training set to get `p(label)`<br>
Model uncertanity defined as suprisal to see the feature representation `-log(p(feature)) = -log(\int p(feature|y)\cdotp(y) dy` and the aleatoric uncertainty
with the expected suprisal of `h(y|feature) = - \int p(y|feature) \cdot log(p_\theta)(y|feature) dy`<br>

### [Uncertainty Estimation Using a Single Deep Deterministic Neural Network](https://arxiv.org/pdf/2003.02037.pdf)
Use the idea of RBF networks. Measure uncertainty as distance between model output and closest centroid.
They use ResNet as feature extractor and remove softmax layer. They then create one centroid for each class and assign the distance 
between model and output by calculating `rbf(Wc*feature - centroid)`, where Wc is a weight matrix for each centroid.<br>
- Loss : Cross entropy loss between distance to centroid and one hot label<br>
- Feature collapse: NN tend to map different inputs to the same feature to discard noise etc. This can be disadventage if we try
to find OOD inputs, which might be collapsed to known features. To overcome this, they regularise the feature maps using gradient penalty.
  
DUQ captures aleatoric (close to multiple centroids) and epistemic (far away from any centroid) uncertainty.
  <br>
## Path Planning
### [An Efficient Sampling-based Method for Online Informative Path Planning in Unknown Environments](https://arxiv.org/pdf/1909.09548.pdf)
Introduce a sample based path planning algorithm similar to RRT*. Once a new node is executed, they do not discard tree but rewire 
the cut off subtrees. Additionaly they introduce a new value formulation for nodes to incoorporate custom goals as for instance tsdf reconstruction.<br>
#### Algorithm
- Each node of tree consists of trajectory, gain, cost and value.
  - Gain can be any function that only depends on the final pose
  - Cost is usually distance
  - Value is a fusion of cost and gain into an utility value
    
Tree expansion:
- If there are not enough viewpoints available, sample next viewpoint close otherwise sample one globally exploring unmapped space.
Updating tree:
- Set current node to root node
- Rewire old branches that are not connected due to root node change if possible<br>
<br>
<br>
## RL for Active Learning
#### [Semantic Curiosity for Active   Learning](https://arxiv.org/pdf/2006.09367.pdf)
Use inconsistentency in detector as reward. Objects should obtain the same label for different viewpoints.
They measure inconsistencies by measuring the temporal entropy of prediction- if an object is labeled with different classes as the viewpoint changes, it will have 
high temporal entropy.
Trajectories with high temporal entropies are then labeled via an oracle and used to retrain.<br>
Implementation
Use exploration policy which is used to sample N trajectories which should then be labeled.<br>
1) Create a semantic mapping from RGB-D images and store it in a 3-dimensional tensor (categories, width, length). 
   The height component was removed/reduced. This map is used to associate predictions while the agent is moving.
   Goal is to obtain a map that has many different assignement for each point.
   Then train reinforcement algorithm.
   
#### Baselines
- Random
- Prediction Error Curiosity: [https://arxiv.org/abs/1705.05363](https://arxiv.org/abs/1705.05363)
- Object Exploration: Use detected objects as reward
- Coverage Exploration: Maximize explored area [https://openreview.net/forum?id=SyMWn05F7](https://openreview.net/forum?id=SyMWn05F7)
- Active Neural Slam: Maximize explored area [https://openreview.net/forum?id=HklXn1BKDH](https://openreview.net/forum?id=HklXn1BKDH)

### [Embodied Visual Active Learning for Semantic Segmentation](https://arxiv.org/pdf/2012.09503.pdf)
We study the task of embodied visual active learning, where an agent is set to explore a 3d environment with the goal to acquire visual scene understanding by actively selecting views for which to request annotation.<br>
The agents are equipped with a semantic segmentation network and seek to acquire informative views, move and explore in order to propagate annotations in the neighbourhood of those views, then refine the underlying segmentation network by online retraining.<br>
#### Evaluation:
To assess how successful an agent is on the task, we test how accurate the perception module is on multiple random viewpoints selected uniformly in the area explored by the agent.<br>
The goal is to maximize the mIoU and mean accuracy of the segmentation network on the views in the area explored by the agent.<br>
#### Task:
The agent begins each episode randomly positioned and rotated in a 3d environment, with a randomly initialized semantic segmentation network. The ground truth segmentation mask for the first view is given for the initial training of the segmentation network. The agent can choose movement actions (MoveForward, MoveLeft, MoveRight, RotateLeft, RotateRight with 25 cm movements and 15 degree rotations), or perception actions (Annotate, Collect)
If the agent moves or rotates, the ground truth mask is propagated using optical flow<br>
#### RL:
- Actions<br>
- State
    - Given by: InputImage x SemanticSegmentationMask x PropagatedAnnotation x FeatureMap<br>
- Reward
    - Use final mIoU improvement of final segmentation.<br>
- Baselines
    - Random actions 
    - Only rotate left
    - Bounce: Walk forward until hits wall than sample new random direction
    - Frontier Exploration: Build a map online
    - Space filler: Follow a shortest space filling curve within view radius r. <br>

### [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf)
We formulate curiosity as the error in an agent's ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model<br>
Train a model to encode state into a feature vector $\phi(s_t)$. Train a second network 
to predict action that was taken for two given states.<br>
Additionally, train an inverse network that predicts $s_t+1$ given $s_t$ and action $a$.<br>
Intrinsic reward than is calculated as difference between predicted next state (feature) and true next state.<br>

## Informative Path Planing

### [An informative path planning framework for UAV-based terrain monitoring](https://link.springer.com/content/pdf/10.1007/s10514-020-09903-2.pdf)
They introduce a path planning algorithm for a drone with top down view and a 2D Map that should be explored<br>
A lot of focus is also put on a map representation that is continuous and uses GP e.g. Gas concentration as each point. <br>
They do not use RRT but splines 
#### Problem Statement:
The aim is to maximize the information collected about the environment, while respecting resource constraints, such as energy, time, or distance budgets.<br>
We seek an optimal trajectory ψ∗ in the space of all continuous trajectories for maximum gain in some information-theoretic measure: ψ∗<br>
#### Discrete case: 
Map terrain to 2D occupancy grid. Update each cell depending on classifier output and probability that this output is correct given by a sensor model.

#### Cont. case:    
Use GPs to encode correlations in environmental distribution. Observations are weighted with a variance matrix depending on the height of the UAV

#### Path Planning
First calculate optimal initial trajectory based on coarse grid search in the 3D workspace. <br>
Then refine solution using Covariance Matrix Adaptation Evolution Strategy

#### Utility definition
- Pure Exploration: Maximize the entropy of the Map given by H(map_prior_measure) - H(map_posterior_measure)
- Region of interest: Maximize entropy but only use points that satisfy interest condition. E.g. only take point where p(weed) > 10%.


### [Informative Path Planning for Extreme Anomaly Detection in Environment Exploration and Monitoring](https://arxiv.org/pdf/2005.10040.pdf)
**7.April 2021**
Also use GP to model map
- Assumes each measurement of UAV can be written as f(z,t) + e, e ~ N(0,sigma)
- z_n+1 = arg min \int _S(Zn,z) a(z,t; f,D) * Ds, f is surrogate function modeled with GP, Dn data measured so far
- Use dublin paths to optimize (S(Zn,z))

## Acquisition Functions a()
- Uncertainty sampling: Use uncertainty of GP -> a(x) = sigma^2(x)
- Integrated variance reduction (IVR): Assume we would observe point x. How much will this reduce variance at point x?
- Reduce complexity tails 

### [Online Informative Path Planning for Active Classification Using UAVs](https://arxiv.org/pdf/1609.08446.pdf)
2D occupancy grid. Each cell is a bernoulli random variable indicating the probability of weed occupancy. <br>
Update cells based on log likelyhood given classification assignement.

Use 12-degree polynomial paths to connect different viewpoints.


### [Volumetric Occupancy Mapping With Probabilistic Depth Completion for Robotic Navigation](https://arxiv.org/pdf/2012.03023.pdf)
Infer uncertainty from depth images and include them into the tsdf volume.
Also use depth completition network

### [An Exploration of Embodied Visual Exploration](https://arxiv.org/pdf/2001.02192.pdf)

### [LEARNING EXPLORATION POLICIES FOR NAVIGATION](https://openreview.net/pdf?id=SyMWn05F7)
Numerous past works have tackled the problem of task-driven navigation.  But,how to effectively explore a new environment to enable a variety of down-streamtasks has received much less attention.  In this work,  we study how agents canautonomously explore realistic and complex 3D environments without the contextof task-rewards.  We propose a learning-based approach and investigate differentpolicy architectures, reward functions, and training paradigms.  We find that useof policies with spatial memory that are bootstrapped with imitation learning andfinally finetuned with coverage rewards derived purely from on-board sensors canbe effective at exploring novel environments.  We show that our learned explo-ration  policies  can  explore  better  than  classical  approaches  based  on  geometryalone and generic learning-based exploration techniques.

### [Emergence of exploratory look-around behaviors through active observation completion](https://robotics.sciencemag.org/content/4/30/eaaw6326/tab-pdf)
Standard computer vision systems assume access to intelligently captured inputs (e.g., photos from a human
photographer), yet autonomously capturing good observations is a major challenge in itself. We address the
problem of learning to look around: How can an agent learn to acquire informative visual observations? We
propose a reinforcement learning solution, where the agent is rewarded for reducing its uncertainty about the
unobserved portions of its environment. 