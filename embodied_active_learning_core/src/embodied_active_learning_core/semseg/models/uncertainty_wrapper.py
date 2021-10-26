import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sklearn for fitting
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as SKLEARN_PCA


class UncertaintyModel(nn.Module):
  """
  Wraps around any torch module and tries to extract features, clusters them and appends the negative log likelyhood
  to the output of the model
  """

  def __init__(self, base_model, feature_layer_name=None, n_feature_for_uncertainty=128, n_components=3, weights=None,
               means=None, covariances=None, covariance_type="tied", reg_covar=1e-6):
    """
    base_model: Any torch.nn classifier either returning prediction or a [tuple predictions, features]
    feature_layer_name: If model outputs features -> None, otherwise name of the feature layer whose activation should be used
    """
    super().__init__()
    self.base_model = base_model
    self.feature_layer_name = feature_layer_name
    self.features_from_output = feature_layer_name is None

    if not self.features_from_output:
      # Create hook to get features from intermediate pytorch layer
      self.features = {}

      def get_activation(name, features=self.features):
        def hook(model, input, output):
          features[name] = output.detach()

        return hook

      # get feature layer
      feature_layer = getattr(self.base_model, feature_layer_name)
      # register hook to get features
      feature_layer.register_forward_hook(get_activation(feature_layer_name))
    self.clustering = FeatureClustering(256, n_feature_for_uncertainty, n_components=n_components, weights=weights,
                                        means=means, covariances=covariances, covariance_type=covariance_type,
                                        reg_covar=reg_covar)

  def get_predictions(self, images):
    out = self.base_model(images)
    if self.features_from_output:
      prediction = out[0]
      features = out[1]
    else:
      # Features not part of model output, can access them from the model hook in self.features
      prediction = out
      features = self.features[self.feature_layer_name]
    return prediction, features

  def forward(self, images):
    prediction, features = self.get_predictions(images)

    with torch.no_grad():
      nll = -self.clustering(features.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
      nll = F.interpolate(nll, prediction.shape[-2:], mode='bilinear', align_corners=True)

    return prediction, nll


class UncertaintyFitter:
  """ Helper class to fit the GMM and PCA module """

  def __init__(self, uncertainty_model, total_features=-1, features_per_batch=500):
    self.uncertainty_model = uncertainty_model
    self.features = []
    self.features_per_batch = features_per_batch
    self.total_features = total_features

  def __call__(self, batch):
    images = batch.cuda()
    features = self.uncertainty_model.get_predictions(images)[1]
    features = features.cpu().detach().numpy()
    # reshaping
    feature_size = features.shape[1]
    features = features.transpose([0, 2, 3, 1]).reshape([-1, feature_size])
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(features.shape[0],
                                         size=[self.features_per_batch],
                                         replace=False)]
    self.features.append(features)

  def __enter__(self):
    self.features = []
    self.uncertainty_model.eval()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if len(self.features) != 0:
      features = np.asarray(self.features).reshape(-1, self.features[0].shape[-1])

      if self.total_features != -1 and features.shape[0] > self.total_features:
        # subsample features again
        features = features[np.random.choice(features.shape[0], size=self.total_features, replace=False), :]

      self.uncertainty_model.clustering.fit(features)
    self.features = []


class FeatureClustering(nn.Module):
  """ Feature clustering module. Performs PCA and then cluster using GMM """

  def __init__(self,
               n_in_features,
               n_features,
               n_components,
               apply_pca=True,
               means=None,
               covariances=None,
               weights=None,
               covariance_type="tied",
               reg_covar=1e-6):
    super().__init__()
    self.apply_pca = apply_pca
    self.n_features = n_features
    self.gmm = _TorchGMM(n_features, n_components, means, covariances, weights, covariance_type, reg_covar)
    self.pca = PCA(n_in_features, n_features)

  def fit(self, features):
    reduced_features = self.pca.fit(features)
    self.gmm.fit(reduced_features)

  def forward(self, features):
    if self.apply_pca:
      shape = features.shape
      features = self.pca(features).reshape([*shape[:-1], self.n_features])
    return self.gmm(features)


class PCA(nn.Module):
  """ PCA implementation based on sklearn. """

  def __init__(self, num_in_features, num_reduced_features, mean=None, components=None):
    super().__init__()
    if mean is None:
      mean = torch.empty((num_in_features,))
    if components is None:
      components = torch.empty((num_in_features, num_reduced_features))

    # Mean that should be subtracted [1, n_features]
    self.mean = torch.nn.parameter.Parameter(mean)
    self.components = torch.nn.parameter.Parameter(components)
    self.num_reduced_features = num_reduced_features
    self.fitted = False

  def from_sklearn(self, pca):

    if torch.cuda.is_available():
      self.mean = torch.nn.parameter.Parameter(torch.from_numpy(pca.mean_).cuda())
      self.components = torch.nn.parameter.Parameter(torch.from_numpy(pca.components_.T).cuda())
    else:
      self.mean = torch.nn.parameter.Parameter(torch.from_numpy(pca.mean_))
      self.components = torch.nn.parameter.Parameter(torch.from_numpy(pca.components_.T))

  def fit(self, features):
    sklearn_pca = SKLEARN_PCA(self.num_reduced_features)
    print("fitting PCA Module. Reducing feature dimension from {} to {}".format(features.shape[-1],
                                                                                self.num_reduced_features))
    sklearn_pca.fit(features)
    self.from_sklearn(sklearn_pca)
    self.fitted = True
    return sklearn_pca.transform(features)

  def _load_from_state_dict(self, *args, **kwargs):
    try:
      super(PCA, self)._load_from_state_dict(*args, **kwargs)
    except ValueError:
      print("Could not initialize PCA module")
    self.fitted = True

  def forward(self, features):
    if not self.fitted:
      raise RuntimeError("[ERROR] PCA module has not been fit!")

    n_feat = features.shape[-1]
    mean_features = features.reshape(-1, n_feat) - self.mean[None, :]
    out = torch.matmul(mean_features, self.components)
    return out


class _TorchGMM(nn.Module):

  def __init__(self,
               n_features,
               n_components,
               means=None,
               covariances=None,
               weights=None,
               covariance_type="tied",
               reg_covar=1e-6):
    super().__init__()
    self.init_gmm(n_features, n_components, means, covariances, weights)
    self.n_features = n_features
    self.n_components = n_components
    self.covariance_type = covariance_type
    self.reg_covar = reg_covar

  def fit(self, all_features):
    # Fit sklearn GMM to obtain parameters
    print("Fitting GMM Model, Params: num_components:{}, covariance_type: {}, reg_covar: {}".format(self.n_components,
                                                                                                    self.covariance_type,
                                                                                                    self.reg_covar))
    gmm = GaussianMixture(
      n_components=self.n_components,
      covariance_type=self.covariance_type,
      reg_covar=self.reg_covar,
    )
    gmm.fit(all_features)

    # Convert params to torch and save them
    cov = gmm.covariances_
    if self.covariance_type == 'tied':
      # covariance for each component is the same
      cov = np.tile(np.expand_dims(cov, 0), (self.n_components, 1, 1))
    elif self.covariance_type == 'diag':
      # transform from diagonal vector to matrix
      newcov = np.zeros((self.n_components, cov.shape[-1], cov.shape[-1]))
      for i in range(self.n_components):
        # np.diag only works on 1-dimensional arrays
        newcov[i] = np.diag(cov[i])
      cov = newcov
    cov = torch.as_tensor(cov)

    self.init_gmm(self.n_features, n_components=self.n_components,
                  means=torch.as_tensor(gmm.means_),
                  covariances=cov,
                  weights=torch.as_tensor(gmm.weights_))

  def init_gmm(self, n_features, n_components, means, covariances, weights):
    if weights is None:
      weights = torch.rand((n_components,))
    if covariances is None:
      covariances = torch.tile(torch.unsqueeze(torch.eye(n_features), 0),
                               (n_components, 1, 1))
    if means is None:
      means = torch.rand((n_components, n_features))

    if torch.cuda.is_available():
      means = means.cuda()
      covariances = covariances.cuda()
      weights = weights.cuda()

    self.register_buffer('means', means)
    self.register_buffer('covariances', covariances)
    self.register_buffer('weights', weights)

    mix = torch.distributions.Categorical(self.weights)
    comp = torch.distributions.MultivariateNormal(self.means, self.covariances)
    self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

  def _load_from_state_dict(self, *args, **kwargs):
    super(_TorchGMM, self)._load_from_state_dict(*args, **kwargs)
    try:
      # Ugly fix to make sure distributions can be loaded -> recreate distributions
      mix = torch.distributions.Categorical(self.weights)
      comp = torch.distributions.MultivariateNormal(self.means, self.covariances)
      self.gmm = torch.distributions.MixtureSameFamily(mix, comp)
    except ValueError as e:
      print("Could not load GMM module")

  def forward(self, x):
    return torch.unsqueeze(self.gmm.log_prob(x), 3)
