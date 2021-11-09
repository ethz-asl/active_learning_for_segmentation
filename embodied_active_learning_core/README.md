# Embodied Active Learning Core

Contains core functionalities of embodied active learning, many also used in the volumetric labeling package

### Installation
Install PIP package
```
cd embodied_active_learning_core
pip install -e .
```

### Adding a new Model:
1. (optional) Add your model (e.g. deeplab3.py) to the ```semseg/models folder```
2. In ```config/config.py``` add a distinct name for your model (e.g. ```NETWORK_CONFIG_ONLINE_DEEPLAB = "online-deeplab"```)
3. In ```semseg/model_manager.py``` import your new name and add it to the list of online networks ```ONLINE_LEARNING_NETWORKS``` (if it is an online one)
4. Update the function ```get_model_for_config``` to return your model. 
5. To use clustering based uncertatiny, also update ``get_uncertainty_net``. Node that you probabily only have to change the name of the name of the feature layer in the ucnertainty fitter:
    ```python
    model_with_uncertainty = UncertaintyModel(model, feature_layer, n_feature_for_uncertainty='your_feature_layer_name',
                                                  n_components=network_config.classes if n_components is None else n_components,
                                                  covariance_type=covariance_type, reg_covar=reg_covar)
    ```
6. For online learning, also add your model to the function ```get_optimizer_params_for_model```


### Adding a new Sampling Strategy for the trainings buffer

1. Edit ```online_learning/replay_buffer.py``` and add new enums for BufferUpdatingMethod and/or BufferSamplingMethod. 
    ```python
    # Example for sampling only recent images
    class BufferUpdatingMethod(Enum):
      SAMPLE_LAST_N = 3
      @staticmethod
      def from_string(name: str):
        if name == "last_n":
          return BufferUpdatingMethod.SAMPLE_LAST_N
    ```
2. Update ```draw_samples``` for sampling strategy and ```add_sample``` for updating strategies
    ```python
    def add_sample(self, sample: TrainSample):
    if self.__len__() >= self.max_buffer_length:
      print(f"Labeled buffer getting too large. Max Length: {self.max_buffer_length}. Going to downsample it")
      # Resample buffer
      if self.replacement_strategy == BufferUpdatingMethod.SAMPLE_LAST_N:
         self.entries = self.entries[(-self.max_buffer_length // )2:]
     ```

### Add a new uncertainty estimation method
1. Edit ```uncertainty/uncertainty_estimator.py```
    ```python
    class MyUncertaintyEstimator(UncertaintyEstimator):
      def __init__(self, model):
        self.model = model
    
      def predict(self, image, gt_image):
        prediction = self.model(image)
        sem_seg = np.argmax(prediction, axis=-1).astype(np.uint8)
        uncertainty = #uncertainty magic goes here
    
        return sem_seg, uncertainty
    ```
2. Add your uncertainty estimator to ```get_uncertainty_estimator_for_network```.