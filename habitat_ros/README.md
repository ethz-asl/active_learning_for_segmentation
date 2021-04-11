# Habitat ROS
Wrapper that connects the habitat simulator with ROS. 
It allows to specify velocity inputs for a single agent using the ´/cmd_vel´ topic.
Additionally, sensor measurement such as RGB, Depth and Semantic Images are published to ros topics.

## Adding a new Sensor
Assuming a sensor was already added to the habitat environment e.g. using the config file. 
Measurements of this sensor can be published as a specific topic by defining two functions that are decorated with the @sensorCallback and @rosPublisherCreator decorators.
The sensorCallback function gets called with the observation that is stored inside the 

### Example of publishing RGB Sensor Measurements:
Register callback functions annotated with decorator
**sensor_callbacks.py**
```python3
from habitat_ros.decorators import sensorCallback, rosPublisherCreator
from sensor_msgs.msg import Image
import numpy as np
import rospy

@sensorCallback
def rgbCallback(img):
  """ 
  This callback converts a numpy RGB image (img) to a ros message that can be published
  Args:
      img: Numpy array
  """
  rgb_msg = Image()
  rgb_msg.height = img.shape[0]
  rgb_msg.width = img.shape[1]
  rgb_msg.data = img.astype(np.uint8).flatten().tolist()
  rgb_msg.encoding = "rgb8"
  return rgb_msg

@rosPublisherCreator
def ImagePublisher(topic):
  """ 
  Creates a rospy publisher that can be used to publish a certain topic
  Args:
      topic: name for the topic
  """
  
  return rospy.Publisher(
    topic, Image, queue_size=1
  )
```

**config.yaml**
```yaml
sensors:
  frequency: 10 # Frequency of sensor measurements
  availableSensors:
    - name: "rgb"
      callback: "rgbCallback" # Must match the callback function annotated with @sensorCallback
      publisher: "ImagePublisher" # Must match the publisher function annotated with @rosPublisherCreator
      topic: "~rgb"
```

