---
title: Get Started
subtitle: Get Started with InsightFace
category:
  - Docs
author: Jia Guo
date: 2021-03-21T10:21:30.428Z
featureImage: /uploads/836946.jpeg
---
# Installation

Install insightface python package with following command:

`pip install insightface`

Please also install the correct MXNet package according to your system configuration.

This python package is used to call our pre-trained face models in an easy way.

# A Quick Example

##### Firstly, we defined some tool methods:

```python
import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
```

##### Then, we download and show the example image:

```python
url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)
```

##### Init FaceAnalysis module by its default models:
```python
model = insightface.app.FaceAnalysis()

```

##### Here we use CPU to do all the job. Please change ctx-id to a non-negative number if you have GPUs

```python
model.prepare(ctx_id = -1, nms=0.4)
```

##### Do image inference:
```python
faces = model.get(img)

```

##### Show inference results:
```python
for idx, face in enumerate(faces):
    print("Face [%d]:"%idx)
    print("\tage:%d"%(face.age))
    gender = 'Male'
    if face.gender==0:
        gender = 'Female'
      print("\tgender:%s"%(gender))
      print("\tembedding shape:%s"%face.embedding.shape)
      print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
      print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
      print("")
```
