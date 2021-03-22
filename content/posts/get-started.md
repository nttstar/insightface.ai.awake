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

```
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

```
url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)
```

##### Init FaceAnalysis module by its default models:

```
model = insightface.app.FaceAnalysis()
```

##### Here we use CPU to do all the job. Please change ctx-id to a non-negative number if you have GPUs

```
model.prepare(ctx_id = -1, nms=0.4)
```

##### Do image inference:

```
faces = model.get(img)
```

##### Show inference results:

```
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

# ArcFace Face Recognition Example

##### Firstly, we defined some tool methods:

```
def l2norm(feat):
    return np.sqrt(np.sum(feat*feat)+0.000001)

def feature_similarity(feat1, feat2):
    norm_feat1 = feat1 / l2norm(feat1)
    norm_feat2 = feat2 / l2norm(feat2)
    return np.dot(norm_feat1, norm_feat2)
```

##### Get an online pre-trained ArcFace model by its name:

```
model = insightface.model_zoo.get_model('arcface_r100_v1')
model.prepare(ctx_id = -1)
```

##### Then, read some aligned images for comparison:

```
img1 = cv2.imread('./aligned_1.jpg') #the path is fake
img2 = cv2.imread('./aligned_2.jpg') #the path is fake
```

##### Do face recognition, get the feature embedding vectors.

```
feat1 = model.get_embedding(img1)
feat2 = model.get_embedding(img2)
```

##### Get the similarity of two faces:

```
sim = feature_similarity(feat1, feat2)
print("The similarity is %.5f"%sim)
```

# RetinaFace Face Detection Example

##### Get an online pre-trained RetinaFace model by its name:

```
model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id = -1)
```

##### Do face detection, get the bounding box and key-points:

```
bboxes, landmarks = model.detect(img, threshold=0.5, scale=1.0)
```