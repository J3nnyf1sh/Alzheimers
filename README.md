Uses Tensorflow/keras and python.

current requirements:
- TensorFlow / Keras
- NumPy

Processes one image (MRI scan of a brain) and predicts the classification of the tumour, if any.
The 4 Classifications are:
- (0) Glioma
- (1) Menigioma
- (2) Pituitary
- (4) No Tumour

The dataset used is: https://www.kaggle.com/datasets/lakshancooray23/tumour-classification-images/data
- The number of image files is very large, the number of images used is limited in the code
- When handling thousands of images the code may fail to run, so if this limits are changed this must be kept in mind

Planning on further improvements such as:
- User interaction
-- Input an image
-- Implement GUI
- Using MatPlotlib to visualise model performance and visualisation
