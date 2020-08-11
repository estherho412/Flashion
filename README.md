# Flashion

“Flashion - a flash in fashion, accelerates user experience in searching the most favorable fashion item by just a click."
Fashon attributes classification models are built on Convolutional Netural Networks utilizing VGG19 transfer learning.
Based on the predicted attributes classification, our AI-recommender will suggest shopping websites with similar fashion items.


Data Collection

Above 20K images with the following clothign attributes are scraped from several online shopping websites and Google
Category - blazers, dresses, hoodies & sweatshirts, jacket & coats, jeans, knitwear & cardigans, pants, shorts, skirts, tops
Pattern - checks, dots, floral, graphic, solid, stripes
Color - black & white, blue, brown, green, light pink & beige, orange, purple, red & pink, yellow

Image Augmentation

The technique to create variations of the images by translating, rotating, or flipping the pixels of the image.
This is to expand the training dataset in order to improve the performance and ability of the model to take advantage of the capabilities of bigger data.

Model Construction (Benchmarking Model)

We adopted transfer learning for building the model. In particular, we use the state of the art model-VGG19 which was trained on more than a million images from the ImageNet database in the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC).At the first attempt, we freeze the feature selection layers on VGG19 for the training.

Model Tuning

We tuned the weights for a few more layers on the pre-trained model ,adjusted the learning rate of the ADAM optimizer, added a few more layers in the fully-connected classifier. The accuracy score improved few to ten points comparing to the benchmarking model, obtaining about 60% for each fashion attributes.

Flask App Development and deployment
Our flask application allows user to upload the fashion photo they like and return suggested shopping websites with similar fashion items.
If you are interested in the flask app, please leave a message to us.



