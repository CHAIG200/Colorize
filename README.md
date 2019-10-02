# Why this project.
This project was used to be a proof of concept and also to answer the research question that i had selected for my honours project. This also allowedme to further my knowledge in the area of machine learning also.
```
“What deep neural network model design is most effective for colorizing a greyscale image back into color automatically ”
```
# What type of neural network was used
Due to the nature of the project being focused around imagery it was decided that a convolutional neural network also known as CNN was to be used as it offered more features and support in this type of project.

# Accuracy
When it came to the accuracy of the model that i had generated it suprised me with the accuracy that it had after just a short training period and with the limited dataset that i had access to.
```
Training Time: 26 Hours - Due to CPU limitation
```
```
Dataset Size: 13,000 Images - 6,000 for training and 6,000 for testing
```
## First Test
For the first test i used an image of a White male with this test i aimed to test if the network could correctly colorize the correct skin tone. In this test i received a **74.15% Accuracy** this was lower than expected but this was also due to the background not being colorized as the network was only trained on faces.
<img src="https://i.imgur.com/0fFCecJ.png"></img>

## Second Test
For the second test i used an image of an african american male with this test i aimed to test if the network could correctly colorize the correct skin tone. In this test i received a **90.19% Accuracy** this was higher than expected but this was due to the person taking up more of the image than the background resulting in a higher accuracy.
<img src="https://i.imgur.com/i5C055D.png"></img>

## Third Test
For the third test i used an image of a White male and a african american male with this test i aimed to test if the network could correctly colorize the correct skin tone of both people . In this test i received a **83.83% Accuracy** this was around what i was expecting as what i learned form the previous two tests that the background would lower the accuracy of the overall image.
<img src="https://i.imgur.com/WpowJtO.png"></img>

## Overall
**This project was able to correctly identify and colorize image of people and apply the correct skin tone and color to the image**
