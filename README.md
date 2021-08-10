# Dog-Years-Age-Classification

## Introduction
Machine learning algorithms have made great progress in identifying age progression in humans, but for dogs little work has been done. I came across the
[DogAge Dataset](https://www.researchgate.net/publication/335699398_Automatic_Estimation_of_Dog_Age_The_DogAge_Dataset_and_Challenge), a rarely attempted multi-class categorical dataset containing images of dogs broken into 3 categories: Adult, Young, and Senior. 

### Use Case
This classification of dogs based on age could potentially be implemented by pet adoption sites and agencies to help categorize or correctly recategorize dogs based on their ages.

## Dataset
The original dataset contained two separate dataset each categorized into 3 classes: Young (0 −2 years), Adult (2 −5 years), and Senior (>6 years).

## EDA
![Images per Class](img/Images_per_Category2.png)

![Example Dogs](img/Example_Dogs2.png)

![Mean Dogs](img/Mean_Dog.png)
![Mode Dogs](img/Mode_Dog.png)

## Modeling


## Application
To display the results of the trained models, I created a small GUI application using the Tkinter python library.

The app allows the user to input an image file of a dog, the app the displays the image and outputs the predicted age and breed of the dog.
