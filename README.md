# Fast Food Classification

This project involves building a deep learning model to classify fast food images into 10 different categories. The model utilizes the EfficientNetB0 architecture, a pre-trained model known for its efficiency and high performance in image classification tasks.

## Project Overview

The goal of this project is to accurately classify images of various fast food items into one of the following 10 categories:
- **Baked Potato**
- **Burger**
- **Crispy Chicken**
- **Donut**
- **Fries**
- **Hot Dog**
- **Pizza**
- **Sandwich**
- **Taco**
- **Taquito**

### Dataset

- **Size:** 15,000 images
- **Resolution:** 224x224 pixels
- **Classes:** 10

The dataset is well-balanced across the 10 classes, ensuring that the model can learn effectively from a diverse set of images.

### Data Augmentation

To improve model robustness, various data augmentation techniques were applied during training. These include rotations, shifts, shears, zooms, and horizontal flips. Augmentation helps enhance the diversity of the training data and prevents overfitting.

### Model Architecture

The model is based on **EfficientNetB0**, a convolutional neural network architecture that is pre-trained on the ImageNet dataset. EfficientNetB0 is selected for this project due to its efficiency in both accuracy and computational cost.

### Fine-Tuning

The pre-trained EfficientNetB0 model was fine-tuned for this project. Most of the model’s layers were frozen to retain learned features, and only the last 20 layers were retrained to adapt the pre-trained weights to the specific classes in this project.

### Future Work

- **Enhanced Data Augmentation:** Explore advanced data augmentation techniques to further increase the diversity of the training dataset, which may improve the model's generalization.
- **Model Optimization:** Investigate other model architectures and hyperparameter tuning to enhance performance, potentially improving accuracy and reducing overfitting.
- **Deployment:** Develop and deploy the model as a web service or application for real-time fast food classification, making it accessible for practical use cases and demonstrations.
- **Transfer Learning:** Experiment with different pre-trained models and transfer learning strategies to see if they offer better performance or efficiency.
- **User Interface:** Create an intuitive user interface for easier interaction with the model, allowing users to upload images and receive classification results seamlessly.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- The pre-trained EfficientNetB0 model was provided by TensorFlow/Keras.
- Thanks to the open-source community for their invaluable resources and contributions.

### Contact
For any questions, issues, or collaboration opportunities, feel free to reach out via [email](mailto:Mohameed.Abdalkadeer@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/mo-abdalkader/).
