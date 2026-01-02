# 🍔 Fast Food Classification System

<div align="center">

![Fast Food Banner](https://images.unsplash.com/photo-1561758033-d89a9ad46330?w=1200&h=400&fit=crop)

*AI-Powered Fast Food Recognition using Deep Learning*

</div>

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An advanced deep learning project for classifying fast food images into 10 different categories using EfficientNetB0 architecture. The project features both a Flask web application and a professional Streamlit interface with comprehensive food information and AI visualization capabilities.

![Fast Food Classification](https://img.icons8.com/color/96/000000/hamburger.png)

---

## 🎯 Live Demo

### 🌐 **Try It Now!**

<div align="center">

[![Streamlit App](https://img.shields.io/badge/🍔_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://fast-food-cls.streamlit.app/)

**[🚀 Launch Interactive Demo](https://fast-food-cls.streamlit.app/)**

*Experience the full-featured AI-powered fast food classifier with real-time predictions, feature maps visualization, and comprehensive food encyclopedia!*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Deployment Options](#-deployment-options)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

This project builds a state-of-the-art deep learning model to accurately classify images of various fast food items into one of 10 categories. The model leverages the **EfficientNetB0** architecture, a pre-trained convolutional neural network known for its exceptional balance between accuracy and computational efficiency.

### 🍕 Supported Food Categories

The system can identify these 10 fast food items:

1. 🥔 **Baked Potato** - Classic comfort food
2. 🍔 **Burger** - American icon
3. 🍗 **Crispy Chicken** - Golden and delicious
4. 🍩 **Donut** - Sweet indulgence
5. 🍟 **Fries** - Crispy potato perfection
6. 🌭 **Hot Dog** - Stadium favorite
7. 🍕 **Pizza** - Italian masterpiece
8. 🥪 **Sandwich** - Versatile classic
9. 🌮 **Taco** - Mexican delight
10. 🌯 **Taquito** - Rolled and crispy

---

## ✨ Features

### 🤖 **Core AI Capabilities**

- **High-Accuracy Classification**: EfficientNetB0-based model with exceptional performance
- **Transfer Learning**: Fine-tuned on 15,000 fast food images
- **Real-Time Predictions**: Instant classification with confidence scores
- **Robust Data Augmentation**: Enhanced model generalization through advanced augmentation
- **Multi-Class Support**: Accurately distinguishes between 10 food categories

### 🖥️ **Flask Web Application**

- **Simple Interface**: Clean, minimalist design for quick classifications
- **Image Upload**: Direct file upload functionality
- **Instant Results**: Real-time classification with confidence percentages
- **Lightweight**: Fast loading and responsive performance
- **Static Serving**: Efficient image handling and display

### 🎨 **Streamlit Professional Dashboard**

#### **Multi-Page Layout**
- 🏠 **Home - Classify Food**: Main classification interface with advanced options
- 📚 **Food Encyclopedia**: Comprehensive database of all 10 food categories
- ℹ️ **About**: Project information, technical details, and developer info

#### **Rich Classification Features**
- **Drag & Drop Upload**: Intuitive image upload interface
- **Real-Time Analysis**: Instant AI-powered classification
- **Confidence Visualization**: Color-coded confidence levels (High/Medium/Low)
- **All Predictions Chart**: Interactive Plotly visualization showing confidence for all 10 categories
- **Feature Maps Display**: Visualize internal neural network activations at different layers
- **Detailed Results Card**: Beautiful presentation of classification results

#### **Educational Content** (For Each Food)
- 📝 **Detailed Descriptions**: Comprehensive information about each food type
- 🍴 **Ingredients List**: Common ingredients and components
- 🔥 **Fun Facts**: Fascinating trivia and historical tidbits
- 🌍 **Origin & History**: Cultural background and development
- 💪 **Nutritional Information**: Calorie counts and nutritional highlights
- 🎭 **Popular Variations**: Regional styles and creative interpretations

#### **Advanced Visualization**
- **Interactive Charts**: Plotly-powered confidence score visualizations
- **Feature Map Analysis**: See what the neural network "sees" at different layers
- **Confidence Color Coding**: Green (high), Yellow (medium), Red (low)
- **Responsive Design**: Works seamlessly on desktop and mobile devices

#### **User Experience**
- **Professional Theme**: Clean red/yellow/orange color scheme matching fast food branding
- **Smooth Animations**: Polished transitions and hover effects
- **Food Emojis**: Visual appeal with category-specific emojis
- **Visitor Counter**: Real-time tracking displayed in bottom-left corner
- **Intuitive Navigation**: Easy-to-use sidebar with clear page organization

---

## 📊 Dataset

### Dataset Specifications

- **Total Images**: 15,000 high-quality food images
- **Resolution**: 224×224 pixels (optimized for EfficientNetB0)
- **Classes**: 10 evenly distributed categories
- **Balance**: Well-balanced across all classes for effective learning
- **Format**: RGB color images

### Data Preprocessing

- **Normalization**: Pixel values scaled to [0, 1] range
- **Resizing**: Standardized to 224×224 pixels
- **Color Mode**: RGB (3 channels)

### Data Augmentation Techniques

To improve model robustness and prevent overfitting, the following augmentation techniques were applied during training:

- **Rotation**: Random rotations up to 20 degrees
- **Width/Height Shifts**: Random shifts up to 20%
- **Shear Transformations**: Shear mapping up to 20%
- **Zoom**: Random zoom up to 20%
- **Horizontal Flip**: Random horizontal flipping
- **Fill Mode**: Nearest neighbor filling for transformed regions

These augmentations significantly enhance the diversity of the training data, enabling the model to generalize better to unseen images.

---

## 🏗️ Model Architecture

### EfficientNetB0 Foundation

The model is built on **EfficientNetB0**, a state-of-the-art convolutional neural network architecture from the EfficientNet family. EfficientNetB0 was selected for this project due to:

- **Compound Scaling**: Balanced scaling of depth, width, and resolution
- **High Efficiency**: Achieves top accuracy with fewer parameters
- **Pre-trained Weights**: Leverages ImageNet pre-training for faster convergence
- **Computational Efficiency**: Optimal balance between accuracy and speed

### Architecture Details

```
Input Layer (224×224×3)
    ↓
EfficientNetB0 Base (Pre-trained on ImageNet)
    ↓ (Most layers frozen)
Last 20 Layers (Fine-tuned)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (10 units, Softmax)
```

### Fine-Tuning Strategy

- **Transfer Learning**: Leveraged pre-trained ImageNet weights
- **Layer Freezing**: Froze early layers to retain low-level feature extraction
- **Fine-Tuning**: Retrained last 20 layers for task-specific adaptation
- **Custom Head**: Added custom classification layers for 10-class output

### Model Performance

- **Training Accuracy**: Achieved high accuracy on training set
- **Validation Accuracy**: Strong generalization to validation data
- **Model Size**: ~17MB (H5 format)
- **Inference Speed**: Real-time classification (< 1 second per image)

---

## 🚀 Deployment Options

This project includes **two deployment options** to suit different use cases:

### 1. Flask Application (Simple & Fast)

**Best for:**
- Quick deployments
- Minimal interface requirements
- Integration into existing Flask projects
- REST API implementation

**Features:**
- Lightweight web interface
- Simple image upload and classification
- Confidence score display
- Fast response times

**Run Flask App:**
```bash
python main.py
```
Access at: `http://localhost:5000`

### 2. Streamlit Dashboard (Professional & Feature-Rich)

**Best for:**
- Professional presentations
- Educational demonstrations
- Comprehensive user experience
- Data visualization and analysis

**Features:**
- Multi-page professional interface
- Interactive visualizations with Plotly
- Feature map analysis
- Comprehensive food encyclopedia
- Real-time visitor tracking
- Educational content for each food category

**Run Streamlit App:**
```bash
streamlit run streamlit_app.py
```
Access at: `http://localhost:8501`

---

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- 50MB free disk space (for model and dependencies)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mo-Abdalkader/fastfood-classifier.git
   cd fastfood-classifier
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   
   For Flask application:
   ```bash
   pip install flask tensorflow pillow numpy
   ```
   
   For Streamlit application:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   - Ensure `model.h5` is in the root directory
   - Model file should be ~17MB

5. **Run your preferred application**
   
   Flask:
   ```bash
   python main.py
   ```
   
   Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📱 Usage

### Flask Application Usage

1. Start the Flask server: `python main.py`
2. Navigate to `http://localhost:5000`
3. Click "Choose File" and select a fast food image
4. Click "Upload" to classify
5. View the predicted category and confidence score

### Streamlit Application Usage

#### **Classify Food (Home Page)**

1. Navigate to **"🏠 Home - Classify Food"**
2. Upload a food image (JPEG or PNG)
3. Configure analysis options:
   - ✅ **Show Feature Maps**: Visualize neural network layers
   - ✅ **Show All Predictions**: See confidence for all 10 categories
   - ✅ **Show Detailed Food Info**: Get comprehensive information
4. Click **"🔍 Classify Food"**
5. Explore results:
   - Prediction with confidence score
   - Detailed food information (ingredients, fun facts, nutrition)
   - Interactive confidence chart
   - Feature maps (if enabled)

#### **Browse Food Encyclopedia**

1. Navigate to **"📚 Food Encyclopedia"**
2. Select any food category from dropdown
3. Explore comprehensive information:
   - Description and overview
   - Ingredients list
   - Fun historical facts
   - Origin and cultural background
   - Nutritional information
   - Popular variations

#### **Learn About the Project**

1. Navigate to **"ℹ️ About"**
2. Discover:
   - How the AI works
   - Technical specifications
   - Developer information
   - Project goals and vision

### 🎯 Tips for Best Results

- **Image Quality**: Use clear, well-lit photographs
- **Focus**: Ensure the food item is the main subject
- **Angle**: Front or top-down views work best
- **Background**: Simple backgrounds improve accuracy
- **Resolution**: Higher resolution images (but under 200MB)
- **Format**: JPEG or PNG formats supported

---

## 🔧 Technical Details

### Project Structure

```
fastfood-classifier/
├── main.py                    # Flask application
├── streamlit_app.py          # Streamlit application
├── model.h5                   # Trained EfficientNetB0 model
├── requirements.txt           # Streamlit dependencies
├── visitor_stats.json         # Visitor tracking (auto-generated)
├── static/                    # Flask static files
│   ├── uploaded images       # Temporary uploads
│   └── css/styles.css        # Flask styling
├── templates/                 # Flask HTML templates
│   └── index.html            # Flask main page
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

### Technology Stack

#### **Machine Learning**
- **Framework**: TensorFlow 2.x / Keras
- **Architecture**: EfficientNetB0 (pre-trained on ImageNet)
- **Training**: Transfer learning with fine-tuning
- **Optimization**: Adam optimizer
- **Loss Function**: Categorical crossentropy

#### **Web Frameworks**
- **Flask 2.0**: Lightweight WSGI web application framework
- **Streamlit 1.31+**: Modern web framework for ML applications

#### **Data Processing**
- **NumPy**: Numerical computing and array operations
- **Pillow (PIL)**: Image loading and preprocessing

#### **Visualization**
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Feature map visualization

#### **Development**
- **Python 3.11**: Programming language
- **JSON**: Data storage and configuration

### Model Training Details

- **Base Model**: EfficientNetB0 (ImageNet weights)
- **Fine-Tuning**: Last 20 layers retrained
- **Frozen Layers**: Early layers preserved for low-level features
- **Input Shape**: 224×224×3 (RGB)
- **Output Classes**: 10 (Softmax activation)
- **Batch Size**: 32
- **Epochs**: Multiple epochs with early stopping
- **Validation Split**: 20% of training data

---

## 🔮 Future Work

### Planned Enhancements

1. **Enhanced Data Augmentation**
   - Explore advanced techniques like CutMix and MixUp
   - Implement AutoAugment for automated policy search
   - Add color jittering and contrast adjustments

2. **Model Optimization**
   - Experiment with EfficientNetB3-B7 variants
   - Implement ensemble methods for improved accuracy
   - Optimize for mobile deployment with TensorFlow Lite
   - Reduce model size through quantization

3. **Extended Classification**
   - Expand to 20+ food categories
   - Add nutritional value estimation
   - Implement multi-label classification (ingredients detection)
   - Region-specific food variations

4. **Advanced Features**
   - Real-time video classification
   - Batch processing capabilities
   - API endpoint for third-party integration
   - Mobile application development (iOS/Android)

5. **User Experience**
   - Add user authentication and profiles
   - Implement classification history tracking
   - Create food recommendation system
   - Add social sharing capabilities

6. **Analysis Tools**
   - Confusion matrix visualization
   - Model performance dashboard
   - A/B testing framework
   - User feedback integration

7. **Production Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline setup
   - Cloud deployment (AWS/GCP/Azure)

---

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Here's how you can contribute:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
6. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation for any changes
- Ensure all tests pass before submitting PR
- Write clear, descriptive commit messages

### Areas for Contribution

- 🐛 Bug fixes and error handling
- ✨ New features and enhancements
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Testing and quality assurance
- 🌐 Internationalization and localization

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

### MIT License Summary

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❗ License and copyright notice required
- ❌ Liability and warranty not provided

---

## 📞 Contact

**Mohamed Abdalkader**

I'm a passionate developer focused on AI, machine learning, and creating intuitive user experiences. Feel free to reach out for questions, collaborations, or just to connect!

- 📧 **Email**: [Mohameed.Abdalkadeer@gmail.com](mailto:Mohameed.Abdalkadeer@gmail.com)
- 💼 **LinkedIn**: [mo-abdalkader](https://www.linkedin.com/in/mo-abdalkader/)
- 💻 **GitHub**: [Mo-Abdalkader](https://github.com/Mo-Abdalkader)

### Get in Touch

- 💬 Questions about the project? Open an issue!
- 🤝 Interested in collaboration? Send me a message!
- 🐛 Found a bug? Report it on GitHub!
- 💡 Have a feature idea? I'd love to hear it!

---

## 🙏 Acknowledgments

### Special Thanks

- **TensorFlow/Keras Team**: For the excellent deep learning framework and pre-trained EfficientNetB0 model
- **Streamlit Team**: For creating an amazing framework for ML applications
- **Flask Community**: For the lightweight and flexible web framework
- **Open Source Community**: For countless resources, tutorials, and inspiration
- **ImageNet**: For providing the foundation dataset for transfer learning
- **Food Enthusiasts**: Who make projects like this meaningful and fun! 🍔🍕🌮

### Resources Used

- EfficientNet paper: ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946)
- TensorFlow documentation and tutorials
- Streamlit documentation and community examples
- Stack Overflow community for troubleshooting

---

## ⚠️ Disclaimer

This application is designed for **educational and entertainment purposes only**. 

- **Nutritional Information**: All nutritional data provided is approximate and should not be used as a substitute for professional dietary advice
- **Classification Accuracy**: While the model achieves high accuracy, results may vary based on image quality, lighting, and angle
- **Medical/Health Advice**: This tool does not provide medical or health advice. Consult healthcare professionals for dietary recommendations
- **Food Safety**: The app does not assess food safety, freshness, or quality

---

## 📈 Project Stats

- **Model Accuracy**: High performance on validation set
- **Parameters**: ~4M trainable parameters
- **Training Dataset**: 15,000 images
- **Classes**: 10 food categories
- **Inference Time**: < 1 second per image
- **Supported Formats**: JPEG, PNG
- **Max Upload Size**: 200MB

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub! It helps others discover the project and motivates continued development.

---

**Made with ❤️ and 🍕 by Mohamed Abdalkader**

*Enjoy classifying your fast food! 🍔🎉*

