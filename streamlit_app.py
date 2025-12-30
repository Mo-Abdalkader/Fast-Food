import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Fast Food Classification",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Main theme colors - Professional & Clean */
    :root {
        --primary: #FF6B6B;
        --secondary: #FFD93D;
        --accent: #FFA502;
        --success: #6BCF7F;
        --dark: #2C3E50;
        --light: #F8F9FA;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FF6B6B, #FFA502);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Food info cards */
    .food-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #FF6B6B;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .food-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .food-card h3 {
        color: #FF6B6B;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    .food-card p {
        color: #2C3E50;
        line-height: 1.8;
        margin-bottom: 0.8rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #FFF5F5, #FFF9E6);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px solid #FF6B6B;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
        text-align: center;
        margin: 2rem 0;
    }
    
    .prediction-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
    }
    
    .prediction-label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF6B6B;
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #6BCF7F;
        font-weight: 700;
    }
    
    .confidence-medium {
        color: #FFA502;
        font-weight: 700;
    }
    
    .confidence-low {
        color: #FF6B6B;
        font-weight: 700;
    }
    
    /* Info sections */
    .info-section {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #FFD93D;
    }
    
    .info-section h4 {
        color: #FF6B6B;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
    }
    
    .info-section ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .info-section li {
        padding: 0.4rem 0;
        color: #2C3E50;
    }
    
    .info-section li:before {
        content: "🍴 ";
        margin-right: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #F0F0F0;
        color: #6c757d;
    }
    
    .footer a {
        color: #FF6B6B;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #FFA502;
        text-decoration: underline;
    }
    
    /* Visitor counter */
    .visitor-counter {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: white;
        padding: 10px 20px;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
        color: #2C3E50;
        z-index: 1000;
        border: 2px solid #FFD93D;
    }
    
    .visitor-counter i {
        color: #FF6B6B;
        margin-right: 5px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B, #FFA502);
        color: white;
        border-radius: 50px;
        padding: 0.6rem 2.5rem;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #FFF9E6;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #FFA502);
    }
</style>
""", unsafe_allow_html=True)

# ========== Constants ==========
CLASS_LABELS = {
    0: 'Baked Potato',
    1: 'Burger',
    2: 'Crispy Chicken',
    3: 'Donut',
    4: 'Fries',
    5: 'Hot Dog',
    6: 'Pizza',
    7: 'Sandwich',
    8: 'Taco',
    9: 'Taquito'
}

FOOD_EMOJIS = {
    'Baked Potato': '🥔',
    'Burger': '🍔',
    'Crispy Chicken': '🍗',
    'Donut': '🍩',
    'Fries': '🍟',
    'Hot Dog': '🌭',
    'Pizza': '🍕',
    'Sandwich': '🥪',
    'Taco': '🌮',
    'Taquito': '🌯'
}

FOOD_INFO = {
    'Baked Potato': {
        'description': 'A classic comfort food made by baking a whole potato until the skin is crispy and the inside is soft and fluffy.',
        'ingredients': ['Russet or Idaho potatoes', 'Butter', 'Sour cream', 'Cheese', 'Bacon bits', 'Chives', 'Salt & pepper'],
        'fun_fact': '🚀 A baked potato was the first food eaten in space by a human! Soviet cosmonaut Yuri Gagarin enjoyed one during his historic 1961 flight.',
        'origin': '🌍 Peru and Bolivia - Potatoes have been cultivated in South America for over 7,000 years!',
        'nutrition': '💪 High in potassium (more than a banana!), vitamin C, and fiber. A medium potato has about 160 calories.',
        'variations': ['Loaded baked potato', 'Twice-baked potato', 'Sweet potato', 'Hasselback potato']
    },
    'Burger': {
        'description': 'An iconic American sandwich featuring a ground meat patty (usually beef) served in a sliced bun with various toppings.',
        'ingredients': ['Ground beef patty', 'Burger bun', 'Lettuce', 'Tomato', 'Onion', 'Pickles', 'Cheese', 'Ketchup', 'Mustard', 'Mayo'],
        'fun_fact': '🍔 Americans eat approximately 50 billion burgers per year - that\'s about 3 burgers per week per person!',
        'origin': '🌍 While the exact origin is debated, the modern burger is believed to have been created in Hamburg, Germany, then popularized in the USA in the early 1900s.',
        'nutrition': '💪 Excellent source of protein, iron, and B vitamins. A typical burger ranges from 250-800 calories depending on size and toppings.',
        'variations': ['Cheeseburger', 'Bacon burger', 'Veggie burger', 'Double burger', 'Slider']
    },
    'Crispy Chicken': {
        'description': 'Juicy chicken pieces coated in a seasoned, crispy breading and deep-fried to golden perfection.',
        'ingredients': ['Chicken pieces', 'Flour', 'Eggs', 'Breadcrumbs or batter', 'Spices (paprika, garlic, pepper)', 'Oil for frying'],
        'fun_fact': '🍗 The pressure fryer used for making crispy fried chicken was invented by Colonel Sanders in 1939, revolutionizing fast food!',
        'origin': '🌍 Scottish immigrants brought frying techniques to the American South, where African Americans added their own spice blends, creating the modern fried chicken.',
        'nutrition': '💪 High in protein with about 250-400 calories per piece. The skin contains most of the fat and calories.',
        'variations': ['Nashville hot chicken', 'Korean fried chicken', 'Chicken tenders', 'Popcorn chicken', 'Buffalo wings']
    },
    'Donut': {
        'description': 'A sweet, deep-fried dough confection, typically ring-shaped, glazed or topped with icing, sprinkles, or filled with cream or jam.',
        'ingredients': ['Flour', 'Sugar', 'Eggs', 'Milk', 'Butter', 'Yeast or baking powder', 'Glaze or icing', 'Various toppings'],
        'fun_fact': '🍩 National Donut Day (first Friday of June) was created in 1938 to honor Salvation Army volunteers who served donuts to soldiers in WWI!',
        'origin': '🌍 Dutch settlers brought "olykoeks" (oil cakes) to America in the 1600s. The hole in the middle was added in the mid-1800s for even cooking.',
        'nutrition': '💪 A typical glazed donut has 250-300 calories. Best enjoyed as an occasional treat!',
        'variations': ['Glazed donut', 'Jelly-filled', 'Cream-filled', 'Old-fashioned', 'Chocolate frosted', 'Crullers']
    },
    'Fries': {
        'description': 'Crispy, golden strips of deep-fried potatoes, seasoned with salt and often served with various dipping sauces.',
        'ingredients': ['Potatoes (Russet or Idaho)', 'Oil for frying', 'Salt', 'Optional seasonings'],
        'fun_fact': '🍟 Despite the name "French fries," they likely originated in Belgium in the 1600s! Belgians get quite defensive about this.',
        'origin': '🌍 Belgium claims invention in the 1600s, but the term "French fries" comes from the "frenching" cutting technique, not France.',
        'nutrition': '💪 A medium serving has about 365 calories and provides vitamin C and potassium. Baked fries are a healthier alternative!',
        'variations': ['Curly fries', 'Waffle fries', 'Sweet potato fries', 'Steak fries', 'Shoestring fries', 'Poutine']
    },
    'Hot Dog': {
        'description': 'A grilled or steamed sausage served in a sliced bun, traditionally topped with mustard, ketchup, onions, relish, and more.',
        'ingredients': ['Sausage/frankfurter', 'Hot dog bun', 'Mustard', 'Ketchup', 'Onions', 'Relish', 'Sauerkraut', 'Cheese'],
        'fun_fact': '🌭 Americans consume about 20 billion hot dogs per year - that\'s about 70 hot dogs per person! Peak consumption is between Memorial Day and Labor Day.',
        'origin': '🌍 Frankfurt, Germany claims to have invented the frankfurter in 1487! It came to America with German immigrants in the 1800s.',
        'nutrition': '💪 A typical hot dog has 150-200 calories. Choose all-beef or turkey dogs for better quality protein.',
        'variations': ['Chili dog', 'Chicago-style hot dog', 'New York hot dog', 'Corn dog', 'Cheese dog']
    },
    'Pizza': {
        'description': 'A savory dish consisting of a round, flattened base of dough topped with tomato sauce, cheese, and various toppings, baked in an oven.',
        'ingredients': ['Pizza dough', 'Tomato sauce', 'Mozzarella cheese', 'Various toppings (pepperoni, vegetables, meats)', 'Olive oil', 'Herbs'],
        'fun_fact': '🍕 Pizza is a $145 billion global industry! 93% of Americans eat pizza at least once a month. October is National Pizza Month!',
        'origin': '🌍 Naples, Italy in the 18th century. The Margherita pizza was created in 1889 to honor Queen Margherita, using tomato, mozzarella, and basil (Italian flag colors).',
        'nutrition': '💪 A slice has 250-350 calories. Thin crust with vegetable toppings is healthier. Pizza provides calcium, protein, and lycopene from tomatoes.',
        'variations': ['Neapolitan', 'New York-style', 'Chicago deep-dish', 'Sicilian', 'Detroit-style', 'California-style']
    },
    'Sandwich': {
        'description': 'Two or more slices of bread with various fillings between them, one of the world\'s most versatile and convenient meals.',
        'ingredients': ['Bread (white, wheat, sourdough)', 'Protein (deli meat, chicken, tuna)', 'Cheese', 'Vegetables', 'Condiments', 'Lettuce & tomato'],
        'fun_fact': '🥪 The sandwich is named after John Montagu, the 4th Earl of Sandwich, who in 1762 asked for meat between bread so he could eat without leaving the gaming table!',
        'origin': '🌍 While the concept is ancient, the modern sandwich is credited to 18th-century England. Now it\'s a global phenomenon!',
        'nutrition': '💪 Calories vary widely (300-700+) depending on ingredients. Choose whole grain bread and lean proteins for a healthier option.',
        'variations': ['Club sandwich', 'BLT', 'Reuben', 'Philly cheesesteak', 'Submarine', 'Panini']
    },
    'Taco': {
        'description': 'A traditional Mexican dish consisting of a folded or rolled tortilla filled with meat, beans, cheese, vegetables, and salsa.',
        'ingredients': ['Corn or flour tortilla', 'Seasoned meat (beef, chicken, pork)', 'Lettuce', 'Tomato', 'Cheese', 'Sour cream', 'Salsa', 'Guacamole'],
        'fun_fact': '🌮 Taco Tuesday is trademarked! Taco John\'s owns the trademark in 49 states. Americans consume over 4.5 billion tacos annually!',
        'origin': '🌍 Mexico - Tacos date back to 18th-century Mexican silver mines. The first taco references appear in the 19th century.',
        'nutrition': '💪 A typical taco has 200-300 calories. Soft tacos with grilled protein and plenty of vegetables are the healthiest choice.',
        'variations': ['Hard shell taco', 'Soft taco', 'Fish taco', 'Breakfast taco', 'Al pastor', 'Carnitas']
    },
    'Taquito': {
        'description': 'A Mexican dish consisting of a small rolled tortilla filled with meat or cheese, then deep-fried until crispy.',
        'ingredients': ['Small corn or flour tortilla', 'Shredded chicken or beef', 'Cheese', 'Spices', 'Oil for frying', 'Guacamole & sour cream for dipping'],
        'fun_fact': '🌯 Taquitos and flautas are similar, but taquitos are made with corn tortillas while flautas traditionally use flour tortillas and are larger!',
        'origin': '🌍 San Diego, California is often credited with popularizing taquitos in the 1940s, though rolled tacos existed in Mexico for centuries.',
        'nutrition': '💪 Each taquito contains approximately 140-180 calories. Baked taquitos are a lighter alternative to fried versions.',
        'variations': ['Chicken taquitos', 'Beef taquitos', 'Cheese taquitos', 'Flautas', 'Baked taquitos']
    }
}

VISITOR_STATS_FILE = 'visitor_stats.json'
IMG_SIZE = (224, 224)

# ========== Helper Functions ==========
@st.cache_resource
def load_food_model():
    """Load the trained model with caching"""
    try:
        model = load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_feature_maps(model, img_array, layer_indices=[2, 5, 8]):
    """Extract feature maps from specified layers"""
    feature_maps = []
    
    # Get valid layer indices
    valid_indices = [i for i in layer_indices if i < len(model.layers)]
    
    if valid_indices:
        try:
            layer_outputs = [model.layers[i].output for i in valid_indices]
            feature_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
            features = feature_model.predict(img_array)
            
            # Handle single layer output
            if not isinstance(features, list):
                features = [features]
            
            for i, feature in enumerate(features):
                if len(feature.shape) == 4:  # Ensure it's a convolutional layer output
                    feature_maps.append({
                        'layer_index': valid_indices[i],
                        'layer_name': model.layers[valid_indices[i]].name,
                        'feature': feature
                    })
        except Exception as e:
            st.warning(f"Could not extract feature maps: {e}")
    
    return feature_maps

def update_visitor_count():
    """Update and return visitor count"""
    if not os.path.exists(VISITOR_STATS_FILE):
        stats = {'total_visitors': 0}
    else:
        try:
            with open(VISITOR_STATS_FILE, 'r') as f:
                stats = json.load(f)
        except:
            stats = {'total_visitors': 0}
    
    if 'visited' not in st.session_state:
        stats['total_visitors'] = stats.get('total_visitors', 0) + 1
        st.session_state.visited = True
        
        try:
            with open(VISITOR_STATS_FILE, 'w') as f:
                json.dump(stats, f)
        except:
            pass
    
    return stats.get('total_visitors', 0)

def create_confidence_chart(predictions, class_labels):
    """Create a horizontal bar chart for all predictions"""
    classes = [class_labels[i] for i in sorted(class_labels.keys())]
    confidences = [predictions[0][i] * 100 for i in sorted(class_labels.keys())]
    
    # Sort by confidence
    sorted_pairs = sorted(zip(classes, confidences), key=lambda x: x[1], reverse=True)
    classes_sorted, confidences_sorted = zip(*sorted_pairs)
    
    # Color based on confidence
    colors = ['#FF6B6B' if c == max(confidences_sorted) else '#FFD93D' if c > 20 else '#E0E0E0' 
              for c in confidences_sorted]
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes_sorted,
            x=confidences_sorted,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{c:.1f}%' for c in confidences_sorted],
            textposition='auto',
            textfont=dict(size=12, color='white', family='Arial Black'),
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Food Categories",
        xaxis_title="Confidence (%)",
        yaxis_title="Food Category",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,0.5)',
        font=dict(size=13, color='#2C3E50'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    
    return fig

def get_confidence_class(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 80:
        return "confidence-high"
    elif confidence >= 50:
        return "confidence-medium"
    else:
        return "confidence-low"

# ========== Main Application ==========
def main():
    # Update visitor count
    visitor_count = update_visitor_count()
    
    # Visitor counter (fixed position)
    st.markdown(f"""
    <div class="visitor-counter">
        👥 <strong>{visitor_count}</strong> visitors
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍔 Fast Food Classification</h1>
        <p>Identify your favorite fast food using AI-powered image recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hamburger.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home - Classify Food", "📚 Food Encyclopedia", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🍽️ Supported Foods")
        for food_name in CLASS_LABELS.values():
            st.markdown(f"{FOOD_EMOJIS[food_name]} {food_name}")
        
        st.markdown("---")
        st.markdown("### Quick Info")
        st.info("""
        This AI system can identify 10 different types of fast food from images. 
        Upload a photo to get instant classification with detailed nutritional and historical information!
        """)
    
    # Main content based on selected page
    if page == "🏠 Home - Classify Food":
        show_classification_page()
    elif page == "📚 Food Encyclopedia":
        show_encyclopedia_page()
    elif page == "ℹ️ About":
        show_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Fast Food Classification System</strong> © 2024</p>
        <p>Developed by <a href="https://www.linkedin.com/in/mo-abdalkader/" target="_blank">Mohamed Abdalkader</a></p>
        <p>
            <a href="mailto:Mohameed.Abdalkadeer@gmail.com">📧 Email</a> | 
            <a href="https://github.com/Mo-Abdalkader" target="_blank">💻 GitHub</a>
        </p>
        <p style="font-size: 0.85rem; color: #999; margin-top: 1rem;">
            ⚠️ For educational purposes only. Nutritional information is approximate.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_classification_page():
    """Main classification page"""
    # Load model
    model = load_food_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check the model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.markdown("### 📤 Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose a food image (JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of fast food for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### 🔬 Analysis Options")
            
            show_feature_maps = st.checkbox("Show Feature Maps", value=False)
            show_all_predictions = st.checkbox("Show All Predictions", value=True)
            show_detailed_info = st.checkbox("Show Detailed Food Info", value=True)
            
            analyze_button = st.button("🔍 Classify Food", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Analyzing food image..."):
                # Preprocess and predict
                processed_img = preprocess_image(image, target_size=IMG_SIZE)
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                food_name = CLASS_LABELS.get(predicted_class, 'Unknown')
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Classification Results")
                
                # Result card
                confidence_class = get_confidence_class(confidence)
                st.markdown(f"""
                <div class="result-card">
                    <div class="prediction-emoji">{FOOD_EMOJIS[food_name]}</div>
                    <p class="prediction-label">Detected Food</p>
                    <p class="prediction-value">{food_name}</p>
                    <p class="prediction-label">Confidence: <span class="{confidence_class}">{confidence:.2f}%</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown("#### Confidence Level")
                st.progress(confidence / 100)
                
                # Detailed food information
                if show_detailed_info:
                    st.markdown("---")
                    st.markdown(f"## 📖 About {food_name}")
                    
                    info = FOOD_INFO[food_name]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>📝 Description</h3>
                            <p>{info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>🔥 Fun Fact</h3>
                            <p>{info['fun_fact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>🌍 Origin</h3>
                            <p>{info['origin']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>🍴 Common Ingredients</h3>
                            <ul>
                                {''.join([f'<li>{ingredient}</li>' for ingredient in info['ingredients']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>💪 Nutritional Info</h3>
                            <p>{info['nutrition']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="food-card">
                            <h3>🎭 Popular Variations</h3>
                            <ul>
                                {''.join([f'<li>{variation}</li>' for variation in info['variations']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # All predictions chart
                if show_all_predictions:
                    st.markdown("---")
                    st.markdown("### 📊 All Classification Scores")
                    fig = create_confidence_chart(predictions, CLASS_LABELS)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature maps
                if show_feature_maps:
                    st.markdown("---")
                    st.markdown("### 🗺️ Neural Network Feature Maps")
                    st.info("Feature maps show what the AI 'sees' at different layers of the neural network")
                    
                    with st.spinner("Generating feature maps..."):
                        feature_maps = get_feature_maps(model, processed_img)
                        
                        if feature_maps:
                            for fm in feature_maps:
                                st.markdown(f"#### Layer {fm['layer_index']}: {fm['layer_name']}")
                                
                                # Display first 16 feature maps
                                features = fm['feature'][0]
                                n_features = min(16, features.shape[-1])
                                
                                cols = st.columns(4)
                                for i in range(n_features):
                                    with cols[i % 4]:
                                        feature_img = features[:, :, i]
                                        # Normalize feature map
                                        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min() + 1e-8)
                                        st.image(feature_img, caption=f"Filter {i+1}", use_column_width=True)
                        else:
                            st.warning("Could not generate feature maps for this model architecture")

def show_encyclopedia_page():
    """Food encyclopedia page with detailed information"""
    st.markdown("## 📚 Fast Food Encyclopedia")
    st.markdown("Learn everything about your favorite fast foods!")
    
    # Food selector
    selected_food = st.selectbox(
        "Select a food to learn more:",
        list(CLASS_LABELS.values()),
        format_func=lambda x: f"{FOOD_EMOJIS[x]} {x}"
    )
    
    info = FOOD_INFO[selected_food]
    
    # Display detailed information
    st.markdown(f"""
    <div class="result-card">
        <div class="prediction-emoji">{FOOD_EMOJIS[selected_food]}</div>
        <h1>{selected_food}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout for information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="food-card">
            <h3>📝 Description</h3>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="food-card">
            <h3>🍴 Common Ingredients</h3>
            <ul>
                {''.join([f'<li>{ingredient}</li>' for ingredient in info['ingredients']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="food-card">
            <h3>🎭 Popular Variations</h3>
            <ul>
                {''.join([f'<li>{variation}</li>' for variation in info['variations']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="food-card">
            <h3>🔥 Fun Fact</h3>
            <p>{info['fun_fact']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="food-card">
            <h3>🌍 Origin & History</h3>
            <p>{info['origin']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="food-card">
            <h3>💪 Nutritional Information</h3>
            <p>{info['nutrition']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_about_page():
    """About page with project information"""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    <div class="food-card">
        <h3>🎯 Project Overview</h3>
        <p>
            This Fast Food Classification System uses deep learning and computer vision to identify 
            10 different types of popular fast foods from images. The system is powered by a 
            convolutional neural network (CNN) trained on thousands of food images.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="food-card">
            <h3>🤖 How It Works</h3>
            <p>
                <strong>1. Image Upload:</strong> You upload a photo of fast food<br><br>
                <strong>2. Preprocessing:</strong> The image is resized and normalized<br><br>
                <strong>3. AI Analysis:</strong> A neural network analyzes visual features<br><br>
                <strong>4. Classification:</strong> The system identifies the food type<br><br>
                <strong>5. Results:</strong> You get detailed information and confidence scores
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="food-card">
            <h3>🎓 Educational Value</h3>
            <p>
                This project demonstrates practical applications of:
            </p>
            <ul>
                <li>Deep Learning & Computer Vision</li>
                <li>Image Classification with CNNs</li>
                <li>Transfer Learning Techniques</li>
                <li>Model Deployment with Streamlit</li>
                <li>Interactive Web Applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="food-card">
            <h3>🍔 Supported Foods</h3>
            <p>The system can identify these 10 fast food categories:</p>
            <ul>
        """, unsafe_allow_html=True)
        
        for food_name in CLASS_LABELS.values():
            st.markdown(f"<li>{FOOD_EMOJIS[food_name]} {food_name}</li>", unsafe_allow_html=True)
        
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="food-card">
            <h3>⚙️ Technical Stack</h3>
            <ul>
                <li><strong>Framework:</strong> TensorFlow/Keras</li>
                <li><strong>UI:</strong> Streamlit</li>
                <li><strong>Visualization:</strong> Plotly</li>
                <li><strong>Image Processing:</strong> PIL, NumPy</li>
                <li><strong>Language:</strong> Python 3.11</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="food-card">
        <h3>👨‍💻 Developer</h3>
        <p>
            Created by <strong>Mohamed Abdalkader</strong> - A passionate developer focused on 
            AI, machine learning, and creating intuitive user experiences.
        </p>
        <p>
            📧 Email: <a href="mailto:Mohameed.Abdalkadeer@gmail.com">Mohameed.Abdalkadeer@gmail.com</a><br>
            💼 LinkedIn: <a href="https://www.linkedin.com/in/mo-abdalkader/" target="_blank">mo-abdalkader</a><br>
            💻 GitHub: <a href="https://github.com/Mo-Abdalkader" target="_blank">Mo-Abdalkader</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="food-card">
        <h3>⚠️ Disclaimer</h3>
        <p>
            This application is for educational and entertainment purposes only. The nutritional 
            information provided is approximate and should not be used as a substitute for 
            professional dietary advice. Food classification accuracy depends on image quality 
            and may not always be 100% accurate.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the main application
main()
