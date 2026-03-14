# 🌾 AI Crop Recommendation System

A machine learning-powered web application that recommends the best crops based on soil and weather parameters.

🚀 **[Live Demo](https://agrisense-crop-recommendation-system.streamlit.app/)** | 📂 **[GitHub Repository](https://github.com/CosmicCoderX/AI-Crop-Recommendation-System)**

## Screenshots

<img width="1902" alt="Main Interface" src="https://github.com/user-attachments/assets/2ec830ad-9661-41c9-a8fa-efd7e8b5d3ec" />
<p></p>
<img width="1892" alt="Prediction Results" src="https://github.com/user-attachments/assets/7b6a77e7-daaf-4b52-940e-4c01c39cfb43" />

## Features

- 🤖 AI-powered crop predictions with confidence scores
- 📊 Top 3 crop recommendations
- 📈 Visual analytics for optimal growing conditions
- 🎨 Modern, responsive UI
- 🖼️ Crop images and detailed insights

## Supported Crops

**22+ crops** including Rice, Wheat, Maize, Cotton, various Pulses, and Fruits (Apple, Banana, Mango, Grapes, etc.)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/CosmicCoderX/AI-Crop-Recommendation-System.git
cd AI-Crop-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. Enter soil parameters (N, P, K, pH)
2. Add weather data (Temperature, Humidity, Rainfall)
3. Click "Predict Top 3 Crops"
4. View recommendations with confidence scores

## Input Parameters

| Parameter | Unit | Range |
|-----------|------|-------|
| Nitrogen (N) | kg/ha | 0-140 |
| Phosphorus (P) | kg/ha | 5-145 |
| Potassium (K) | kg/ha | 5-205 |
| Temperature | °C | 8-44 |
| Humidity | % | 14-100 |
| pH | - | 3.5-9.9 |
| Rainfall | mm | 20-300 |

## Project Structure

```
├── app.py                      # Main application
├── model.pkl                   # Trained ML model
├── scaler.pkl                  # Feature scaler
├── Crop_recommendation.csv     # Dataset
├── requirements.txt            # Dependencies
└── images/                     # Crop images
```

## Technologies

- **Frontend:** Streamlit
- **ML:** scikit-learn
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

**CosmicCoderX**
- GitHub: [@CosmicCoderX](https://github.com/CosmicCoderX)

---

⭐ Star this repository if you find it helpful!
