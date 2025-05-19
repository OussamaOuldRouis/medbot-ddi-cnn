import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sqlite3
import gradio as gr

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Initialize encoders
def initialize_encoders():
    try:
        # Try to load existing encoders
        drug_encoder = joblib.load('model/drug_encoder.joblib')
        interaction_encoder = joblib.load('model/interaction_encoder.joblib')
    except FileNotFoundError:
        print("Encoders not found. Creating new encoders...")
        # Create new encoders
        drug_encoder = LabelEncoder()
        interaction_encoder = LabelEncoder()
        
        # Connect to SQLite database
        conn = sqlite3.connect('event.db')
        
        # Get unique drugs from the drug table
        drugs_df = pd.read_sql('SELECT DISTINCT name FROM drug', conn)
        drug_names = drugs_df['name'].tolist()
        
        # Get unique interaction types
        interaction_types = ['no_interaction', 'mild', 'moderate', 'severe']
        
        # Fit encoders
        drug_encoder.fit(drug_names)
        interaction_encoder.fit(interaction_types)
        
        # Save encoders
        joblib.dump(drug_encoder, 'model/drug_encoder.joblib')
        joblib.dump(interaction_encoder, 'model/interaction_encoder.joblib')
        print("Encoders created and saved successfully.")
        
        # Close database connection
        conn.close()
    
    return drug_encoder, interaction_encoder

# Load the model and encoders with error handling
try:
    model = tf.keras.models.load_model('model/ddi_model.h5')
    drug_encoder, interaction_encoder = initialize_encoders()
    print("Model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading model or encoders: {str(e)}")
    raise Exception("Failed to load model or encoders. Please ensure all model files are present.")

class DrugPair(BaseModel):
    drug1: str
    drug2: str

class PredictionResponse(BaseModel):
    has_interaction: bool
    severity: Optional[str] = None
    description: Optional[str] = None
    confidence: float
    warning: Optional[str] = None

def get_drug_embedding(drug_name: str) -> Tuple[np.ndarray, bool]:
    """Get drug embedding from SQLite database. Returns (embedding, is_known_drug)."""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('event.db')
        
        # Query the drug table for the drug's features
        query = '''
        SELECT pathway, target, enzyme
        FROM drug 
        WHERE name = ?
        '''
        drug_data = pd.read_sql(query, conn, params=(drug_name,))
        
        # Create a single feature vector of size 572
        feature_vector = np.zeros(572)  # VECTOR_SIZE from final.py
        
        if drug_data.empty:
            # Drug not found in database, return zero vector
            conn.close()
            return feature_vector, False
        
        # Combine all features into a single vector
        for feature in ['pathway', 'target', 'enzyme']:
            if feature in drug_data.columns:
                feature_values = drug_data[feature].iloc[0].split('|')
                for i, value in enumerate(feature_values):
                    if i < 572:  # VECTOR_SIZE
                        feature_vector[i] = 1
        
        # Close database connection
        conn.close()
        
        return feature_vector, True
    except Exception as e:
        print(f"Error getting drug embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting drug embedding: {str(e)}")

def get_interaction_description(drug1: str, drug2: str) -> Optional[str]:
    """Get interaction description from SQLite database."""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('event.db')
        
        # Query the event table
        query = '''
        SELECT interaction 
        FROM event 
        WHERE (name1 = ? AND name2 = ?) OR (name1 = ? AND name2 = ?)
        '''
        interaction_data = pd.read_sql(query, conn, params=(drug1, drug2, drug2, drug1))
        
        # Close database connection
        conn.close()
        
        if not interaction_data.empty:
            return interaction_data['interaction'].iloc[0]
        return None
    except Exception as e:
        print(f"Error getting interaction description: {str(e)}")
        return None

def get_all_interaction_descriptions() -> List[str]:
    """Get all unique interaction descriptions from the database."""
    try:
        conn = sqlite3.connect('event.db')
        descriptions_df = pd.read_sql('SELECT DISTINCT interaction FROM event WHERE interaction IS NOT NULL', conn)
        conn.close()
        return descriptions_df['interaction'].tolist()
    except Exception as e:
        print(f"Error getting interaction descriptions: {str(e)}")
        return []

def get_severity_from_description(description: str) -> str:
    """Determine severity based on the interaction description."""
    description = description.lower()
    if any(word in description for word in ['severe', 'serious', 'life-threatening', 'fatal']):
        return 'severe'
    elif any(word in description for word in ['moderate', 'significant', 'substantial']):
        return 'moderate'
    elif any(word in description for word in ['mild', 'minor', 'slight']):
        return 'mild'
    return 'mild'  # default to mild if no clear severity indicators

def predict_interaction(drug1: str, drug2: str) -> Dict:
    """Predict interaction between two drugs."""
    try:
        # Get embeddings for both drugs
        drug1_embedding, drug1_known = get_drug_embedding(drug1)
        drug2_embedding, drug2_known = get_drug_embedding(drug2)
        
        # Stack embeddings to create input of shape (1, 572, 2)
        combined_embedding = np.stack([drug1_embedding, drug2_embedding], axis=1)
        combined_embedding = np.expand_dims(combined_embedding, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(combined_embedding)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # First try to get the interaction from the database
        description = get_interaction_description(drug1, drug2)
        
        # If no description in database, use the model's prediction
        if description is None:
            # Get all possible descriptions
            conn = sqlite3.connect('event.db')
            descriptions_df = pd.read_sql('SELECT DISTINCT interaction FROM event WHERE interaction IS NOT NULL', conn)
            conn.close()
            all_descriptions = descriptions_df['interaction'].tolist()
            if predicted_class < len(all_descriptions):
                description = all_descriptions[predicted_class]
        
        # Determine severity based on the description
        severity = get_severity_from_description(description) if description else None
        
        # Determine if there's an interaction
        has_interaction = description is not None and confidence > 0.1
        
        # Generate warning if drugs are not in database
        warning = None
        if not drug1_known and not drug2_known:
            warning = f"Both {drug1} and {drug2} are not in our database. Prediction may be less accurate."
        elif not drug1_known:
            warning = f"{drug1} is not in our database. Prediction may be less accurate."
        elif not drug2_known:
            warning = f"{drug2} is not in our database. Prediction may be less accurate."
        
        return {
            "has_interaction": has_interaction,
            "severity": severity,
            "description": description,
            "confidence": confidence * 100,  # Convert to percentage
            "warning": warning
        }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(drug_pair: DrugPair):
    return predict_interaction(drug_pair.drug1, drug_pair.drug2)

# Gradio interface
def gradio_predict(drug1: str, drug2: str) -> str:
    result = predict_interaction(drug1, drug2)
    
    output = []
    if result["warning"]:
        output.append(f"⚠️ {result['warning']}")
    
    if result["severity"]:
        severity_color = {
            "severe": "🔴",
            "moderate": "🟠",
            "mild": "🟡"
        }
        output.append(f"{severity_color[result['severity']]} {result['severity'].upper()} Interaction Detected")
        output.append(f"Confidence: {result['confidence']:.1f}%")
    
    if result["description"]:
        output.append(f"\nDescription:\n{result['description']}")
    
    return "\n".join(output)

# Create Gradio interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(label="Drug 1"),
        gr.Textbox(label="Drug 2")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Drug-Drug Interaction Predictor",
    description="Enter two drug names to predict their potential interaction.",
    examples=[
        ["Abemaciclib", "Amiodarone"],
        ["Aspirin", "Warfarin"],
        ["Lisinopril", "Ibuprofen"]
    ]
)

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 