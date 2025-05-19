from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import gradio as gr
from prediction_service import predict_interaction

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DrugPair(BaseModel):
    drug1: str
    drug2: str

class PredictionResponse(BaseModel):
    has_interaction: bool
    severity: Optional[str] = None
    description: Optional[str] = None
    confidence: float
    warning: Optional[str] = None

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