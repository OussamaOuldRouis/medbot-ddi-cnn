# Drug-Drug Interaction Prediction Service

A FastAPI-based service that provides drug interaction predictions using machine learning. The service combines a TensorFlow model with a SQLite database to predict and describe potential interactions between drugs.

## Features

- Real-time drug interaction predictions
- Severity classification (mild, moderate, severe)
- Confidence scoring
- Detailed interaction descriptions
- Drug embedding generation
- Gradio web interface
- RESTful API endpoints
- Comprehensive error handling
- Database-backed drug information

## Tech Stack

- Python 3.8+
- FastAPI
- TensorFlow
- SQLite
- Pandas
- NumPy
- Gradio
- scikit-learn
- joblib

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- SQLite3
- Virtual environment (recommended)

## Installation

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
# On Unix or MacOS:
source venv/bin/activate
```

2. Install required packages:
```bash
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn joblib gradio sqlite3
```

3. Ensure the following files are present in the `model` directory:
- `ddi_model.h5`: TensorFlow model file
- `drug_encoder.joblib`: Drug name encoder
- `interaction_encoder.joblib`: Interaction type encoder

4. Ensure the SQLite database file `event.db` is present in the project directory

## Configuration

The service uses the following default configurations:
- Server runs on `0.0.0.0:8000`
- Model input size: 572 features
- Confidence threshold: 0.1 (10%)
- Maximum feature vector size: 572

## Usage

1. Start the server:
```bash
python prediction_service.py
```

2. The service provides the following endpoints:
   - `POST /predict`: Main prediction endpoint
   - `GET /`: Gradio web interface

3. Example prediction request:
```json
{
    "drug1": "Aspirin",
    "drug2": "Warfarin"
}
```

## API Response Format

The prediction endpoint returns responses in the following format:
```json
{
    "has_interaction": boolean,
    "severity": "mild|moderate|severe",
    "description": "string",
    "confidence": float,
    "warning": "string"
}
```

## Database Schema

The service uses a SQLite database with the following tables:

1. `drug` table:
   - `name`: Drug name
   - `pathway`: Drug pathway information
   - `target`: Drug target information
   - `enzyme`: Drug enzyme information

2. `event` table:
   - `name1`: First drug name
   - `name2`: Second drug name
   - `interaction`: Interaction description

## Model Architecture

The prediction service uses a TensorFlow model that:
- Takes drug embeddings as input
- Processes features through neural network layers
- Outputs interaction predictions with confidence scores
- Supports both known and unknown drugs

## Error Handling

The service includes comprehensive error handling for:
- Database connection issues
- Model loading failures
- Invalid drug names
- Missing data
- API request validation

## Gradio Interface

The service includes a Gradio web interface that provides:
- User-friendly input forms
- Real-time predictions
- Visual severity indicators
- Example drug pairs
- Confidence score display

## Performance Considerations

- Model predictions are cached
- Database connections are managed efficiently
- Feature vectors are pre-computed
- Batch processing support
- Memory-efficient data handling

## Security

- CORS middleware enabled
- Input validation
- Error message sanitization
- Database query parameterization

## Development

To modify the system:
1. Update the model architecture in `ddi_model.h5`
2. Modify the database schema if needed
3. Adjust the confidence thresholds
4. Update the severity classification logic
5. Modify the feature vector generation

## Testing

1. Test the API endpoints:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"drug1":"Aspirin","drug2":"Warfarin"}'
```

2. Access the Gradio interface:
   - Open `http://localhost:8000` in your browser
   - Use the provided example drug pairs
   - Test with custom drug combinations

## Deployment

For production deployment:
1. Set up proper environment variables
2. Configure SSL/TLS
3. Set up proper logging
4. Implement rate limiting
5. Set up monitoring
6. Configure backup procedures
7. Implement proper error reporting

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
MODEL_PATH=model/ddi_model.h5
DB_PATH=event.db
HOST=0.0.0.0
PORT=8000
```

## Troubleshooting

Common issues and solutions:
1. If the model fails to load:
   - Check if model files exist in the correct location
   - Verify model file permissions
   - Check TensorFlow version compatibility

2. If database queries fail:
   - Verify database file exists
   - Check database file permissions
   - Ensure correct table schema

3. If predictions are inaccurate:
   - Check drug name spelling
   - Verify drug exists in database
   - Check model confidence scores

## License

[Add your license information here]

## Support

For support, please [add contact information or support channels] 