const checkInteraction = async () => {
  if (!drug1 || !drug2) {
    setError('Please select both drugs');
    return;
  }

  setLoading(true);
  setError(null);
  setResult(null);

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        drug1: drug1,
        drug2: drug2,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to check interaction');
    }

    const data = await response.json();
    setResult({
      hasInteraction: data.prediction !== 'no_interaction',
      severity: data.prediction,
      description: data.description || `Predicted ${data.prediction} interaction with ${(data.confidence * 100).toFixed(1)}% confidence`,
      confidence: data.confidence
    });
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false);
  }
}; 