from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
from zodbpickle import pickle
with open('LSTM_model.pkl', 'rb') as file:
    model = pickle.loads(file)

app = FastAPI()

#  request and response formats using Pydantic BaseModel
class PredictionRequest(BaseModel):
    external_status: str

class PredictionResponse(BaseModel):
    internal_status: str

#  API endpoint for model prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict_internal_status(request: PredictionRequest):
    try:
        # Tokenize and pad the input text
        encoded_input = tokenizer.texts_to_sequences([request.external_status])
        padded_input = pad_sequences(encoded_input, maxlen=max_sequence_length, padding='post')
        
        prediction = model.predict(padded_input)
        
        predicted_class = np.argmax(prediction)
        
        # Decode the predicted class label
        predicted_internal_status = label_encoder.inverse_transform([predicted_class])[0]
        
        # Return the predicted internal status
        return {"internal_status": predicted_internal_status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
