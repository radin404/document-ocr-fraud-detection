from fastapi import FastAPI, UploadFile, File
from ocr import extract_text
from fraud_detection import FraudDetector
import shutil

app = FastAPI()

@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    # Save uploaded file
    with open("data/raw/temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text
    text = extract_text("data/raw/temp.jpg")
    
    # Detect fraud
    detector = FraudDetector()
    fraud_result = detector.detect_anomalies(text)
    
    return {"text": text, "fraud_analysis": fraud_result}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)