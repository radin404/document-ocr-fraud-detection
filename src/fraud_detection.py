from transformers import pipeline

class FraudDetector:
    def __init__(self):
        self.llm = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def detect_anomalies(self, text):
        # Check for suspicious keywords (e.g., mismatched names, dates)
        keywords = ["fake", "expired", "mismatch"]
        if any(keyword in text.lower() for keyword in keywords):
            return "Potential fraud detected: Suspicious keywords found."
        
        # Use LLM to assess semantic inconsistencies
        llm_result = self.llm(text)
        if llm_result[0]['label'] == 'NEGATIVE':
            return "Potential fraud detected: Negative sentiment or inconsistencies."
        return "Document appears valid."

# Example usage:
if __name__ == "__main__":
    detector = FraudDetector()
    result = detector.detect_anomalies("This is a fake document.")
    print(result)