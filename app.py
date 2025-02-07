import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a publicly available question-answering model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Expanded healthcare knowledge base
context = """
A healthcare chatbot assists users by answering common medical-related queries.
1. Symptoms: If you have symptoms like fever, cough, or headache, it's recommended to see a doctor for a proper diagnosis.
2. Appointments: You can book an appointment with a doctor for consultation or follow-ups.
3. Medications: Always take medications as prescribed by your healthcare provider. If you miss a dose, consult your doctor.
4. Emergency: If you experience severe chest pain, difficulty breathing, or sudden dizziness, seek emergency medical attention immediately.
5. General Health: Maintain a healthy diet, exercise regularly, and follow medical guidelines for a better lifestyle.
This chatbot provides basic guidance, but it is not a replacement for a professional healthcare provider.
"""

def healthcare_chatbot(user_input):
    """Function to process user queries related to healthcare"""
    try:
        user_input = user_input.lower()
        
        # Predefined responses for specific topics
        if "symptom" in user_input:
            return "If you're experiencing symptoms, I recommend consulting a doctor for an accurate diagnosis."
        elif "appointment" in user_input:
            return "Would you like me to guide you through the process of scheduling a doctor's appointment?"
        elif "medication" in user_input or "medicine" in user_input:
            return "Always take prescribed medications as directed. If you have concerns about dosage, consult a doctor."

        # Use QA model for other health-related queries
        response = qa_model(question=user_input, context=context)

        # Check if the model generates a meaningful answer
        answer = response['answer']
        if answer.lower() == "healthcare chatbot" or len(answer) < 5:
            return "I'm sorry, I couldn't find a relevant answer. Please consult a medical professional."
        
        return answer

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main function to run Streamlit chatbot"""
    st.title("Healthcare Assistance Chatbot")
    
    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
                st.write("Healthcare Assistance:", response)
        else:
            st.warning("Please enter a message to get a response.")

if __name__ == "__main__":
    main()