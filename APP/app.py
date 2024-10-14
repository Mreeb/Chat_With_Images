import streamlit as st
from PIL import Image
from ollama import generate

def explain_image_streaming(image, prompt):
    """Analyze image and generate a streaming response."""
    # System message to explain the task for analysis
    system_message = (
        """You are LLAVA, a powerful Vision-Language Assistant. Your job is to analyze and interpret both text and visual inputs, providing accurate, insightful, and creative responses. For images, describe content clearly, explain relationships between objects, and infer deeper meaning when necessary. For text, respond with clarity, precision, and contextual relevance. Combine visual and textual insights when appropriate, and tailor responses to the user’s needs, maintaining a helpful and professional tone."""
    )
    
    # Convert image to bytes for model input
    image_bytes = image.read()

    # Streaming the response
    explanation = ""
    response_placeholder = st.empty()  # Placeholder for streaming updates

    for response in generate('llava', system_message + prompt, images=[image_bytes], stream=True):
        explanation += response['response']
        response_placeholder.markdown(f"### Response:\n\n{explanation}")

    return explanation

# Initialize session state for response persistence
if "response" not in st.session_state:
    st.session_state.response = ""

# Streamlit app layout
st.title("Image Analysis with Prompt Input (Streaming)")

# Image Upload Feature
uploaded_image = st.file_uploader("Upload an image (graph or chart)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Prompt Input and Response
    prompt = st.text_input("Enter Prompt")

    if st.button("Send"):
        if prompt:
            # Show spinner while the model processes the image and prompt
            with st.spinner('Processing your request...'):
                st.session_state.response = explain_image_streaming(uploaded_image, prompt)
            st.success("Analysis complete!")
        else:
            st.error("Please enter a prompt.")
