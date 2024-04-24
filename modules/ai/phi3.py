import torch
import transformers

# Function to generate text with the Phi-3-Mini-128K-Instruct model
def ask_phi3(system_content, user_content):
    # Model ID for the Phi-3-Mini-128K-Instruct model
    model_id = "microsoft/Phi-3-mini-128k-instruct"

    # Initialize the text-generation pipeline with appropriate device and data type
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="cuda",  # Adjust based on your hardware
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Define the messages to set up the context for the chatbot
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Generate the response using the pipeline
    response = pipeline(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
    )

    # Return the generated text
    return response[0]["generated_text"]
