import torch
import transformers

# Function to generate text with a pirate chatbot style
def ask_llama3(system_content, user_content):
    # Model ID for the Meta-Llama-3-8B-Instruct model
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
    

    # Initialize the text-generation pipeline with appropriate device and data type
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device="auto",
    # )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device="auto",
    )

    # Define the messages to set up the context for the chatbot
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Create the prompt from the messages
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Define end-of-sequence tokens for response termination
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")  # Convert empty token to ID
    ]

    # Generate text with the specified parameters
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Extract the generated text excluding the original prompt
    response = outputs[0]["generated_text"][len(prompt):]

    # Return the generated text
    return response
