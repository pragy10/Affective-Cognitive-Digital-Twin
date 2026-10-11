import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openrouter_with_fallbacks():
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please add it to your .env file in the root of the project like this:")
        print("OPENROUTER_API_KEY=your_actual_key_here")
        return

    print("Connecting to OpenRouter API...")
    
    # Initialize the OpenAI client pointing to OpenRouter's base URL
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    print("Sending test prompt: 'Hello Tutor. Are you ready to help students?'\n")
    
    # List of models to try in order of preference
    # We mix different providers to ensure high availability
    fallback_models = [
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.1-8b-instruct",
        "anthropic/claude-3-haiku",
        "mistralai/mistral-7b-instruct"
    ]
    
    success = False
    
    for model_name in fallback_models:
        print(f"Attempting with model: {model_name}...")
        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {
                        "role": "user",
                        "content": "Hello Tutor. Are you ready to help students?"
                    }
                ],
                # OpenRouter specific headers
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000", # Required by OpenRouter for ranking
                    "X-Title": "ACDT Capstone Project", # Optional
                },
                timeout=15.0 # Set a timeout so we don't wait forever before trying the next model
            )
            
            reply = response.choices[0].message.content
            print("\n--- OpenRouter Response ---")
            print(reply)
            print("---------------------------")
            print(f"\nSuccess! Successfully connected and got a response using: {model_name}")
            
            success = True
            break # Exit the loop since we succeeded!
            
        except Exception as e:
            print(f"Failed with {model_name}. Error: {e}\nSwitching to next fallback...\n")
    
    if not success:
        print("Critical Failure: All fallback models failed. Please check your API key and internet connection.")

if __name__ == "__main__":
    test_openrouter_with_fallbacks()
