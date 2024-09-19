import nest_asyncio
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv
load_dotenv()

# Apply asyncio patch if running in a notebook
nest_asyncio.apply()

# Define config path to the root config file
config_path = "./config"  # Path to your config directory
config = RailsConfig.from_path(config_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")
model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")

llm = model
print(llm,"\n\n",model)

# Initialize the LLMRails instance
rails = LLMRails(config)
print(rails)

# Define a function to generate responses based on user input
def generate_response(user_message):
    response = rails.generate(messages=[{
        "role": "user",
        "content": user_message
    }])
    return response["content"]

def main():
    print("Welcome to the shopping Assistant!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            break

        bot_response = generate_response(user_message)
        print(f"Bot: {bot_response}\n")

        # Optional: Retrieve and print conversation history and LLM call summary
        info = rails.explain()
        print("Conversation History:")
        print(info.colang_history)
        print("\nLLM Calls Summary:")
        info.print_llm_calls_summary()

if __name__ == "__main__":
    main()