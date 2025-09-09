from langchain_community.llms import Ollama


def main():
    # Initialize the Ollama model with the desired model name
    model = Ollama(model="gpt-oss:latest")

    # Define the prompt to be sent to the model
    prompt = "hi"

    # Get the response from the model
    response = model.invoke(prompt)

    # Print the response
    print(response)


if __name__ == "__main__":
    main()
