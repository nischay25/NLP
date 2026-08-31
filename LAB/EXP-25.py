from openai import OpenAI

client = OpenAI()

prompt = input("Enter your prompt: ")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)

print("\nGenerated Text:\n")
print(response.output_text)
