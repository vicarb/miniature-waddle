import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def chat_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

while True:
    user_input = input("You: ")
    bot_response = chat_with_gpt3(user_input)
    print("GPT-3: " + bot_response)

