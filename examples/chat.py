from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'who is prime minister of india?',
  },
]

response = chat('llama3.2', messages=messages)
print(response['message']['content'])
