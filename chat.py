import ollama
response = ollama.chat(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": "who are atrick Marlow and Vladimir Vuskovic?"},
    ],
)
print(response["message"]["content"])