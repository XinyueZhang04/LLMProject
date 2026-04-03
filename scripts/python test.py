import ollama

while True:
    user_input = input("You: ")
    response = ollama.chat(
        model='qwen2:1.5b',
        messages=[{'role':'user', 'content': user_input}]
    )
    print("AI:", response['message']['content'])