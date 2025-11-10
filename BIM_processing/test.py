import ollama

res = ollama.chat(
    model="llama3.2-vision:11b",
    messages=[{
        "role": "user",
        "content": "Caption this image in one short sentence.",
        "images": ["workers.jpeg"],  # PATH
    }],
    options={"temperature": 0}
)
print(res["message"]["content"])
