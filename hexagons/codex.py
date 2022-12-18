import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def codex_predict(prompt, **kwargs):
    request_kwargs = {
        "temperature": 0.1,
        "max_tokens": 256,
        "top_p": 1,
        "best_of": 5,
        "frequency_penalty": 0,
        "stop": ["#"]
    }
    for key, value in kwargs.items():
        request_kwargs[key] = value
    while True:
        try:
            response = openai.Completion.create(
                model="code-davinci-002",
                prompt=prompt,
                **request_kwargs
            )
            break
        except openai.error.RateLimitError as e:
            print("Rate limit error! Retrying...")
            sleep(60)
            continue
    target = response["choices"][0]["text"]
    full_program = prompt + target
    return full_program


