from openai import OpenAI

class Qwen2LLM:
    def __init__(self, api_key, model="Qwen/Qwen2-7B-Instruct"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    def chat(self, messages, system_prompt, stream=False):
        system_prompt = {"role": "system", "content": system_prompt}

        cur_messages = [system_prompt] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=cur_messages,
            temperature=0.7,
            stream=stream
        )
        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            yield response.choices[0].message.content
