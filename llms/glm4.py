from openai import OpenAI

class GLM4LLM:
    def __init__(self, api_key, model="glm-4-flash"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")

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
