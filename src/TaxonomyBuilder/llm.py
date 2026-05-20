class LLMProvider:
    """Base class for all LLM providers."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key, model):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        message = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()

class GoogleProvider(LLMProvider):
    def __init__(self, api_key, model):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        message = self.client.models.generate_content(
                model=self.model, contents=prompt
            )
        return message.text.strip()