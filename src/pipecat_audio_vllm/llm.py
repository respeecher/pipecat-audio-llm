from pipecat.services.openai.base_llm import BaseOpenAILLMService


class AudioLLMService(BaseOpenAILLMService):
    class InputParams(BaseOpenAILLMService.InputParams):
        pass

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        *,
        model: str = "",
        api_key: str = "",
        **kwargs,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key, **kwargs)
