from collections import deque
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregator,
)
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)


class AudioContextAggregator(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        context_aggregator: LLMContextAggregator,
        *,
        start_secs: float = 0.2,
    ):
        super().__init__()
        self._context = context
        self._context_aggregator = context_aggregator
        self._audio_frames = deque()
        self._audio_duration = 0
        self._start_secs = start_secs
        self._is_user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_user_speaking = False
            await self._context.add_audio_frames_message(
                audio_frames=self._audio_frames, text=""
            )
            await self._context_aggregator.push_frame(
                LLMContextFrame(context=self._context)
            )
        elif isinstance(frame, InputAudioRawFrame):
            self._audio_frames.append(frame)
            self._audio_duration += self._get_duration(frame)

            if not self._is_user_speaking:
                while self._audio_duration > self._start_secs:
                    popped_frame = self._audio_frames.popleft()
                    self._audio_duration -= self._get_duration(popped_frame)

        await self.push_frame(frame, direction)

    @staticmethod
    def _get_duration(frame: InputAudioRawFrame) -> float:
        return len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
