from collections import deque
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import (
    TranscriptionFrame,
    InterimTranscriptionFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.utils.time import time_now_iso8601


class AudioContextAggregator(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        *,
        start_secs: float = 0.2,
        text: str | None = None,
        push_visual_transcription: bool = False,
    ):
        super().__init__()
        self._context = context
        self._audio_frames = deque()
        self._audio_duration = 0
        self._start_secs = start_secs
        self._is_user_speaking = False
        self._text = text

        self._push_visual_transcription = push_visual_transcription
        self._visual_transcription = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_user_speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_user_speaking = False

            message = await self._context.create_audio_message(
                audio_frames=self._audio_frames, text=self._text or ""
            )

            self._audio_frames.clear()
            self._audio_duration = 0

            if self._text is None:
                assert message["content"][0]["type"] == "text"
                del message["content"][0]

            self._context.add_message(message)

            await self.push_frame(LLMContextFrame(context=self._context))

            if self._push_visual_transcription:
                await self.push_frame(
                    TranscriptionFrame(
                        self._visual_transcription,
                        "",
                        time_now_iso8601(),
                    )
                )
                self._visual_transcription = ""

        elif isinstance(frame, InputAudioRawFrame):
            self._audio_frames.append(frame)
            self._audio_duration += self._get_duration(frame)

            if self._is_user_speaking:
                new_transcription_length = max(1, int(self._audio_duration * 5))

                if self._push_visual_transcription and new_transcription_length > len(
                    self._visual_transcription
                ):
                    self._visual_transcription += chr(0x1F4AC) * (
                        new_transcription_length - len(self._visual_transcription)
                    )

                    await self.push_frame(
                        InterimTranscriptionFrame(
                            self._visual_transcription,
                            "",
                            time_now_iso8601(),
                        )
                    )
            else:
                while self._audio_duration > self._start_secs:
                    popped_frame = self._audio_frames.popleft()
                    self._audio_duration -= self._get_duration(popped_frame)

        await self.push_frame(frame, direction)

    @staticmethod
    def _get_duration(frame: InputAudioRawFrame) -> float:
        # Note: official foundational examples in Pipecat use another formula for some reason:
        # len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
        return len(frame.audio) / 2 / frame.num_channels / frame.sample_rate
