from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)


class AudioUserTurnStopStrategy(BaseUserTurnStopStrategy):
    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._vad_user_speaking = False

    async def reset(self):
        await super().reset()
        self._vad_user_speaking = False

    async def cleanup(self):
        await super().cleanup()
        await self._turn_analyzer.cleanup()

    async def process_frame(self, frame: Frame):
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, SpeechControlParamsFrame):
            await self._handle_speech_control_params(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)

    async def _start(self, frame: StartFrame):
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(
            SpeechControlParamsFrame, turn_params=self._turn_analyzer.params
        )

    async def _handle_speech_control_params(self, frame: SpeechControlParamsFrame):
        self._turn_analyzer.update_vad_start_secs(frame.vad_params.start_secs)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        if state == EndOfTurnState.COMPLETE:
            await self.trigger_user_turn_stopped()

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        self._vad_user_speaking = False

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()

        if prediction:
            await self.push_frame(MetricsFrame(data=[prediction]))

        if state == EndOfTurnState.COMPLETE:
            await self.trigger_user_turn_stopped()
