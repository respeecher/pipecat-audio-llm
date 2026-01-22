from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy


class AudioUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    async def _maybe_trigger_user_turn_stopped(self):
        if self._turn_complete:
            await self.trigger_user_turn_stopped()
