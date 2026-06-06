"""Future-blind time-windowed data container."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class DataWindow:
    """Token event history visible only up to the current simulation time."""

    token: str
    start_time: datetime
    end_time: datetime
    current_time: datetime
    data: pd.DataFrame

    def get_visible_data(self) -> pd.DataFrame:
        if "timestamp" not in self.data.columns:
            return self.data.iloc[: self._current_row_index() + 1]
        return self.data[self.data["timestamp"] <= self.current_time]

    def advance_time(self, days: int = 1) -> None:
        self.current_time += timedelta(days=days)
        if self.current_time > self.end_time:
            self.current_time = self.end_time

    def has_more_data(self) -> bool:
        return self.current_time < self.end_time

    def _current_row_index(self) -> int:
        if "timestamp" not in self.data.columns:
            return len(self.data) - 1
        visible = self.data[self.data["timestamp"] <= self.current_time]
        return max(0, len(visible) - 1)