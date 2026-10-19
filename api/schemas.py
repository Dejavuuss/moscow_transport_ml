"""Pydantic-схемы API: валидация входа и автогенерируемая документация (Swagger на /docs)."""
from __future__ import annotations

from pydantic import BaseModel, Field

PERIOD = r"^\d{4}[Qq][1-4]$"


class HistoryPoint(BaseModel):
    period: str
    incoming: float
    closed: bool


class ForecastResponse(BaseModel):
    station: str
    line: str
    period: str = Field(..., description="Квартал прогноза, например 2026Q3")
    forecast: int = Field(..., description="Прогноз входов пассажиров за квартал")
    low: int = Field(..., description="Нижняя граница (10-й перцентиль ошибок на val)")
    high: int = Field(..., description="Верхняя граница (90-й перцентиль)")
    actual: float | None = Field(None, description="Факт, если квартал уже есть в данных (бэктест)")
    anchor_lag_quarters: int = Field(..., description="От какого прошлого квартала строился прогноз (4 = год назад)")
    model_version: str | None = None
    history: list[HistoryPoint]


class Observation(BaseModel):
    station: str = Field(..., examples=["Курская"])
    line: str | None = Field(None, examples=["Кольцевая линия"], description="Нужна для пересадочных станций")
    period: str = Field(..., pattern=PERIOD, examples=["2026Q2"])
    incoming: float = Field(..., ge=0, examples=[2500000])
    outgoing: float = Field(..., ge=0, examples=[2550000])


class Verdict(BaseModel):
    station: str
    line: str
    period: str
    actual: float
    expected: int | None = Field(None, description="Прогноз (None — нет истории, например станция давно закрыта)")
    change: float | None = Field(None, description="Факт / прогноз − 1")
    status: str = Field(..., description="ok | anomaly")
    anomaly_type: str = Field(..., description="ok | closure | partial_closure | station_drop | station_surge | "
                                                "line_event | imbalance")
    severity: str = Field(..., description="none | warning | critical")
    explanation: str
    context_known: bool = Field(..., description="Известен ли контекст города/линии за этот квартал")
    scores: dict[str, float | None]
