"""SQLAlchemy ORM models (SQLite for the MVP; switch DATABASE_URL to PostgreSQL for production)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Market(Base):
    __tablename__ = "markets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Candle(Base):
    __tablename__ = "candles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    tick_volume: Mapped[float] = mapped_column(Float, default=0.0)
    spread: Mapped[float] = mapped_column(Float, default=0.0)
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
                      Index("ix_candle_lookup", "symbol", "timeframe", "timestamp"))


class FeatureRow(Base):
    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    values: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_feature"),)


class Prediction(Base):
    """Immutable: UPDATE statements are rejected (see listener below). Outcomes live in prediction_results."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # candle close time
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    horizon: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    probability_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_down: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_kind: Mapped[str] = mapped_column(String(32), default="raw model probability")
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_agreement: Mapped[str | None] = mapped_column(String(8), nullable=True)
    edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "timestamp", "model_version", name="uq_prediction"),)


@event.listens_for(Prediction, "before_update")
def _block_prediction_update(mapper, connection, target):  # pragma: no cover - needs SQLAlchemy runtime
    raise RuntimeError("predictions are immutable; write outcomes to prediction_results")


class PredictionResult(Base):
    __tablename__ = "prediction_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True, index=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actual_price: Mapped[float] = mapped_column(Float)
    actual_return: Mapped[float] = mapped_column(Float)
    actual_direction: Mapped[str] = mapped_column(String(4))
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error: Mapped[float | None] = mapped_column(Float, nullable=True)  # expected_return - actual_return


class AnomalyRow(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    anomaly_score: Mapped[float] = mapped_column(Float)
    detected: Mapped[bool] = mapped_column(Boolean)
    types: Mapped[list] = mapped_column(JSON)
    details: Mapped[dict] = mapped_column(JSON)


class SimilarityRow(Base):
    __tablename__ = "similarity_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_similar: Mapped[int] = mapped_column(Integer)
    details: Mapped[dict] = mapped_column(JSON)


class MarketScoreRow(Base):
    __tablename__ = "market_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    market_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    components: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)  # full standard result
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_market_score"),)


class EventRow(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    priority: Mapped[float] = mapped_column(Float)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    signature: Mapped[dict] = mapped_column(JSON)  # values used for "has the situation changed?"
    why: Mapped[list] = mapped_column(JSON)


class PostRow(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True)
    post_type: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(12), default="pending", index=True)  # pending|sent|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    version: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    meta: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "version", name="uq_model_version"),)


class ModelMetricRow(Base):
    __tablename__ = "model_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    scope: Mapped[str] = mapped_column(String(64), index=True)  # e.g. overall, symbol:XAUUSD, regime:TREND_UP
    kind: Mapped[str] = mapped_column(String(32), index=True)  # live_performance | walk_forward | feature_importance
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON)


class ModelArtifactRow(Base):
    """Model files kept in the database (base64, chunked) so they survive stateless CI runs (GitHub Actions + Turso).

    kind: "bundle" (joblib, compressed), "meta" (json) or "active" (json pointer to the live version).
    """
    __tablename__ = "model_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    version: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(12))
    part: Mapped[int] = mapped_column(Integer, default=0)
    data: Mapped[str] = mapped_column(Text)  # base64 text: the libSQL driver has no BLOB-parameter support
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "version", "kind", "part", name="uq_model_artifact"),)
