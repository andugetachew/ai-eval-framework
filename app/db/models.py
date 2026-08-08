import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending/running/completed/failed

    results: Mapped[list["EvalResult"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id"))
    item_id: Mapped[str] = mapped_column(String)
    variant: Mapped[str | None] = mapped_column(String, nullable=True)
    scorer_name: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool | None] = mapped_column(nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped["EvalRun"] = relationship(back_populates="results")