"""MySQL persistence for API query records and user feedback."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_query: Mapped[Optional[str]] = mapped_column(Text)
    route_type: Mapped[Optional[str]] = mapped_column(String(20))
    topic: Mapped[Optional[str]] = mapped_column(String(120))
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cache_type: Mapped[Optional[str]] = mapped_column(String(20))
    similarity: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query_logs.id"), nullable=False, index=True)
    rating: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Database:
    def __init__(self, database_url: str, connect_timeout: int = 5):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": connect_timeout},
        )
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        with self.engine.connect():
            Base.metadata.create_all(self.engine)

    def create_query(self, **values) -> QueryLog:
        with self.session_factory() as session:
            query = QueryLog(**values)
            session.add(query)
            session.commit()
            session.refresh(query)
            return query

    def create_feedback(self, query_id: int, rating: str, comment: Optional[str]) -> Feedback:
        with self.session_factory() as session:
            if session.scalar(select(QueryLog.id).where(QueryLog.id == query_id)) is None:
                raise LookupError("query_id does not exist")
            feedback = Feedback(query_id=query_id, rating=rating, comment=comment)
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            return feedback
