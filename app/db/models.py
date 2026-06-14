"""SQLAlchemy ORM models for the catalog, RFQs, line items, and audit log."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CatalogItem(Base):
    """One SKU from the distributor's catalog CSV."""

    __tablename__ = "catalog_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)
    type: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str] = mapped_column(String)
    diameter: Mapped[str] = mapped_column(String)
    thread_pitch_or_tpi: Mapped[str] = mapped_column(String)
    length: Mapped[str] = mapped_column(String)
    grade_or_class: Mapped[str] = mapped_column(String)
    material: Mapped[str] = mapped_column(String)
    finish_coating: Mapped[str] = mapped_column(String)
    head_type: Mapped[str] = mapped_column(String)
    drive_type: Mapped[str] = mapped_column(String)
    standard: Mapped[str] = mapped_column(String)
    thread_direction: Mapped[str] = mapped_column(String)
    units: Mapped[str] = mapped_column(String, index=True)
    price_usd: Mapped[float] = mapped_column(Float)


class RFQ(Base):
    """A single ingested RFQ (one file/text blob, may contain many line items)."""

    __tablename__ = "rfqs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String)  # e.g. "cli", or original filename
    raw_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="draft")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    line_items: Mapped[list["LineItem"]] = relationship(back_populates="rfq")


class LineItem(Base):
    """A single extracted RFQ line item, with match + pricing results."""

    __tablename__ = "line_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rfq_id: Mapped[int] = mapped_column(ForeignKey("rfqs.id"))
    line_number: Mapped[int] = mapped_column(Integer)

    raw_text: Mapped[str] = mapped_column(Text)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str] = mapped_column(String, default="")
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    target_date: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    matched_sku: Mapped[str | None] = mapped_column(String, nullable=True)
    match_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_alternatives: Mapped[list] = mapped_column(JSON, default=list)
    match_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_bucket: Mapped[str | None] = mapped_column(String, nullable=True)

    unit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ext_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    lead_time: Mapped[str | None] = mapped_column(String, nullable=True)
    human_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    rfq: Mapped["RFQ"] = relationship(back_populates="line_items")


class AuditLogEntry(Base):
    """Append-only record of every extraction, match, price, and human edit."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    rfq_id: Mapped[int | None] = mapped_column(ForeignKey("rfqs.id"), nullable=True)
    line_item_id: Mapped[int | None] = mapped_column(ForeignKey("line_items.id"), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
