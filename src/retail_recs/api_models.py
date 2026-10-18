"""Shared HTTP response types; importing these never loads the ML runtime."""
from pydantic import BaseModel, FiniteFloat


class ProductRecommendation(BaseModel):
    product_id: int
    score: FiniteFloat
    description: str | None = None
    category: str | None = None
    department: str | None = None


class Recommendations(BaseModel):
    household_id: int
    snapshot_cutoff: int
    model_version: str
    recommendations: list[ProductRecommendation]


class Health(BaseModel):
    status: str
    model_version: str
    snapshot_cutoff: int
    households: int
    candidate_rows: int
