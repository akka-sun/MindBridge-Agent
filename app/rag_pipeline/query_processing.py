from __future__ import annotations

from .models import QueryPlan


class QueryPlanner:
    def __init__(self, client=None):
        self.client = client

    def plan(self, query: str, history: str = "") -> QueryPlan:
        if not query.strip():
            return QueryPlan(original_query=query or "空查询")
        if not self.client:
            return QueryPlan(original_query=query)
        raw = self.client.generate_json("plan_query", query, {})
        raw["original_query"] = query
        return QueryPlan.model_validate(raw)

