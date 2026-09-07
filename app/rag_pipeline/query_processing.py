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
        try:
            raw = self.client.generate_json("plan_query", query, {})
        except Exception:
            # Preserve experiment continuity when a provider returns malformed
            # JSON or is temporarily rate-limited; the original query remains
            # a valid fallback plan and the failure is visible in API logs.
            return QueryPlan(original_query=query)
        # Models may add explanatory fields; keep only the planner contract.
        plan = {"original_query": query,
                "rewritten_queries": raw.get("rewritten_queries", [])[:2],
                "sub_questions": raw.get("sub_questions", [])[:3],
                "requires_decomposition": bool(raw.get("requires_decomposition", False))}
        return QueryPlan.model_validate(plan)
