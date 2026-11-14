"""Synthetic demo data — generic domains, no real customer content.

A small golden set used by scripts/demo.py and the policy tests: each case
pairs a verbose prompt (with filler and repeated context) with the keywords
a good optimization must preserve.
"""
from __future__ import annotations

from tokenopt.dedup import ContextBlock
from tokenopt.evaluation import GoldenCase


def golden_set() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="refund-policy",
            prompt=(
                "Summarize the refund policy for a customer who bought a wireless "
                "headset. The customer wants to know the return window, whether a "
                "refund goes to the original payment method, and if restocking fees "
                "apply. The policy states a 30-day return window, refunds to the "
                "original payment method within 5 business days, and no restocking "
                "fee for unopened items. Please keep the answer concise and factual."
            ),
            context=[
                ContextBlock(id="c1", text=(
                    "Returns are accepted within 30 days of delivery for most items."),
                    source="policy-doc", score=0.95),
                ContextBlock(id="c2", text=(
                    "Returns are accepted within 30 days of delivery for most items."),
                    source="policy-doc-mirror", score=0.90),
                ContextBlock(id="c3", text=(
                    "Refunds are issued to the original payment method within 5 "
                    "business days. No restocking fee applies to unopened items."),
                    source="policy-doc", score=0.92),
            ],
            expected_keywords={"refund", "30-day", "payment", "restocking", "business"},
        ),
        GoldenCase(
            id="shipping-sla",
            prompt=(
                "A support agent needs the shipping service levels for standard and "
                "express delivery. Standard delivery takes 3 to 5 business days and "
                "costs 4.99. Express delivery takes 1 to 2 business days and costs "
                "12.99. Orders placed before 2 PM ship the same day. Please list "
                "the two options with their delivery times and prices."
            ),
            context=[
                ContextBlock(id="s1", text=(
                    "Standard shipping: 3-5 business days, $4.99. Express shipping: "
                    "1-2 business days, $12.99."),
                    source="shipping-guide", score=0.96),
                ContextBlock(id="s2", text=(
                    "Orders placed before 2 PM local time ship the same business day."),
                    source="shipping-guide", score=0.88),
            ],
            expected_keywords={"standard", "express", "business", "4.99", "12.99"},
        ),
        GoldenCase(
            id="warranty-claim",
            prompt=(
                "Explain how to file a warranty claim for a portable speaker. The "
                "warranty covers manufacturing defects for 12 months from purchase. "
                "The customer needs proof of purchase and the serial number. Claims "
                "are reviewed within 3 business days and approved claims receive a "
                "replacement unit. Accidental damage is not covered by the warranty."
            ),
            context=[
                ContextBlock(id="w1", text=(
                    "The 12-month warranty covers manufacturing defects only. "
                    "Accidental damage is excluded."),
                    source="warranty-doc", score=0.94),
                ContextBlock(id="w2", text=(
                    "To file a claim, provide proof of purchase and the serial number. "
                    "Reviews complete within 3 business days."),
                    source="warranty-doc", score=0.91),
            ],
            expected_keywords={"warranty", "12-month", "serial", "purchase", "replacement"},
        ),
        GoldenCase(
            id="inventory-alert",
            prompt=(
                "Draft an inventory alert for the warehouse team. The wireless "
                "keyboard SKU is below its reorder point of 50 units, with 32 units "
                "on hand. The supplier lead time is 7 days and the recommended "
                "reorder quantity is 200 units. The alert should state the SKU, the "
                "current stock, the reorder point, and the recommended action."
            ),
            context=[
                ContextBlock(id="i1", text=(
                    "SKU WK-2041 wireless keyboard: 32 units on hand, reorder point 50, "
                    "supplier lead time 7 days."),
                    source="inventory-feed", score=0.97),
            ],
            expected_keywords={"keyboard", "reorder", "32", "200", "supplier"},
        ),
    ]


def verbose_support_prompt() -> str:
    """A long, filler-heavy support prompt used by the demo."""
    return (
        "Hello, I hope you are doing very well today. I am writing to you "
        "because I have a question that I really need some help with, and I "
        "was hoping you could assist me. Basically, the situation is that I "
        "purchased a wireless headset last month, and I am now wondering about "
        "the refund policy. To be perfectly clear, I would like to know the "
        "return window, whether a refund goes to the original payment method, "
        "and if any restocking fees apply. The policy states a 30-day return "
        "window, refunds to the original payment method within 5 business "
        "days, and no restocking fee for unopened items. As I mentioned "
        "before, I just want a concise and factual answer about the refund "
        "policy for the wireless headset I bought."
    )


def default_policies() -> list[dict]:
    """Two starter policies for the demo and tests."""
    return [
        {
            "name": "balanced",
            "version": 1,
            "params": {"target_ratio": 0.6, "dedup_threshold": 0.85,
                       "max_tokens": 8_000},
            "description": "Moderate compression, quality-first.",
        },
        {
            "name": "aggressive",
            "version": 1,
            "params": {"target_ratio": 0.4, "dedup_threshold": 0.80,
                       "max_tokens": 4_000},
            "description": "Heavy compression for cost-sensitive tenants.",
        },
    ]
