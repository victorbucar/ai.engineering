"""Product-description extraction pipeline under test.

Same mechanism as ../code/structured.py (Claude + output_config json_schema), but
the schema is a Pydantic model and the input text is actually passed through.
"""
import json
from typing import Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "claude-haiku-5-5"

Category = Literal[
    "audio", "computers", "phones", "wearables", "gaming",
    "home", "kitchen", "fitness", "clothing", "toys", "other",
]


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Product name/model without the brand, no marketing taglines")
    brand: str | None = Field(description="Manufacturer or brand, null if not stated")
    category: Category
    price: float | None = Field(description="Current selling price as a plain number")
    currency: str | None = Field(description="ISO 4217 code, e.g. USD")
    in_stock: bool | None = Field(description="Availability, null if not mentioned")
    colors: list[str] = Field(description="Lowercase English color names")
    weight_grams: int | None = Field(description="Product weight in grams, rounded to nearest integer")
    warranty_months: int | None = Field(description="Warranty length in months")
    rating: float | None = Field(description="Average rating normalized to a 0-5 scale")
    release_date: str | None = Field(description="YYYY-MM-DD, only when a full date is given")


FIELDS = list(Product.model_fields)

SYSTEM_PROMPT = """You extract structured data from a product description. Rules:
- name: the product's name/model as written in its title or first mention, with the brand removed and without taglines or marketing copy, do not drop nouns.
- brand: the manufacturer or brand. null if not stated.
- category: the single best fit from the allowed values; "other" if none fits.
- price: the price a customer pays now, as a plain number (after any discount; ignore original/"was" prices, competitor prices, shipping and bundle-deal prices). null if no price is given, if the text gives only a range or a "from/starting at" price, or if several variants have different prices.
- currency: ISO 4217 code of the prices mentioned ("$" means USD unless the text says otherwise). null only if no price is mentioned at all.
- in_stock: true if the text says in stock, available now, back in stock, or ships within 2 days. false if sold out, out of stock, pre-order, coming soon, backordered, or shipping takes longer than 2 days. null if availability is not mentioned.
- colors: color/finish names mentioned for the product, lowercase, in English, in the order given, do not make up colors or change the original if uncertain. [] if none.
- weight_grams: the product's own weight (not shipping weight) converted to grams and rounded to the nearest integer. null if not stated.
- warranty_months: warranty length in months (1 year = 12, 90 days = 3). null if none is stated, if it is "lifetime", or if the text says there is no warranty.
- rating: the average customer/reviewer rating converted to a 0-5 scale (e.g. 9/10 -> 4.5, 80/100 -> 4.0). null if not stated.
- release_date: YYYY-MM-DD, only if the text gives a full day, month and year of release or launch. null otherwise (restock dates are not release dates).
Never guess: use null when the text does not support a value."""


def extract(text: str, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None) -> dict:
    """Run the pipeline on one description. Returns raw model text plus metadata."""
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        output_config={
            "format": {"type": "json_schema", "schema": Product.model_json_schema()}
        },
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    return {
        "raw": raw,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


if __name__ == "__main__":
    out = extract("Hario V60 ceramic coffee dripper, $26. White. Back in stock!")
    print(json.dumps(json.loads(out["raw"]), indent=2))
