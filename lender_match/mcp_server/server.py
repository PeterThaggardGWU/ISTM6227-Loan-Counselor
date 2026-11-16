"""
mcp_server/server.py
────────────────────────────────────────────────────────────────────────────────
Loan Arranger — MCP Server
ISTM6218 Group Project

Exposes the Loan Arranger recommendation service as a Claude Desktop tool
via the Model Context Protocol (stdio transport).

This server contains NO LLM logic. It is a thin adapter:
  1. Receives a structured tool call from Claude
  2. Validates and forwards it to the FastAPI inference service
  3. Returns the JSON result to Claude
  4. Claude narrates the result to the user in natural language

Tools exposed:
  get_lender_recommendations  — main recommendation tool
  get_lenders_in_area         — list lenders active in a geography (exploration)
  check_service_health        — verify the inference service is reachable

Claude Desktop configuration (claude_desktop_config.json):
  {
    "mcpServers": {
      "loan-arranger": {
        "command": "python",
        "args": ["C:/path/to/mcp_server/server.py"],
        "env": {
          "LOAN_ARRANGER_SERVICE_URL": "http://localhost:8000"
        }
      }
    }
  }

Usage:
  python mcp_server/server.py
  LOAN_ARRANGER_SERVICE_URL=http://localhost:8000 python mcp_server/server.py
────────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ── Configuration ─────────────────────────────────────────────────────────────
SERVICE_URL     = os.environ.get("LOAN_ARRANGER_SERVICE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30.0   # seconds

# ── Logging — stderr only (stdout is reserved for MCP stdio transport) ────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [MCP]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── MCP Server instance ───────────────────────────────────────────────────────
server = Server("loan-arranger")


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [

        Tool(
            name="get_lender_recommendations",
            description=(
                "Recommends mortgage lenders for a borrower based on their "
                "financial profile and target property location. Returns a ranked "
                "list of lenders active in the specified area with predicted "
                "approval probabilities derived from historical HMDA data (2019-2024). "
                "Call this when a user asks where to apply for a mortgage, which "
                "lenders to approach, wants to know their best chances of approval, "
                "or requests personalized lender recommendations. "
                "The default model (variant B) is ECOA-compliant and does not use "
                "race, ethnicity, sex, or age as inputs. "
                "Model variant A includes demographic features and is available for "
                "academic comparison only if the user explicitly opts in."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter US state code, e.g. 'TX', 'VA', 'CA'.",
                    },
                    "county_code": {
                        "type": "string",
                        "description": (
                            "5-digit FIPS county code e.g. '48029' for Bexar County TX. "
                            "Optional — provide this OR county_name/city instead."
                        ),
                    },
                    "county_name": {
                        "type": "string",
                        "description": (
                            "County name e.g. 'Bexar County' or just 'Bexar'. "
                            "The service resolves this to a FIPS code automatically. "
                            "Use when the user mentions a county name."
                        ),
                    },
                    "city": {
                        "type": "string",
                        "description": (
                            "City name e.g. 'San Antonio'. "
                            "The service resolves this to the containing county automatically. "
                            "Use when the user mentions a city rather than a county."
                        ),
                    },
                    "loan_amount": {
                        "type": "number",
                        "description": "Requested loan amount in dollars e.g. 350000.",
                    },
                    "income": {
                        "type": "number",
                        "description": "Annual gross income in dollars e.g. 120000.",
                    },
                    "property_value": {
                        "type": "number",
                        "description": "Estimated property value in dollars. Optional.",
                    },
                    "debt_to_income_ratio": {
                        "type": "number",
                        "description": "DTI ratio as a percentage e.g. 35.0. Optional.",
                    },
                    "loan_purpose": {
                        "type": "integer",
                        "description": "1=Purchase (default) 2=Home improvement 31=Cash-out refi 32=No-cash refi 4=Other.",
                        "default": 1,
                    },
                    "loan_type": {
                        "type": "integer",
                        "description": "1=Conventional (default) 2=FHA 3=VA 4=USDA.",
                        "default": 1,
                    },
                    "lien_status": {
                        "type": "integer",
                        "description": "1=First lien (default) 2=Subordinate.",
                        "default": 1,
                    },
                    "occupancy_type": {
                        "type": "integer",
                        "description": "1=Principal residence (default) 2=Investment 3=Second home.",
                        "default": 1,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of recommendations to return (default 5, max 20).",
                        "default": 5,
                    },
                    "model_variant": {
                        "type": "string",
                        "description": (
                            "'B' = default ECOA-compliant model (no demographics). "
                            "'A' = includes demographics, only if user explicitly requests it."
                        ),
                        "default": "B",
                    },
                    "min_applications": {
                        "type": "integer",
                        "description": "Min historical applications for a lender to appear (default 50).",
                        "default": 50,
                    },
                    "derived_race": {
                        "type": "string",
                        "description": "Self-reported race. Only used with model_variant='A'.",
                    },
                    "derived_ethnicity": {
                        "type": "string",
                        "description": "Self-reported ethnicity. Only used with model_variant='A'.",
                    },
                    "derived_sex": {
                        "type": "string",
                        "description": "Self-reported sex. Only used with model_variant='A'.",
                    },
                    "applicant_age": {
                        "type": "string",
                        "description": "Age bracket e.g. '25-34'. Only used with model_variant='A'.",
                    },
                },
                "required": ["state_code", "loan_amount", "income"],
            },
        ),

        Tool(
            name="get_lenders_in_area",
            description=(
                "Lists mortgage lenders that have historically operated in a given "
                "state or county, sorted by volume of applications. "
                "Use this when a user wants to know which lenders are active in "
                "their market, or to explore options before requesting a full "
                "recommendation. Does not predict approval likelihood — "
                "use get_lender_recommendations for ranked recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state_code": {
                        "type": "string",
                        "description": "Two-letter state code e.g. 'TX'.",
                    },
                    "county_code": {
                        "type": "string",
                        "description": "Optional 5-digit FIPS county code.",
                    },
                    "min_applications": {
                        "type": "integer",
                        "description": "Minimum application threshold (default 100).",
                        "default": 100,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lenders to return (default 20, max 100).",
                        "default": 20,
                    },
                },
                "required": ["state_code"],
            },
        ),

        Tool(
            name="check_service_health",
            description=(
                "Checks whether the Loan Arranger inference service is running and "
                "reports which model artifacts are loaded. Call this if the user "
                "reports the service is unavailable, or to confirm readiness before "
                "making a recommendation request."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool call handler
# ─────────────────────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:

        if name == "get_lender_recommendations":
            log.info(
                f"get_lender_recommendations  "
                f"state={arguments.get('state_code')}  "
                f"county={arguments.get('county_code', 'n/a')}  "
                f"loan=${arguments.get('loan_amount', 0):,.0f}  "
                f"income=${arguments.get('income', 0):,.0f}  "
                f"variant={arguments.get('model_variant', 'B')}"
            )
            try:
                resp = await client.post(f"{SERVICE_URL}/recommend", json=arguments)
            except httpx.ConnectError:
                return _error(
                    f"Cannot reach the Loan Arranger service at {SERVICE_URL}. "
                    "Please start it with: "
                    "uvicorn main:app --host 0.0.0.0 --port 8000"
                )
            except httpx.TimeoutException:
                return _error(
                    "The recommendation request timed out. "
                    "The service may be loading — please wait a moment and retry."
                )

            if resp.status_code == 404:
                return _error(resp.json().get("detail", "No lenders found."))
            if resp.status_code == 400:
                return _error(resp.json().get("detail", "Invalid request parameters."))
            if resp.status_code != 200:
                return _error(f"Service error HTTP {resp.status_code}: {resp.text[:300]}")

            return [TextContent(type="text", text=resp.text)]

        elif name == "get_lenders_in_area":
            state = arguments.get("state_code", "").upper()
            params = {k: v for k, v in {
                "county_code":      arguments.get("county_code"),
                "min_applications": arguments.get("min_applications", 100),
                "limit":            min(arguments.get("limit", 20), 100),
            }.items() if v is not None}

            log.info(f"get_lenders_in_area  state={state}  params={params}")
            try:
                resp = await client.get(f"{SERVICE_URL}/lenders/{state}", params=params)
            except httpx.ConnectError:
                return _error(f"Cannot reach service at {SERVICE_URL}.")
            if resp.status_code != 200:
                return _error(f"Service error HTTP {resp.status_code}: {resp.text[:300]}")
            return [TextContent(type="text", text=resp.text)]

        elif name == "check_service_health":
            log.info("check_service_health")
            try:
                resp = await client.get(f"{SERVICE_URL}/health")
                return [TextContent(type="text", text=resp.text)]
            except httpx.ConnectError:
                return [TextContent(type="text", text=json.dumps({
                    "status":     "unreachable",
                    "error":      f"Cannot connect to {SERVICE_URL}",
                    "suggestion": "Start the service: uvicorn main:app --host 0.0.0.0 --port 8000",
                }))]

        else:
            return _error(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _error(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": message}))]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    log.info(f"Loan Arranger MCP server starting  (service: {SERVICE_URL})")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())
