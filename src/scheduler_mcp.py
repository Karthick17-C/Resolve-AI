import json
import os
from mcp.server.fastmcp import FastMCP

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8001"))

mcp = FastMCP("CustomerCallScheduler", host=HOST, port=PORT)

BOOKED_SLOTS = ["2026-11-25 10:00", "2026-11-25 14:00"]


@mcp.tool()
def check_availability(date_time: str) -> str:
    """Checks if a support technician is available at YYYY-MM-DD HH:MM."""
    if date_time in BOOKED_SLOTS:
        return f"Slot {date_time} is ALREADY BOOKED. Suggest another hour."
    return f"Slot {date_time} is AVAILABLE."


@mcp.tool()
def book_customer_call(customer_name: str, email: str, date_time: str, issue_summary: str) -> str:
    """Books a customer support call after verifying slot availability."""
    if date_time in BOOKED_SLOTS:
        return f"Booking failed: {date_time} is already taken."
    BOOKED_SLOTS.append(date_time)
    return json.dumps({
        "status": "Confirmed",
        "customer": customer_name,
        "email": email,
        "scheduled_time": date_time,
        "issue": issue_summary,
    })


if __name__ == "__main__":
    mcp.run(transport="streamable-http")