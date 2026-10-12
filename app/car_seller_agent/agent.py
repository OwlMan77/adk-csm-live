import random

from google.adk.agents import Agent

from .data import CarInventory, car_to_carousel_item, _SAMPLE_CARS
from .subagents.car_detail_agent import car_detail_agent


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_random_cars() -> dict:
    """Returns a random selection of 3 cars from the dealership inventory."""
    selection = random.sample(_SAMPLE_CARS, 3)
    return CarInventory(
        cars=selection,
        items=[car_to_carousel_item(car) for car in selection],
        type="carousel",
    ).model_dump()


# ---------------------------------------------------------------------------
# Inventory agent
# ---------------------------------------------------------------------------

car_inventory_agent = Agent(
    name="car_inventory_agent",
    model="gemini-2.5-flash",
    description="Returns a carousel of 3 random available cars from the dealership inventory.",
    instruction="""
    You are a car dealership inventory agent.

    When asked for available cars, call the get_random_cars tool. It returns a
    CarInventory containing:
      - cars: full car details
      - items: the same cars mapped to the chat carousel format (id, title,
               description, imageUrl, price, actionLabel)
      - type: always "carousel"

    Return the tool result directly — do not summarize or reformat it, and do
    not fabricate car data yourself.
    """,
    tools=[get_random_cars],
    output_schema=CarInventory,
    output_key="inventory",
)


# ---------------------------------------------------------------------------
# Root orchestrator
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="car_seller_agent",
    model="gemini-2.5-flash",
    description="Car dealership assistant that shows inventory and explains car specs.",
    instruction="""
    You are a car dealership assistant. Route the user's request to the right
    specialist:

    - If the user asks to browse, see, or list cars → transfer to car_inventory_agent.
    - If the user selects a specific car or asks for details/specs (providing
      a car ID like 'toyota-corolla-2021') check 'inventory' cars for the details and list them the details that match the id provide.

    Do not handle requests yourself — always delegate to the appropriate agent.
    """,
    sub_agents=[car_inventory_agent, car_detail_agent],
)
