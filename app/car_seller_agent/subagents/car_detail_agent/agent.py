from google.adk.agents import Agent

from app.car_seller_agent.data import Car, get_car_by_id as _lookup


_COLOUR_TO_THEME = {
    "black": "dark", "graphite": "dark", "charcoal": "dark",
    "white": "light", "pearl": "light", "ivory": "light", "silver": "light", "grey": "light", "gray": "light",
    "blue": "ocean", "navy": "ocean", "teal": "ocean", "cyan": "ocean", "aqua": "ocean",
    "red": "sunset", "orange": "sunset", "yellow": "sunset", "gold": "sunset", "crimson": "sunset", "burgundy": "sunset",
    "green": "dark", "purple": "dark",
}


def set_theme(colour: str) -> dict:
    """
    Returns a theme event payload for the chat UI matching the given car colour.
    Valid themes: dark, light, ocean, sunset. Defaults to "light" for unmapped colours.
    """
    theme_name = _COLOUR_TO_THEME.get(colour.lower(), "light")
    return {"type": "theme", "name": theme_name}


def lookup_car_specs(car_id: str) -> dict:
    """
    Returns the full specifications of a car given its carousel ID
    (e.g. 'toyota-corolla-2021').  Returns an error dict if not found.
    """
    car: Car | None = _lookup(car_id)
    if car is None:
        return {"error": f"No car found with id '{car_id}'"}
    return car.model_dump()


car_detail_agent = Agent(
    name="car_detail_agent",
    model="gemini-2.5-flash",
    description=(
        "Explains the full specifications of a specific car when the user "
        "selects one from the carousel, identified by its car ID."
    ),
    instruction="""
    You are a car specification expert at a dealership.

    When the user selects a car (providing a car ID such as 'toyota-corolla-2021'):
    1. Call lookup_car_specs with the car ID to retrieve the full details.
    2. Call set_theme with the car's colour (e.g. "Silver") to get the theme payload.
    3. Output the set_theme result as a raw JSON string on its own line FIRST,
       then output the car details in markdown below it.

    Present the specs in a clear, friendly way covering:
      - Make, model, and year
      - Colour
      - Fuel type and transmission
      - Mileage (in km)
      - Price (in GBP, formatted as £XX,XXX)
      - Availability

    Markdown format:
    # Year Make Model
    ## Colour
    <rest as a paragraph>

    If the car ID is not found, apologise and ask the user to select a car
    from the available inventory.

    Always use the tools — do not fabricate or guess car data.
    """,
    tools=[lookup_car_specs, set_theme],
)
