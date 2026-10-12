import random
from typing import List
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Car(BaseModel):
    make: str
    model: str
    year: int
    colour: str
    mileage: int
    price: float
    fuelType: str        # e.g. "Petrol", "Diesel", "Electric", "Hybrid"
    transmission: str    # e.g. "Manual", "Automatic"
    available: bool
    image_url: str


class CarouselItem(BaseModel):
    id: str
    imageUrl: str | None = None
    content: str
    actionLabel: str | None = None
    actionUrl: str | None = None


class CarInventory(BaseModel):
    cars: List[Car]
    items: List[CarouselItem]
    type: str


class ThemeResponse(BaseModel):
    type: str = "theme"
    name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def car_to_carousel_item(car: Car) -> CarouselItem:
    return CarouselItem(
        id=f"{car.make}-{car.model}-{car.year}".lower().replace(" ", "-"),
        imageUrl=car.image_url,
        content=f"## {car.year} {car.make} {car.model}\n**£{car.price:,.0f}** · {car.colour} · {car.fuelType} · {car.transmission} · {car.mileage:,} km",
        actionLabel="Enquire Now",
    )


def get_car_by_id(car_id: str) -> Car | None:
    for car in _SAMPLE_CARS:
        slug = f"{car.make}-{car.model}-{car.year}".lower().replace(" ", "-")
        if slug == car_id.lower():
            return car
    return None


# ---------------------------------------------------------------------------
# Sample inventory
# ---------------------------------------------------------------------------

_SAMPLE_CARS = [
    Car(make="Toyota", model="Corolla", year=2021, colour="Silver", mileage=32000, price=18500.0, fuelType="Petrol", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/2018_Toyota_Corolla_%28MZEA12R%29_Ascent_Sport_hatchback_%282018-11-02%29_01.jpg/330px-2018_Toyota_Corolla_%28MZEA12R%29_Ascent_Sport_hatchback_%282018-11-02%29_01.jpg"),
    Car(make="Ford", model="Focus", year=2020, colour="Blue", mileage=45000, price=14200.0, fuelType="Petrol", transmission="Manual", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/2018_Ford_Focus_Titanium_X_1.0.jpg/330px-2018_Ford_Focus_Titanium_X_1.0.jpg"),
    Car(make="BMW", model="3 Series", year=2022, colour="Black", mileage=18000, price=34000.0, fuelType="Diesel", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/2019_BMW_318d_SE_Automatic_2.0_Front.jpg/330px-2019_BMW_318d_SE_Automatic_2.0_Front.jpg"),
    Car(make="Tesla", model="Model 3", year=2023, colour="White", mileage=8000, price=42000.0, fuelType="Electric", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/2019_Tesla_Model_3_Performance_AWD_Front.jpg/330px-2019_Tesla_Model_3_Performance_AWD_Front.jpg"),
    Car(make="Volkswagen", model="Golf", year=2019, colour="Red", mileage=60000, price=12800.0, fuelType="Petrol", transmission="Manual", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/2013_Volkswagen_Golf_SE_BlueMotion_Technology_1.4_Front.jpg/330px-2013_Volkswagen_Golf_SE_BlueMotion_Technology_1.4_Front.jpg"),
    Car(make="Honda", model="Civic", year=2021, colour="Grey", mileage=28000, price=19500.0, fuelType="Hybrid", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/2022_Honda_Civic_Touring_in_Lunar_Silver_Metallic%2C_Front_Left%2C_05-10-2022.jpg/330px-2022_Honda_Civic_Touring_in_Lunar_Silver_Metallic%2C_Front_Left%2C_05-10-2022.jpg"),
    Car(make="Audi", model="A4", year=2022, colour="White", mileage=22000, price=31000.0, fuelType="Diesel", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Audi_A4_B9_sedans_%28FL%29_1X7A2441.jpg/330px-Audi_A4_B9_sedans_%28FL%29_1X7A2441.jpg"),
    Car(make="Nissan", model="Leaf", year=2023, colour="Blue", mileage=5000, price=27000.0, fuelType="Electric", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Nissan_Leaf_%28ZE2%29_autoMOBIL_T%C3%BCbingen_2025_DSC_2752.jpg/330px-Nissan_Leaf_%28ZE2%29_autoMOBIL_T%C3%BCbingen_2025_DSC_2752.jpg"),
    Car(make="Hyundai", model="Tucson", year=2020, colour="Green", mileage=51000, price=16500.0, fuelType="Hybrid", transmission="Automatic", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/2022_Hyundai_Tucson_Preferred%2C_Front_Right%2C_05-24-2021.jpg/330px-2022_Hyundai_Tucson_Preferred%2C_Front_Right%2C_05-24-2021.jpg"),
    Car(make="Mazda", model="CX-5", year=2021, colour="Red", mileage=37000, price=22000.0, fuelType="Petrol", transmission="Manual", available=True, image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/2024_Mazda_CX-5_2.5_S_Select_in_Platinum_Quartz_Metallic%2C_front_right.jpg/330px-2024_Mazda_CX-5_2.5_S_Select_in_Platinum_Quartz_Metallic%2C_front_right.jpg"),
]
