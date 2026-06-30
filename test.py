from dotenv import load_dotenv
load_dotenv()
from src.loader import load
from src.tool import answer_with_tool

_, business_info = load("data")
print(answer_with_tool("what is the mileage rate?", business_info))
print(answer_with_tool("what's the HR email?", business_info))