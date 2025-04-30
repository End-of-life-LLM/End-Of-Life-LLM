from model import Model
from supporting_model import Supporting_Model
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = Model(api_key=api_key)
supporting_model = Supporting_Model(api_key=api_key)


keep_going = True
while keep_going:
    keep_going = model.message_manager()