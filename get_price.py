from dotenv import load_dotenv
import os
from src.model.bot import ThucChienAIBot
load_dotenv()


if __name__ == "__main__":
   api_key = os.getenv("THUC_CHIEN_API_KEY")
   bot = ThucChienAIBot(api_key)
   key_info = bot.get_key_info()
   total_spent = 50
   if key_info:
           print("Thông tin Key:")
           print(f"  - Tên Key: {key_info['info']['key_name']}")
           print(f"  - Chi tiêu: ${key_info['info']['spend']:.6f}")
           print(f"  - Hạn mức chi tiêu: ${total_spent - key_info['info']['spend']:.6f}")
           print(f"  - Models được phép: {key_info['info']['models']}")

