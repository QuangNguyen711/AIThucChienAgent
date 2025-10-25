# File: src/nodes/text2img.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import base64
import os
import time


SYSTEM_PROMPT_VI = """
Bạn là một trợ lý hữu ích hãy tạo hình ảnh theo yêu cầu của người dùng.
"""


SYSTEM_PROMPT_EN = """
You are a helpful assistant, help generate image according to user demand.
"""


def text2img(state: State, config: RunnableConfig) -> State:
   """NODE: Tạo ảnh dựa trên yêu cầu."""
   print("--- Thực hiện Node: text2img ---")


   # --- LẤY CÁC THAM SỐ TỪ STATE ---
   question = state["t2i_question"]
   num_images = state.get("t2i_num_images", 1)
   aspect_ratio = state.get("t2i_aspect_ratio", None) # Mặc định "1:1"
   size = state.get("t2i_size", None)                  # Mặc định không có size
   # ----------------------------------
  
   bot = config["configurable"]["bot"]
  
   # --- GỌI API VỚI ĐẦY ĐỦ THAM SỐ ---
   print(f"Đang tạo ảnh với prompt: '{question}', Tỷ lệ: {aspect_ratio}, Kích thước: {size or 'Mặc định'}")
   response_dict = bot.generate_image(
       model=os.getenv("IMAGE_MODEL_NAME"),
       prompt=question,
       n=num_images,
       size=size,
       aspect_ratio=aspect_ratio,
   )
   # -----------------------------------


   if not response_dict or "data" not in response_dict:
       print("Lỗi: Không nhận được dữ liệu ảnh hợp lệ từ API.")
       state["t2i_output_path"] = "API call failed. No image generated."
       return state


   saved_paths = []
   for i, image_obj in enumerate(response_dict["data"]):
       b64_data = image_obj.get("b64_json")
       if not b64_data:
           print(f"Cảnh báo: Không tìm thấy 'b64_json' trong image object thứ {i+1}")
           continue


       image_data = base64.b64decode(b64_data)
      
       output_dir = "output/images"
       if not os.path.exists(output_dir):
           os.makedirs(output_dir)


       save_path = f"{output_dir}/generated_image_{int(time.time())}_{i+1}.png"
      
       with open(save_path, 'wb') as f:
           f.write(image_data)
       print(f"Image saved to {save_path}")
       saved_paths.append(save_path)
  
   if saved_paths:
       state["t2i_output_path"] = saved_paths[0] if len(saved_paths) == 1 else saved_paths
   else:
       state["t2i_output_path"] = "Failed to save any images."
  
   return state

