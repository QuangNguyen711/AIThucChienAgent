# File: src/nodes/text_img2img.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import base64
import os
import time


def text_img2img(state: State, config: RunnableConfig) -> State:
   """NODE: Chỉnh sửa hoặc phân tích dựa trên prompt và một hoặc nhiều ảnh đầu vào."""
   print("--- Thực hiện Node: text_img2img ---")


   prompt = state.get("ti2i_question")
   # <-- THAY ĐỔI: Lấy danh sách đường dẫn thay vì một đường dẫn
   input_paths = state.get("ti2i_image_paths")
   aspect_ratio = state.get("ti2i_aspect_ratio", "1:1")


   # Kiểm tra đầu vào
   if not prompt or not input_paths:
       print("Lỗi: Cần cung cấp 'ti2i_question' và 'ti2i_image_paths' (danh sách) trong state.")
       state["ti2i_output_path"] = "Error: Missing prompt or image paths list."
       return state
  
   # <-- THAY ĐỔI: Kiểm tra sự tồn tại của từng file trong danh sách
   for path in input_paths:
       if not os.path.exists(path):
           print(f"Lỗi: Không tìm thấy file ảnh đầu vào tại '{path}'.")
           state["ti2i_output_path"] = f"Error: Input file not found at {path}."
           return state


   bot = config["configurable"]["bot"]
  
   # <-- THAY ĐỔI: Truyền danh sách `input_paths` vào hàm
   response_dict = bot.edit_image_gemini(
       model=os.getenv("MULTIMODAL_MODEL_NAME"),
       prompt=prompt,
       image_paths=input_paths,
       aspect_ratio=aspect_ratio
   )


   if not response_dict or "candidates" not in response_dict:
       print("Lỗi: Không nhận được dữ liệu hợp lệ từ API.")
       state["ti2i_output_path"] = "API call failed. No valid response received."
       return state


   saved_paths = []
   text_responses = []


   for i, candidate in enumerate(response_dict.get("candidates", [])):
       try:
           part = candidate["content"]["parts"][0]
       except (KeyError, IndexError, TypeError):
           print(f"Cảnh báo: Cấu trúc không hợp lệ trong candidate thứ {i+1}")
           continue


       if "inlineData" in part:
           b64_data = part["inlineData"].get("data")
           if not b64_data:
               continue
          
           image_data = base64.b64decode(b64_data)
          
           save_path = state.get("ti2i_output_path")
           with open(save_path, 'wb') as f:
               f.write(image_data)
           print(f"Ảnh kết quả được lưu tại: {save_path}")
           saved_paths.append(save_path)
      
       elif "text" in part:
           text_response = part["text"]
           print(f"API đã trả về văn bản: '{text_response}'")
           text_responses.append(text_response)


   # Thay đổi key đầu ra để phản ánh đúng hơn (có thể là text hoặc paths)
   if saved_paths:
       state["ti2i_output_path"] = saved_paths
   elif text_responses:
       state["ti2i_output_path"] = "\n".join(text_responses)
   else:
       state["ti2i_output_path"] = "No image or text content could be extracted."
  
   return state

