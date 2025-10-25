# File: src/nodes/textimg2text.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import os


def textimg2text(state: State, config: RunnableConfig) -> State:
   """NODE: Trả lời câu hỏi hoặc mô tả ảnh dựa trên ảnh và prompt đầu vào."""
   print("--- Thực hiện Node: textimg2text ---")


   # Lấy thông tin cần thiết từ state (ti2t = text-image-to-text)
   prompt = state.get("ti2t_question")
   input_path = state.get("ti2t_image_path")


   # Kiểm tra các đầu vào bắt buộc
   if not prompt or not input_path:
       print("Lỗi: Cần cung cấp 'ti2t_question' và 'ti2t_image_path' trong state.")
       state["ti2t_answer"] = "Error: Missing prompt or input image path."
       return state
      
   if not os.path.exists(input_path):
       print(f"Lỗi: Không tìm thấy file ảnh đầu vào tại '{input_path}'.")
       state["ti2t_answer"] = f"Error: Input file not found at {input_path}."
       return state


   bot = config["configurable"]["bot"]
  
   # Chúng ta có thể tái sử dụng hàm edit_image_gemini vì nó gọi đến endpoint đa năng.
   # Endpoint này sẽ trả về text nếu prompt mang tính câu hỏi/mô tả.
   response_dict = bot.edit_image_gemini(
       model=os.getenv("MULTIMODAL_MODEL_NAME"), # Ví dụ: gemini-2.5-flash-image-preview
       prompt=prompt,
       image_path=input_path
   )


   if not response_dict or "candidates" not in response_dict:
       print("Lỗi: Không nhận được dữ liệu hợp lệ từ API.")
       state["ti2t_answer"] = "API call failed. No valid response received."
       return state


   text_responses = []


   for i, candidate in enumerate(response_dict.get("candidates", [])):
       try:
           part = candidate["content"]["parts"][0]
       except (KeyError, IndexError, TypeError):
           print(f"Cảnh báo: Cấu trúc không hợp lệ trong candidate thứ {i+1}")
           continue


       # Chúng ta chỉ quan tâm đến phần 'text' trong phản hồi
       if "text" in part:
           text_response = part["text"]
           print(f"API đã trả về văn bản: '{text_response}'")
           text_responses.append(text_response)
       else:
           # Ghi lại nếu API trả về một hình ảnh thay vì văn bản
           print(f"Cảnh báo: API đã trả về một hình ảnh thay vì văn bản cho candidate {i+1}")


   # Gộp tất cả các phản hồi văn bản thành một chuỗi duy nhất
   if text_responses:
       state["ti2t_answer"] = "\n".join(text_responses)
   else:
       state["ti2t_answer"] = "No text content could be extracted from the API response."
  
   return state

