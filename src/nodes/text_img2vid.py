# File: nodes/text_img2vid.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import os
import time


SYSTEM_PROMPT_VI = """
Bạn là một trợ lý hữu ích, hãy tạo video từ hình ảnh và yêu cầu của người dùng.
"""


SYSTEM_PROMPT_EN = """
You are a helpful assistant, help generate a video from an image and a user's demand.
"""


def text_img2vid(state: State, config: RunnableConfig) -> State:
   """NODE: Tạo video dựa trên ảnh đầu vào và yêu cầu (prompt)."""
   print("--- Thực hiện Node: textimg2vid ---")


   # Lấy thông tin cần thiết từ state
   question = state.get("ti2v_question")
   input_path = state.get("ti2v_image_path")
   negative_question = state.get("ti2v_negative_question", None)
   aspect_ratio = state.get("ti2v_aspect_ratio", "16:9")
   resolution = state.get("ti2v_resolution", "720p")


   # Kiểm tra các đầu vào bắt buộc
   if not question or not input_path:
       print("Lỗi: Cần cung cấp 'ti2v_question' và 'ti2v_image_path' trong state.")
       state["ti2v_output_path"] = "Error: Missing question or input image path."
       return state


   if not os.path.exists(input_path):
       print(f"Lỗi: Không tìm thấy file ảnh đầu vào tại '{input_path}'.")
       state["ti2v_output_path"] = f"Error: Input file not found at {input_path}."
       return state


   bot = config["configurable"]["bot"]


   # --- Chuẩn bị đường dẫn để lưu video ---
   if not os.path.exists("output"):
       os.makedirs("output")
  
   save_path = state.get("ti2v_output_path", f"output/videos/generated_video_from_image_{int(time.time())}.mp4")
  
   # --- Gọi API để tạo video từ ảnh và question ---
   video_result = bot.generate_video(
       output_file=save_path,
       model=os.getenv("VIDEO_MODEL_NAME"), # Ví dụ: veo-3.0-generate-001
       prompt=question,
       image_path=input_path, # <-- Truyền đường dẫn ảnh vào đây
       negative_prompt=negative_question,
       aspect_ratio=aspect_ratio,
       resolution=resolution
   )


   # --- Xử lý kết quả ---
   if video_result and video_result.get("status") == "success":
       output_path = video_result.get("file_path")
       print(f"Quá trình tạo video hoàn tất. File được lưu tại: {output_path}")
       state["ti2v_output_path"] = output_path
   else:
       print("Lỗi: Quá trình tạo video từ ảnh thất bại.")
       state["ti2v_output_path"] = "API call failed. No video generated from the image."
  
   return state