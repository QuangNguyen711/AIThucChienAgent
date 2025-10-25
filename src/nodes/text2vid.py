# File: src/nodes/text2vid.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import os
import time


SYSTEM_PROMPT_VI = """
Bạn là một trợ lý hữu ích, hãy tạo video theo yêu cầu của người dùng.
"""


SYSTEM_PROMPT_EN = """
You are a helpful assistant, help generate a video according to user demand.
"""


def text2vid(state: State, config: RunnableConfig) -> State:
   """NODE: Tạo video dựa trên yêu cầu (prompt)."""
   print("--- Thực hiện Node: text2vid ---")


   # Lấy thông tin cần thiết từ state
   question = state.get("t2v_question")
   negative_question = state.get("t2v_negative_question", None) # Tùy chọn
   aspect_ratio = state.get("t2v_aspect_ratio", "16:9")     # Tùy chọn, mặc định 16:9
   resolution = state.get("t2v_resolution", "720p")        # Tùy chọn, mặc định 720p


   # Kiểm tra đầu vào bắt buộc
   if not question:
       print("Lỗi: Cần cung cấp 't2v_question' trong state.")
       state["t2v_output_path"] = "Error: Missing question for video generation."
       return state


   bot = config["configurable"]["bot"]


   # --- Chuẩn bị đường dẫn để lưu video ---
   # Hàm bot.generate_video yêu cầu một đường dẫn file đầu ra
   if not os.path.exists("output"):
       os.makedirs("output")
  
   save_path = state.get("t2v_output_path", f"output/videos/generated_video_{int(time.time())}.mp4")
  
   # --- Gọi API để tạo video ---
   # Hàm này sẽ tự xử lý quy trình 3 bước và in ra tiến độ
   video_result = bot.generate_video(
       output_file=save_path,
       model=os.getenv("VIDEO_MODEL_NAME"), # Ví dụ: veo-3.0-generate-001
       prompt=question,
       negative_prompt=negative_question,
       aspect_ratio=aspect_ratio,
       resolution=resolution
   )


   # --- Xử lý kết quả ---
   # bot.generate_video trả về một dict có 'status' và 'file_path' nếu thành công, hoặc None nếu thất bại.
   if video_result and video_result.get("status") == "success":
       output_path = video_result.get("file_path")
       print(f"Quá trình tạo video hoàn tất. File được lưu tại: {output_path}")
       # Cập nhật state với đường dẫn file đã lưu
       state["t2v_output_path"] = output_path
   else:
       print("Lỗi: Quá trình tạo video thất bại.")
       # Cập nhật state với thông báo lỗi
       state["t2v_output_path"] = "API call failed or was interrupted. No video generated."
  
   return state

