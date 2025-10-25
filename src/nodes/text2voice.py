# File: src/nodes/text2voice.py


from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import os
import time


SYSTEM_PROMPT_VI = """
Bạn là một trợ lý hữu ích, hãy chuyển đổi văn bản thành giọng nói theo yêu cầu của người dùng.
"""


SYSTEM_PROMPT_EN = """
You are a helpful assistant, help convert text to speech according to user demand.
"""


def text2voice(state: State, config: RunnableConfig) -> State:
   """NODE: Chuyển đổi văn bản thành giọng nói (Text-to-Speech)."""
   print("--- Thực hiện Node: text2voice ---")


   # Lấy thông tin cần thiết từ state
   question = state.get("t2s_question") # t2s = text-to-speech
   voice_name = state.get("t2s_voice", "Zephyr") # Tùy chọn, mặc định là giọng 'Zephyr'


   # Kiểm tra đầu vào bắt buộc
   if not question:
       print("Lỗi: Cần cung cấp 't2s_question' trong state.")
       state["t2s_output_path"] = "Error: Missing text input for speech generation."
       return state


   bot = config["configurable"]["bot"]


   # --- Chuẩn bị đường dẫn để lưu file âm thanh ---
   if not os.path.exists("output"):
       os.makedirs("output")
  
   save_path = f"output/generated_audio_{int(time.time())}.mp3"
  
   # --- Gọi API để tạo file âm thanh ---
   # Hàm này sẽ gọi API và lưu file trực tiếp vào save_path
   audio_result = bot.generate_speech(
       output_file=save_path,
       model=os.getenv("TTS_MODEL_NAME"), # Ví dụ: gemini-2.5-flash-preview-tts
       input_text=question,
       voice=voice_name
   )


   # --- Xử lý kết quả ---
   # bot.generate_speech trả về một dict có 'status' và 'file_path' nếu thành công, hoặc None nếu thất bại.
   if audio_result and audio_result.get("status") == "success":
       output_path = audio_result.get("file_path")
       print(f"Quá trình tạo âm thanh hoàn tất. File được lưu tại: {output_path}")
       # Cập nhật state với đường dẫn file đã lưu
       state["t2s_output_path"] = output_path
   else:
       print("Lỗi: Quá trình tạo âm thanh thất bại.")
       # Cập nhật state với thông báo lỗi
       state["t2s_output_path"] = "API call failed. No audio generated."
  
   return state

