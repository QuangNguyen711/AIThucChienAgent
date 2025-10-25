# File: src/model/bot.py


import requests
import os
import time
import json
from typing import List, Dict, Any, Optional
import base64


class ThucChienAIBot:
   """
   Một lớp client để tương tác với các API của thucchien.ai.
   Lớp này cung cấp các phương thức để tạo chat, sinh ảnh, video,
   chuyển văn bản thành giọng nói và kiểm tra thông tin API key.
  
   Phiên bản này sử dụng các tham số rõ ràng thay vì **kwargs để tăng tính tường minh.
   """
   BASE_URL = "https://api.thucchien.ai"


   def __init__(self, api_key: str):
       """
       Khởi tạo Bot client.


       Args:
           api_key (str): API key của bạn từ thucchien.ai.
       """
       if not api_key:
           raise ValueError("API key không được để trống.")
       self.api_key = api_key
       self.session = requests.Session()


   def _make_request(
       self,
       method: str,
       endpoint: str,
       auth_type: str = 'bearer',
       data: Optional[Dict[str, Any]] = None,
       output_file: Optional[str] = None
   ) -> Optional[Dict[str, Any]]:
       """
       Một phương thức nội bộ để thực hiện các yêu cầu HTTP đến API.


       Args:
           method (str): Phương thức HTTP (ví dụ: 'GET', 'POST').
           endpoint (str): Endpoint của API (ví dụ: '/chat/completions').
           auth_type (str): Loại xác thực, 'bearer' hoặc 'google'.
           data (Optional[Dict]): Dữ liệu payload cho các request POST.
           output_file (Optional[str]): Đường dẫn để lưu file trả về (cho audio/video).


       Returns:
           Optional[Dict[str, Any]]: Dữ liệu JSON từ phản hồi của API hoặc thông tin file đã lưu.
       """
       url = f"{self.BASE_URL}{endpoint}"
       headers = {"Content-Type": "application/json"}


       if auth_type == 'bearer':
           headers["Authorization"] = f"Bearer {self.api_key}"
       elif auth_type == 'google':
           headers["x-goog-api-key"] = self.api_key
       else:
           raise ValueError("auth_type phải là 'bearer' hoặc 'google'.")


       try:
           response = self.session.request(
               method,
               url,
               json=data,
               headers=headers,
               stream=bool(output_file)
           )
           response.raise_for_status()


           if output_file:
               with open(output_file, 'wb') as f:
                   for chunk in response.iter_content(chunk_size=8192):
                       f.write(chunk)
               print(f"File đã được lưu thành công tại: {output_file}")
               return {"status": "success", "file_path": output_file}
          
           if response.status_code == 204 or not response.content:
               return None
              
           return response.json()


       except requests.exceptions.HTTPError as http_err:
           print(f"Lỗi HTTP: {http_err}")
           print(f"Chi tiết lỗi từ API: {http_err.response.text}")
       except requests.exceptions.RequestException as req_err:
           print(f"Lỗi Request: {req_err}")
       except json.JSONDecodeError:
           print(f"Không thể giải mã JSON từ phản hồi. Phản hồi thô: {response.text}")
      
       return None


   # --- HÀM HELPER MỚI ĐỂ MÃ HÓA ẢNH ---
   def _encode_image_to_base64(self, image_path: str) -> Dict[str, str]:
       """
       Đọc file ảnh, mã hóa sang base64 và xác định mime type.
       """
       # Xác định mime type dựa trên phần mở rộng file
       ext = os.path.splitext(image_path)[1].lower()
       mime_types = {
           '.jpg': 'image/jpeg',
           '.jpeg': 'image/jpeg',
           '.png': 'image/png',
           '.webp': 'image/webp'
       }
       mime_type = mime_types.get(ext)
       if not mime_type:
           raise ValueError(f"Định dạng file không được hỗ trợ: {ext}. Chỉ hỗ trợ JPG, PNG, WEBP.")


       # Đọc file và mã hóa
       try:
           with open(image_path, "rb") as image_file:
               encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
           return {"mime_type": mime_type, "data": encoded_string}
       except FileNotFoundError:
           raise FileNotFoundError(f"Không tìm thấy file ảnh tại đường dẫn: {image_path}")


   # --- Các hàm cho Chat & Image ---


   def create_chat_completion(
       self,
       model: str,
       messages: List[Dict[str, str]],
       temperature: Optional[float] = None,
       max_tokens: Optional[int] = None,
       modalities: Optional[List[str]] = None
   ) -> Optional[Dict[str, Any]]:
       """
       Tạo phản hồi trò chuyện (Chat Completions).


       Args:
           model (str): Tên model (ví dụ: 'gemini-2.5-flash').
           messages (List[Dict[str, str]]): Lịch sử cuộc trò chuyện.
           temperature (Optional[float]): Mức độ sáng tạo của phản hồi.
           max_tokens (Optional[int]): Số lượng token tối đa để tạo.
           modalities (Optional[List[str]]): Dùng để sinh ảnh, ví dụ: ["image"].


       Returns:
           Optional[Dict[str, Any]]: Phản hồi từ API.
       """
       payload = {"model": model, "messages": messages}
       if temperature is not None:
           payload["temperature"] = temperature
       if max_tokens is not None:
           payload["max_tokens"] = max_tokens
       if modalities is not None:
           payload["modalities"] = modalities
          
       return self._make_request("POST", "/chat/completions", data=payload, auth_type='bearer')


   def generate_image(
       self,
       model: str,
       prompt: str,
       n: Optional[int] = 1,
       aspect_ratio: Optional[str] = None,
       size: Optional[str] = None  # <-- THAM SỐ MỚI
   ) -> Optional[Dict[str, Any]]:
       """
       Sinh hình ảnh (Image Generation).


       Args:
           model (str): Tên model (ví dụ: 'imagen-4').
           prompt (str): Mô tả hình ảnh cần tạo.
           n (Optional[int]): Số lượng hình ảnh cần tạo.
           aspect_ratio (Optional[str]): Tỷ lệ khung hình, ví dụ: "1:1", "16:9".
           size (Optional[str]): Kích thước ảnh, ví dụ: "1024x1024".


       Returns:
           Optional[Dict[str, Any]]: Phản hồi từ API.
       """
       payload = {"model": model, "prompt": prompt}
       if n is not None:
           payload["n"] = n
       if aspect_ratio is not None:
           payload["aspect_ratio"] = aspect_ratio
       # --- THÊM LOGIC XỬ LÝ CHO SIZE ---
       if size is not None:
           payload["size"] = size
       # -----------------------------------
          
       return self._make_request("POST", "/images/generations", data=payload, auth_type='bearer')


   def generate_image_gemini(
       self,
       model: str,
       prompt: str,
       aspect_ratio: str = "1:1"
   ) -> Optional[Dict[str, Any]]:
       """
       Sinh/Sửa hình ảnh với Google Gemini.


       Args:
           model (str): Tên model (ví dụ: 'gemini-2.5-flash-image-preview').
           prompt (str): Mô tả hình ảnh cần tạo.
           aspect_ratio (str): Tỷ lệ khung hình (ví dụ: '1:1', '9:16').


       Returns:
           Optional[Dict[str, Any]]: Phản hồi từ API.
       """
       endpoint = f"/gemini/v1beta/models/{model}:generateContent"
       payload = {
           "contents": [{"parts": [{"text": prompt}]}],
           "generationConfig": {
               "imageConfig": {"aspectRatio": aspect_ratio}
           }
       }
       return self._make_request("POST", endpoint, data=payload, auth_type='google')


   # --- HÀM MỚI ĐỂ CHỈNH SỬA ẢNH ---
   def edit_image_gemini(
       self,
       model: str,
       prompt: str,
       image_paths: List[str], # <-- THAY ĐỔI: Từ str thành List[str]
       aspect_ratio: str = "1:1"
   ) -> Optional[Dict[str, Any]]:
       """
       Phân tích hoặc chỉnh sửa hình ảnh dựa trên prompt và một hoặc nhiều ảnh đầu vào.


       Args:
           model (str): Tên model đa phương thức.
           prompt (str): Yêu cầu hoặc câu hỏi về các hình ảnh.
           image_paths (List[str]): Danh sách các đường dẫn đến file ảnh cần xử lý.
           aspect_ratio (str): Tỷ lệ khung hình cho ảnh đầu ra (nếu có).


       Returns:
           Optional[Dict[str, Any]]: Phản hồi từ API.
       """
       # Bắt đầu payload với phần text
       parts = [{"text": prompt}]


       # Lặp qua từng đường dẫn ảnh, mã hóa và thêm vào danh sách parts
       try:
           for path in image_paths:
               print(f"Đang xử lý ảnh: {path}")
               image_data = self._encode_image_to_base64(path)
               parts.append({
                   "inlineData": {
                       "mimeType": image_data["mime_type"],
                       "data": image_data["data"]
                   }
               })
       except (ValueError, FileNotFoundError) as e:
           print(f"Lỗi xử lý ảnh: {e}")
           return None


       endpoint = f"/gemini/v1beta/models/{model}:generateContent"
      
       # Xây dựng payload cuối cùng với tất cả các part (text + images)
       payload = {
           "contents": [{"parts": parts}],
           "generationConfig": {
               "imageConfig": {"aspectRatio": aspect_ratio}
           }
       }
      
       return self._make_request("POST", endpoint, data=payload, auth_type='google')


   # --- Các hàm cho Video ---
  
   def generate_video(
       self,
       output_file: str,
       model: str,
       prompt: str,
       image_path: Optional[str] = None,
       negative_prompt: Optional[str] = None,
       aspect_ratio: Optional[str] = "16:9",
       resolution: Optional[str] = "720p",
       poll_interval: int = 15
   ) -> Optional[Dict[str, Any]]:
       """
       Sinh video từ prompt (và tùy chọn từ một ảnh) theo quy trình 3 bước.
       """
       print("Bước 1/3: Bắt đầu tác vụ sinh video...")
       start_endpoint = f"/gemini/v1beta/models/{model}:predictLongRunning"
      
       parameters = {}
       if negative_prompt is not None:
           parameters["negativePrompt"] = negative_prompt
       if aspect_ratio is not None:
           parameters["aspectRatio"] = aspect_ratio
       if resolution is not None:
           parameters["resolution"] = resolution
          
       instance = {"prompt": prompt}
       if image_path:
           print(f"Sử dụng ảnh đầu vào từ: {image_path}")
           try:
               image_data = self._encode_image_to_base64(image_path)
              
               # --- THAY ĐỔI QUAN TRỌNG ---
               # API yêu cầu cả bytesBase64Encoded và mimeType.
               instance["image"] = {
                   "bytesBase64Encoded": image_data["data"],
                   "mimeType": image_data["mime_type"] # <-- THÊM DÒNG NÀY
               }
               # ---------------------------


           except (ValueError, FileNotFoundError) as e:
               print(f"Lỗi xử lý ảnh đầu vào: {e}")
               return None
      
       payload = {"instances": [instance], "parameters": parameters}
      
       start_response = self._make_request("POST", start_endpoint, data=payload, auth_type='google')


       if not start_response or 'name' not in start_response:
           print("Không thể bắt đầu tác vụ sinh video.")
           return None
       operation_name = start_response['name']
       print(f"Tác vụ đã bắt đầu. Tên tác vụ: {operation_name}")


       # --- Phần Bước 2 và Bước 3 giữ nguyên ---
       print(f"Bước 2/3: Kiểm tra trạng thái mỗi {poll_interval} giây...")
       status_endpoint = f"/gemini/v1beta/{operation_name}"
       while True:
           status_response = self._make_request("GET", status_endpoint, auth_type='google')
           if not status_response:
               print("Không thể lấy trạng thái tác vụ.")
               return None
          
           if status_response.get('done'):
               print("Tác vụ đã hoàn thành.")
               break
          
           print("Tác vụ đang được xử lý, vui lòng chờ...")
           time.sleep(poll_interval)
      
       try:
           video_uri = status_response['response']['generateVideoResponse']['generatedSamples'][0]['video']['uri']
           video_id = video_uri.split('/')[-1].split(':')[0]
           print(f"Bước 3/3: Tải video với ID: {video_id}")
           download_endpoint = f"/gemini/download/v1beta/files/{video_id}:download?alt=media"
           return self._make_request("GET", download_endpoint, auth_type='google', output_file=output_file)
       except (KeyError, IndexError, TypeError):
           print("Không tìm thấy URI video trong phản hồi.")
           print(f"Phản hồi đầy đủ từ API: {status_response}")
           return None
          
   # --- Các hàm cho Audio & Key Info ---


   def generate_speech(
       self,
       output_file: str,
       model: str,
       input_text: str,
       voice: str
   ) -> Optional[Dict[str, Any]]:
       """
       Chuyển văn bản thành giọng nói (Text-to-Speech).


       Args:
           output_file (str): Đường dẫn để lưu file âm thanh (ví dụ: 'audio.mp3').
           model (str): Tên model (ví dụ: 'gemini-2.5-flash-preview-tts').
           input_text (str): Văn bản cần chuyển đổi.
           voice (str): Tên giọng đọc (ví dụ: 'Zephyr').


       Returns:
           Optional[Dict[str, Any]]: Thông tin file đã được lưu.
       """
       payload = {"model": model, "input": input_text, "voice": voice}
       return self._make_request("POST", "/audio/speech", data=payload, auth_type='bearer', output_file=output_file)


   def generate_speech_gemini(
       self,
       model: str,
       prompt: str,
       voice_name: str
   ) -> Optional[Dict[str, Any]]:
       """
       Chuyển văn bản thành giọng nói với Google Gemini.


       Args:
           model (str): Tên model (ví dụ: 'gemini-2.5-flash-preview-tts').
           prompt (str): Văn bản cần chuyển, có thể bao gồm hướng dẫn.
           voice_name (str): Tên giọng đọc (ví dụ: 'Kore').


       Returns:
           Optional[Dict[str, Any]]: Phản hồi từ API chứa dữ liệu audio base64.
       """
       endpoint = f"/gemini/v1beta/models/{model}:generateContent"
       payload = {
           "contents": [{"parts": [{"text": prompt}]}],
           "generationConfig": {
               "responseModalities": ["AUDIO"],
               "speechConfig": {
                   "voiceConfig": {
                       "prebuiltVoiceConfig": {"voiceName": voice_name}
                   }
               }
           }
       }
       return self._make_request("POST", endpoint, data=payload, auth_type='google')
      
   def get_key_info(self) -> Optional[Dict[str, Any]]:
       """
       Kiểm tra thông tin chi tiêu của API key.


       Returns:
           Optional[Dict[str, Any]]: Thông tin chi tiết của key.
       """
       return self._make_request("GET", "/key/info", auth_type='bearer')




if __name__ == '__main__':
   # --- HƯỚNG DẪN SỬ DỤNG ---
   # 1. Thay thế "YOUR_API_KEY" bằng API key của bạn.
   # 2. Bỏ comment (xóa dấu #) ở các phần bạn muốn chạy thử.


   API_KEY = os.environ.get("THUCCHIEN_API_KEY", "YOUR_API_KEY")


   if API_KEY == "YOUR_API_KEY":
       print("Vui lòng thay thế 'YOUR_API_KEY' bằng API key thật của bạn hoặc đặt biến môi trường THUCCHIEN_API_KEY.")
   else:
       bot = ThucChienAIBot(api_key=API_KEY)


       # === 1. Tạo phản hồi trò chuyện (Chat Completions) ===
       # print("\n--- 1. Thử nghiệm Chat Completions ---")
       # messages = [
       #     {"role": "system", "content": "Bạn là một trợ lý ảo hữu ích."},
       #     {"role": "user", "content": "Việt Nam có những di sản thế giới nào được UNESCO công nhận?"}
       # ]
       # chat_response = bot.create_chat_completion(
       #     model="gemini-2.5-flash",
       #     messages=messages,
       #     temperature=0.7
       # )
       # if chat_response:
       #     print(json.dumps(chat_response, indent=2, ensure_ascii=False))


       # === 2. Sinh hình ảnh (Image Generation) ===
       # print("\n--- 2. Thử nghiệm Image Generation ---")
       # image_response = bot.generate_image(
       #     model="imagen-4",
       #     prompt="Một con mèo đang lập trình trên máy tính, phong cách tranh sơn dầu",
       #     n=1,
       #     aspect_ratio="16:9"
       # )
       # if image_response:
       #     print(json.dumps(image_response, indent=2))


       # === 3. Sinh hình ảnh (với Chat Completions) ===
       # print("\n--- 3. Thử nghiệm Image Generation qua Chat ---")
       # image_chat_response = bot.create_chat_completion(
       #     model="gemini-2.5-flash-image-preview",
       #     messages=[{"role": "user", "content": "Hà Nội về đêm, hồ gươm lung linh ánh đèn, phong cách anime."}],
       #     modalities=["image"]
       # )
       # if image_chat_response:
       #      print(json.dumps(image_chat_response, indent=2))


       # === 4. Sinh hình ảnh (với Google Gemini) ===
       # print("\n--- 4. Thử nghiệm Image Generation với Gemini ---")
       # gemini_image_response = bot.generate_image_gemini(
       #     model="gemini-2.5-flash-image-preview",
       #     prompt="Một phi hành gia cưỡi ngựa trên sao Hỏa, ảnh siêu thực",
       #     aspect_ratio="16:9"
       # )
       # if gemini_image_response:
       #     print(json.dumps(gemini_image_response, indent=2))
      
       # === 5. Sinh video (Lưu ý: Quá trình này có thể mất vài phút) ===
       # print("\n--- 5. Thử nghiệm Video Generation ---")
       # video_result = bot.generate_video(
       #     output_file="generated_video.mp4",
       #     model="veo-3.0-generate-001",
       #     prompt="A cinematic shot of a goldfish swimming in a teacup on a rainy day",
       #     aspect_ratio="16:9",
       #     resolution="720p",
       #     negative_prompt="blurry, low quality"
       # )
       # if video_result:
       #     print(f"Kết quả sinh video: {video_result}")


       # === 6. Chuyển văn bản thành giọng nói (Text-to-Speech) ===
       # print("\n--- 6. Thử nghiệm Text-to-Speech ---")
       # speech_result = bot.generate_speech(
       #     output_file="generated_audio.mp3",
       #     model="gemini-2.5-flash-preview-tts",
       #     input_text="Xin chào, đây là một thử nghiệm chuyển văn bản thành giọng nói qua AI Thực Chiến gateway.",
       #     voice="Zephyr"
       # )
       # if speech_result:
       #     print(f"Kết quả tạo giọng nói: {speech_result}")
          
       # === 7. Chuyển văn bản thành giọng nói (với Google Gemini) ===
       # print("\n--- 7. Thử nghiệm Text-to-Speech với Gemini ---")
       # gemini_speech_response = bot.generate_speech_gemini(
       #     model="gemini-2.5-flash-preview-tts",
       #     prompt="Say cheerfully: Have a wonderful day!",
       #     voice_name="Kore"
       # )
       # if gemini_speech_response:
       #      print("Phản hồi thành công, chứa dữ liệu audio base64.")
       #      # print(json.dumps(gemini_speech_response, indent=2))


       # === 8. Kiểm tra chi tiêu (Spend Checking) ===
       print("\n--- 8. Thử nghiệm Kiểm tra thông tin Key ---")
       key_info = bot.get_key_info()
       if key_info:
           print("Thông tin Key:")
           print(f"  - Tên Key: {key_info['info']['key_name']}")
           print(f"  - Chi tiêu: ${key_info['info']['spend']:.6f}")
           print(f"  - Models được phép: {key_info['info']['models']}")
           # print(json.dumps(key_info, indent=2, ensure_ascii=False))


from src.graph.state import State
from src.nodes.text2text import text2text
from src.nodes.text2img import text2img
from src.nodes.textimg2text import textimg2text
from src.nodes.textimg2img import text_img2img


"""
Build a LangGraph-style flow that creates a story about Conan following the specified steps:


1. Generate detailed character description (text -> text)
2. Generate small story with character and story fields
3. Generate character image from description (text -> image) saved to output/artifact
4. Create 3-step plan from story (bối cảnh + kịch bản for each step)
5. Generate images for each step/scene
6. Create final outputs saved to output/story


Flow:
- Character description: text2text
- Small story creation: text2text 
- Character image: text2img (saved to output/artifact)
- Plan generation: text2text (3 steps with bối cảnh + kịch bản)
- Scene images: text2img for each step
- Final outputs: step + character + images saved to output/story
"""


from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import os
import json
import uuid




# Node functions for the story flow
def generate_character_description(state: State, config: RunnableConfig) -> State:
   """Step 1: Generate detailed character description for Conan"""
   print("--- Step 1: Generating character description ---")
  
   state["t2t_question"] = (
       "Create a detailed description of Conan character 12 years old in anime japan. Include appearance, age, clothing. Write in 3 sentences in English."
   )
  
   # Call text2text node
   state = text2text(state, config)
  
   # Store character description for later use
   state["character_description"] = state["t2t_answer"]['choices'][0]["message"]["content"]
   return state




def generate_small_story(state: State, config: RunnableConfig) -> State:
   """Step 2: Generate small story with character and story fields"""
   print("--- Step 2: Generating small story ---")
  
   character_desc = state.get("character_description", "Conan")
  
   state["t2t_question"] = (
       f"Based on the character description: {character_desc}\n\n"
       "Create a short story featuring this character. Write in 10 sentences in English."
   )
   # Call text2text node 
   state = text2text(state, config)
  
   # Store small story
   state["small_story"] = state["t2t_answer"]['choices'][0]["message"]["content"]
   return state




def generate_character_image(state: State, config: RunnableConfig) -> State:
   """Step 3: Generate character image from description"""
   print("--- Step 3: Generating character image ---")
  
   character_desc = state.get("character_description", "A detective character")
  
   # Ensure output directory exists
   os.makedirs("output/artifact", exist_ok=True)
  
   state["t2i_question"] = f"Create image a detailed illustration of: {character_desc}"
   state["t2i_num_images"] = 1
   state["t2i_output_path"] = "output/artifact/character_image.png"
  
   # Call text2img node
   # state = text2img(state, config)
  
   # Store image object
   # state["character_image"] = state["t2i_output_path"]
   state["character_image"] = "output/images/generated_image_1761237726_1.png"
  
   return state


def parse_json_safe(json_string: str) -> Dict[str, Any]:
   """Helper function to safely parse JSON strings."""
   try:
       import re
       pattern = r'```json(.*?)```'
       match = re.search(pattern, json_string, re.DOTALL)
       if match:
           return json.loads(match.group(1))
   except json.JSONDecodeError:
       raise ValueError("Invalid JSON format")


def generate_story_plan(state: State, config: RunnableConfig) -> State:
   """Step 4: Create 3-step plan from story"""
   print("--- Step 4: Generating story plan ---")
  
   small_story = state.get("small_story", "")
   if small_story == "":
       raise ValueError("Small story is empty, cannot generate story plan.")
  
   state["t2t_question"] = (
       f"Based on the following story: {small_story}\n\n"
       "Create a 3-step plan in JSON format:\n"
       "Write in English."
       "{\n"
       '  "step1": {"context": "Scene description", "scenario": "Action description"},\n'
       '  "step2": {"context": "Scene description", "scenario": "Action description"},\n'
       '  "step3": {"context": "Scene description", "scenario": "Action description"}\n'
       "}\n"
   )
  
   # Call text2text node
   state = text2text(state, config)
  
   # Store plan
   state["story_plan"] = state["t2t_answer"]['choices'][0]["message"]["content"]
  
   return state




def generate_scene_images(state: State, config: RunnableConfig) -> State:
   """Step 5: Generate images for each step/scene using textimg2img"""
   print("--- Step 5: Generating scene images using character image ---")
  
   story_plan = state.get("story_plan", "```json{}```")
   character_image = state.get("character_image", "")
  
   # Ensure output directory exists
   os.makedirs("output/artifact", exist_ok=True)
  
   try:
       plan_data = parse_json_safe(story_plan) if isinstance(story_plan, str) else story_plan
   except:
       raise ValueError("Invalid story plan JSON format")
  
   scene_images = {}
  
   for step_key in ["step1", "step2", "step3"]:
       if step_key in plan_data:
           step_data = plan_data[step_key]
           context = step_data.get("context", f"Scene for {step_key}")
           scenario = step_data.get("scenario", f"Action for {step_key}")
          
           # Create text prompt for the scene
           text_prompt = (
               f"Context: {context}. "
               f"Scenario: {scenario}. "
               f"Create image with character in this scene."
           )
          
           # Use textimg2img with character image + text prompt
           state["ti2i_question"] = text_prompt
           state["ti2i_image_paths"] = [character_image]
           state["ti2i_num_images"] = 1
          
           # Call textimg2img node
           state = text_img2img(state, config)


           scene_images[step_key] = state["ti2i_output_path"]
  
   state["scene_images"] = scene_images
  
   return state




def create_final_outputs(state: State, config: RunnableConfig) -> State:
   """Step 6: Create final outputs and save to output/story"""
   print("--- Step 6: Creating final outputs ---")
  
   # Ensure output directory exists
   os.makedirs("output/story", exist_ok=True)
  
   character_desc = state.get("character_description", "")
   character_image = state.get("character_image", "")
   story_plan = state.get("story_plan", "{}")
   scene_images = state.get("scene_images", {})
   small_story = state.get("small_story", "{}")
  
   try:
       plan_data = json.loads(story_plan) if isinstance(story_plan, str) else story_plan
   except:
       plan_data = {
           "step1": {"context": "Scene 1", "scenario": "Action 1"},
           "step2": {"context": "Scene 2", "scenario": "Action 2"},
           "step3": {"context": "Scene 3", "scenario": "Action 3"}
       }
  
   # Create outputs for each step
   outputs = {}
  
   for i, step_key in enumerate(["step1", "step2", "step3"], 1):
       step_data = plan_data.get(step_key, {})
       step_image = scene_images.get(step_key, "")
      
       output = {
           "step": step_key,
           "context": step_data.get("context", f"Scene {i}"),
           "scenario": step_data.get("scenario", f"Action {i}"),
           "character": character_desc,
           "character_image": character_image,
           "scene_image": step_image
       }
      
       outputs[f"output{i}"] = output
      
       # Save individual output file
       with open(f"output/story/output{i}.json", "w", encoding="utf-8") as f:
           json.dump(output, f, ensure_ascii=False, indent=2)
  
   # Save small story
   with open("output/story/small_story.json", "w", encoding="utf-8") as f:
       json.dump({"small_story": small_story}, f, ensure_ascii=False, indent=2)
  
   # Save complete story
   complete_story = {
       "character_description": character_desc,
       "character_image": character_image,
       "small_story": small_story,
       "story_plan": story_plan,
       "outputs": outputs
   }
  
   with open("output/story/complete_story.json", "w", encoding="utf-8") as f:
       json.dump(complete_story, f, ensure_ascii=False, indent=2)
  
   state["final_outputs"] = outputs
   state["complete_story"] = complete_story
  
   print("Final outputs saved to output/story/")
  
   return state




def build_story_flow() -> StateGraph:
   """Build the complete story flow using LangGraph"""
  
   # Create StateGraph
   workflow = StateGraph(State)
  
   # Add all nodes
   workflow.add_node("generate_character_description", generate_character_description)
   workflow.add_node("generate_small_story", generate_small_story)
   workflow.add_node("generate_character_image", generate_character_image)
   workflow.add_node("generate_story_plan", generate_story_plan)
   workflow.add_node("generate_scene_images", generate_scene_images)
   workflow.add_node("create_final_outputs", create_final_outputs)
  
   # Define the flow
   workflow.add_edge(START, "generate_character_description")
   workflow.add_edge("generate_character_description", "generate_small_story")
   workflow.add_edge("generate_small_story", "generate_character_image")
   workflow.add_edge("generate_character_image", "generate_story_plan")
   workflow.add_edge("generate_story_plan", "generate_scene_images")
   workflow.add_edge("generate_scene_images", "create_final_outputs")
   workflow.add_edge("create_final_outputs", END)
  
   return workflow.compile()




def run_story_flow():
   """Run the complete story flow"""
   from src.model.bot import ThucChienAIBot
   from dotenv import load_dotenv
   load_dotenv()
   api_key = os.getenv("THUC_CHIEN_API_KEY")


   # Initialize bot
   bot = ThucChienAIBot(api_key)


   # Build the flow
   workflow = build_story_flow()
  
   # Initial state
   initial_state = State()
  
   # Configuration with bot
   config = RunnableConfig(
       configurable={"bot": bot}
   )
  
   print("Starting Conan story generation flow...")
  
   # Execute the workflow
   final_state = workflow.invoke(initial_state, config)
  
   print("\n=== Story Flow Completed ===")
   print(f"Character description: {final_state.get('character_description', 'N/A')[:100]}...")
   print(f"Character image: {final_state.get('character_image', 'N/A')}")
   print(f"Final outputs saved: {len(final_state.get('final_outputs', {}))}")
  
   return final_state




if __name__ == "__main__":
   run_story_flow()



