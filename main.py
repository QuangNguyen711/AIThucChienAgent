import dotenv
import os
from src.model.bot import ThucChienAIBot
from langchain_core.runnables import RunnableConfig
from src.graph.state import State
from src.graph.builder import build_graph
import json

def main():
    dotenv.load_dotenv()

    bot = ThucChienAIBot(api_key=os.getenv("THUC_CHIEN_API_KEY"))

    config = RunnableConfig(
        configurable={"bot": bot}
    )

    decision = "textimg2img"
    image_input = ""
#     content = """
# Yêu cầu: Từ các thông tin dưới đây sinh cho tôi ảnh của nhân vật này với phong cách truyện tranh phong cách việt nam, cổ điển
# Ảnh: ảnh đen trắng
# Quan trọng: Không được thêm chữ vào ảnh và chỉ sinh một nhân vật duy nhất, không có background
# Mô tả nhân vật:
# Tên: Minh
# Tuổi: 14 tuổi
# Lớp: 8 (học sinh THCS)
# Ngoại hình: Minh có vóc dáng vừa phải, nhanh nhẹn và luôn mang một vẻ ngoài năng động. Cậu thường diện áo phông đơn giản, quần jeans và giày thể thao, đúng phong cách của một cậu học sinh đang tuổi lớn
# """
#     plan = """
# Tiêu đề: "Mật Mã Của Bố"
# Tóm tắt cốt truyện: Minh, một học sinh THCS, nhận được tin nhắn từ một tài khoản Facebook giả mạo bố mình, yêu cầu khẩn cấp chuyển tiền vào một tài khoản lạ để giải quyết "sự cố gấp" và dặn dò không được nói với mẹ. Minh lo lắng và định làm theo, nhưng may mắn nhớ lại lời dặn của thầy cô về việc xác minh thông tin. Minh đã gọi điện trực tiếp cho bố để kiểm tra và phát hiện ra đây là chiêu trò lừa đảo giả danh người thân.
# Thông điệp chính: Luôn xác minh thông tin trực tiếp với người thân qua kênh liên lạc chính thức (gọi điện, gặp mặt) khi nhận được yêu cầu chuyển tiền hoặc thông tin quan trọng.
# Điểm nhấn cho học sinh: Tình huống giả danh người thân, lợi dụng sự tin tưởng và lo lắng của con cái.

# Trả ra cho tôi 10 kịch bản cho câu chuyện trên ở ngôi kể chuyện thứ 3 mỗi scenario là 1 bối cảnh và kịch bản có sự liên kết đến nhau như một câu chuyện dưới dạng file json với cấu trúc như sau:
# {
#     "scenarios": [
#         {
#             "scene": "Scene 1",
#             "scenario": "Detail of Scenario 1"
#         },
#         {
#             "scene": "Scene 2",
#             "scenario": "Detail of Scenario 2"
#         },
#         {
#             "scene": "Scene 3",
#             "scenario": "Detail of Scenario 3"
#         },
#         {
#             "scene": "Scene 4",
#             "scenario": "Detail of Scenario 4"
#         },
#         {
#             "scene": "Scene 5",
#             "scenario": "Detail of Scenario 5"
#         },
#         {
#             "scene": "Scene 6",
#             "scenario": "Detail of Scenario 6"
#         },
#         {
#             "scene": "Scene 7",
#             "scenario": "Detail of Scenario 7"
#         },
#         {
#             "scene": "Scene 8",
#             "scenario": "Detail of Scenario 8"
#         },
#         {
#             "scene": "Scene 9",
#             "scenario": "Detail of Scenario 9"
#         },
#         {
#             "scene": "Scene 10",
#             "scenario": "Detail of Scenario 10"
#         }
#     ]
# }
# """

#     with open("scenarios.json", "r", encoding="utf-8") as f:
#         scenarios = json.load(f)
#         scenarios = scenarios["scenarios"]

#     convert_to_image_prompt = """With the provided scenarios, create a scenarios of prompt for image generation with the following structure:
# {
#     "scenarios": [
#         {
#             "scene": "Scene 1",
#             "scenario": "Prompt for image generation for the scene 1"
#         },
#         {
#             "scene": "Scene 2",
#             "scenario": "Prompt for image generation for the scene 2"
#         },
#         {
#             "scene": "Scene 3",
#             "scenario": "Prompt for image generation for the scene 3"
#         }
#     ]
# }
# """
#     convert_to_image_prompt += f"Here is the scenarios:\n{scenarios}"

    app = build_graph(decision)

#     state = State(
#         t2t_question=convert_to_image_prompt
#     )
#     result = app.invoke(state, config)
#     print(result)

    with open("image_scenario.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)


    for i, scenario in enumerate(scenarios["scenarios"]):
        scene = scenario["scene"]
        scenario = scenario["scenario"]

        prompt = f"With the provided character image, CREATE black and white image for the scene {scene} with the following scenario:\n{scenario}"

        state = State(
            ti2i_image_paths=["output/images/generated_image_1761387977_1.png"],
            ti2i_question=prompt,
            ti2i_aspect_ratio="3:4",
            ti2i_output_path=f"output/image/scene_{i}.png"
        )
        result = app.invoke(state, config)
        print(result)

    # if decision == "text2text":
    #     state = State(
    #         t2t_question=plan
    #     )
    # elif decision == "text2img":
    #     state = State(
    #         t2i_question=content,
    #         t2i_aspect_ratio="3:4",
    #         t2i_size="2480x3508",
    #         t2i_output_path="output/image.jpg"
    #     )
    # elif decision == "textimg2img":
    #     state = State(
    #         ti2i_image_paths=["output/images/generated_image_1761386492_1.png"],
    #         ti2i_question="The cat jump out of the chair",
    #         ti2i_aspect_ratio="3:4",
    #         ti2i_output_path="path/to/image1.jpg"
    #     )
    # elif decision == "textimg2text":
    #     state = State(
    #         ti2t_question="Describe the image",
    #         ti2t_image_path="path/to/image.jpg"
    #         )
    # else:
    #     raise ValueError(f"Invalid decision: {decision}")

    
    result = app.invoke(state, config)
    print(result)




if __name__ == "__main__":
    main()
