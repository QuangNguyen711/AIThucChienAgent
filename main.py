import dotenv
import os
from src.model.bot import ThucChienAIBot
from langchain_core.runnables import RunnableConfig
from src.graph.state import State
from src.graph.builder import build_graph

def main():
    dotenv.load_dotenv()

    bot = ThucChienAIBot(api_key=os.getenv("THUC_CHIEN_API_KEY"))

    config = RunnableConfig(
        configurable={"bot": bot}
    )

    decision = "textimg2img"

    app = build_graph(decision)

    if decision == "text2text":
        state = State(
            t2t_question="What is the capital of France?"
        )
    elif decision == "text2img":
        state = State(
            t2i_question="A cat in a hat",
            t2i_aspect_ratio=None,
            t2i_size="2480x3508",
            t2i_output_path="output/image.jpg"
        )
    elif decision == "textimg2img":
        state = State(
            ti2i_image_paths=["output/images/generated_image_1761386492_1.png"],
            ti2i_question="The cat jump out of the chair",
            ti2i_aspect_ratio="3:4",
            ti2i_output_path="path/to/image1.jpg"
        )
    elif decision == "textimg2text":
        state = State(
            ti2t_question="Describe the image",
            ti2t_image_path="path/to/image.jpg"
            )
    else:
        raise ValueError(f"Invalid decision: {decision}")
    
    result = app.invoke(state, config)
    print(result)




if __name__ == "__main__":
    main()
