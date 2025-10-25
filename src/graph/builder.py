from src.nodes.text2text import text2text
from src.nodes.text2img import text2img
from src.nodes.text2vid import text2vid
from src.nodes.text2voice import text2voice
from src.nodes.text_img2vid import text_img2vid
from src.nodes.textimg2img import text_img2img
from src.nodes.textimg2text import textimg2text
from src.graph.state import State
from langgraph.graph import StateGraph, END


def build_graph(decision: str):
    workflow = StateGraph(State)

    workflow.add_node("text2text", text2text)
    workflow.add_node("text2img", text2img)
    workflow.add_node("text2vid", text2vid)
    workflow.add_node("text2voice", text2voice)
    workflow.add_node("text_img2vid", text_img2vid)
    workflow.add_node("textimg2img", text_img2img)
    workflow.add_node("textimg2text", textimg2text)
  
    if decision == "text2text":
        workflow.set_entry_point("text2text")
        workflow.add_edge("text2text", END)
    elif decision == "text2img":
        workflow.set_entry_point("text2img")
        workflow.add_edge("text2img", END)
    elif decision == "text2vid":
        workflow.set_entry_point("text2vid")
        workflow.add_edge("text2vid", END)
    elif decision == "text2voice":
        workflow.set_entry_point("text2voice")
        workflow.add_edge("text2voice", END)
    elif decision == "text_img2vid":
        workflow.set_entry_point("text_img2vid")
        workflow.add_edge("text_img2vid", END)
    elif decision == "textimg2img":
        workflow.set_entry_point("textimg2img")
        workflow.add_edge("textimg2img", END)
    elif decision == "textimg2text":
        workflow.set_entry_point("textimg2text")
        workflow.add_edge("textimg2text", END)
    
    
    return workflow.compile()


    