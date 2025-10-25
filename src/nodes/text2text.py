# File: nodes/text2text.py


from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..graph.state import State
from typing import List, Dict, Any
import os


SYSTEM_PROMPT_VI = """
Bạn là một trợ lý hữu ích hãy trả lời theo yêu cầu của người dùng.
"""


SYSTEM_PROMPT_EN = """
You are a helpful assistant, help answer according to user demand.
"""


def text2text(state: State, config: RunnableConfig) -> State:
   """NODE: Trả lời câu hỏi."""
   print("--- Thực hiện Node: text2text ---")


   question = state["t2t_question"]


   prompt_messages = [
       {"role": "developer", "content": SYSTEM_PROMPT_VI} if os.getenv("LANGUAGE", "EN") == "VI" else {"role": "developer", "content": SYSTEM_PROMPT_EN},
       {"role": "user", "content": question},
   ]
  
   bot = config["configurable"]["bot"]


   response = bot.create_chat_completion(
       model=os.getenv("TEXT_MODEL_NAME"),
       messages=prompt_messages,
   )
      
   print(f"Trả lời: {response}")
  
   state["t2t_answer"] = response['choices'][0]["message"]["content"]
  
   return state
