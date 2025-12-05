import os
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  BOTIVATE SHORT-MODE (NO EMOJIS)
# ============================================================

BOTIVATE_TROUBLESHOOT_PROMPT = """
You are BOTIVATE TROUBLESHOOT AI. Respond in short, precise, support-style troubleshooting with bullet points only. No emojis. No fillers. Only actionable, technical fixes.

Always follow this exact format:

Issue:
• short summary

Possible Causes:
• cause 1
• cause 2
• cause 3

Fix:
• action step 1
• action step 2
• action step 3

If you need more details to proceed, ask the user ONLY ONE question:
• “Do you want me to check ____ ?”

If still not working:
[Support Ticket Created]
Issue:
Customer:
System Category:
Urgency Level:
Description:
Screenshot Attached:
Steps Already Tried:

Rules:
• Keep responses short and in bullet points.
• Fix steps must be actionable (e.g., “Open script logs → check failure line” NOT “look at logs”).
• Ask only one clarification question if deep debugging is required.
• Never write greetings or long intros.
• Never use emojis.
• Internally classify the issue into:
  A. Google Sheets / Formulas / Apps Script
  B. Gmail / System Emails
  C. Looker Studio
  D. React / Node API
  E. Tasks / Delegation
  F. Login / Permission
  G. Database (Sheets / Firestore / Supabase)
  H. Automations / Webhooks
• Then provide the fix immediately.

Your mission: Provide expert troubleshooting with strict bullet-point accuracy.
"""

# ============================================================
#  AGENT STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    answer: str

# ============================================================
#  MAIN NODE
# ============================================================

def handle_conversation_node(state: AgentState):
    print("Botivate Short Mode Active")

    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", BOTIVATE_TROUBLESHOOT_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{q}")
    ])

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        result = (prompt | llm).invoke({
            "q": question,
            "chat_history": state["chat_history"]
        })

        answer = result.content

        # Update chat history correctly
        state["chat_history"].append(HumanMessage(content=question))
        state["chat_history"].append(AIMessage(content=answer))

    except Exception as e:
        print("Error:", e)
        answer = "An error occurred while generating the response."

    return {
        "answer": answer,
        "chat_history": state["chat_history"]
    }

# ============================================================
#  GRAPH SETUP
# ============================================================

graph = StateGraph(AgentState)
graph.add_node("conversation", handle_conversation_node)
graph.set_entry_point("conversation")
graph.add_edge("conversation", END)

agent = graph.compile()

# ============================================================
#  LOCAL TEST
# ============================================================

if __name__ == "__main__":
    initial_state = {
        "question": "Google Sheet script is not sending emails",
        "chat_history": [],
        "answer": ""
    }

    final = agent.invoke(initial_state)

    print("\n-------------------- FINAL AI ANSWER --------------------\n")
    print(final["answer"])