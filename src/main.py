import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from GooglePlacesAgent import ewriter, writer_gui

memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
my_system_prompt = """You are an enthusiast supportive and talkative virtual assistant that answer user questions 
precisely.  If you do not know the answer please say so."""
my_model_prompt = """Which MLB Baseball Team is leading the American League Central Division in 2025? And what is the 
address of the home baseball stadium this team plays in. Show step by step. Please output the final answer in 
JSON format (team, winning_percentage, address)."""

MultiAgent = ewriter(memory, my_system_prompt, my_model_prompt)
app = writer_gui(MultiAgent.graph, my_system_prompt, my_model_prompt)
app.launch()
