# AIAgentGooglePlaces

Create a basic agent to call external tools for search and google places

# Purpose

To answer prompts that require current information that may not be in the foundational model.

# Features

Gradio UI component to control the agent <br>
Langchain <br>
Uses gemini-2.0-flash LLM from Google <br>
Uses SerpAPIWrapper to call search to augment LLM <br>
Uses GooglePlacesAPIWrapper to call Google Places API to get addresses <br>

# Installation

Create PYCharm AIAgentGooglePlaces project locally in a chosen virtual environment <br>
Add dependencies to virtual environment as described in requirements.txt <br>
Add in src files main.py, GooglePlacesAgent.py <br>
Modify GooglePlacesAgent.py to update your GOOGLE API, SERPAPI API, and GPLACES API keys. <br>


# Usage

Run main.py from PYCharm project <br>
![Run project in PyCharm](RunMainInPyCharm.png) <br>
System will create a local URL "* Running on local URL:  http://127.0.0.1:7860" <br>
Click on link to instance the Gradio UI in your default browser <br>
Click Execute Prompt button to run the agent to create the final answer <br>
![Run agent in browser](ClickExecutePromptButtonInBrowser.png) <br>
System displays final answer that requires use of an external internet search tool and a google places tool <br>
![Final Answer](FinalAnswer.png) <br>

