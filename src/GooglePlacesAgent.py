from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_community import GooglePlacesAPIWrapper
from typing import TypedDict
from typing import Annotated
import os
import warnings
import gradio as gr

warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

os.environ["GOOGLE_API_KEY"] = "YOUR GOOGLE_API_KEY"
os.environ["SERPAPI_API_KEY"] = "YOUR SERPAPI_API_KEY"
os.environ["GPLACES_API_KEY"] = "YOUR GPLACES_API_KEY"

# Define the state for the agent
class State(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def search(query: str):
    """Use the SerpAPI to run a Google Search."""
    local_search = SerpAPIWrapper()
    return local_search.run(query)


@tool
def places(query: str):
    """Use the Google Places API to run a Google Places Query."""
    local_places = GooglePlacesAPIWrapper()
    return local_places.run(query)


search_tools = [search]
places_tools = [places]


class ewriter():
    def __init__(self, memory, system_prompt, model_prompt):
        self.system_prompt = system_prompt
        self.model_prompt = model_prompt

        # Define a new graph
        self.workflow = StateGraph(State)
        self.llm_prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("placeholder", "{messages}")]
        )
        self.llm_search = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                                 temperature=0,
                                                 max_tokens=8196,
                                                 timeout=None,
                                                 max_retries=2,
                                                 max_output_tokens=8196)
        self.llm_with_search_tools = self.llm_search.bind_tools(search_tools)
        self.llm_search_gen = (self.llm_prompt | self.llm_with_search_tools)
        self.workflow.add_node("search_gen", self.search_gen_node)
        self.llm_with_places_tools = self.llm_search.bind_tools(places_tools)
        self.llm_places_gen = (self.llm_prompt | self.llm_with_places_tools)
        self.workflow.add_node("places_gen", self.places_gen_node)
        self.llm_gen = (self.llm_prompt | self.llm_search)

        self.workflow.add_node("final_answer_gen", self.answer_gen_node)

        # Specify the edges between the nodes
        self.workflow.add_edge(START, "search_gen")
        self.workflow.add_edge("search_gen", "places_gen")
        self.workflow.add_edge("places_gen", "final_answer_gen")
        self.workflow.add_edge("final_answer_gen", END)
        self.memory = memory
        self.graph = self.workflow.compile(checkpointer=self.memory)

    def search_gen_node(self, state: State) -> dict[str, list[AIMessage]]:
        ai_msg = self.llm_search_gen.invoke(state)
        messages = state['messages']
        messages.append(ai_msg);
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"search": search}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
        return {"messages": messages}

    def places_gen_node(self, state: State) -> dict[str, list[AIMessage]]:
        ai_msg = self.llm_places_gen.invoke(state)
        messages = state['messages']
        messages.append(ai_msg);
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"places": places}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
        return {"messages": messages}

    def answer_gen_node(self, state: State) -> dict[str, list[AIMessage]]:
        message = self.llm_gen.invoke(state)
        return {"messages": [message]}


class writer_gui():
    def __init__(self, graph, system_prompt, user_prompt, share=False):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()

    def run_agent(self, start, topic, stop_after):
        if start:
            self.iterations.append(0)
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        model_qry = {'task': topic,
                     "messages": [("user", topic)]}
        self.response = self.graph.invoke(model_qry, self.thread)
        self.iterations[self.thread_id] += 1
        self.partial_message += str(self.response)
        self.partial_message += f"\n------------------\n\n"
        # print("Hit the end")
        return

    def get_disp_state(self, ):
        current_state = self.graph.get_state(self.thread)
        nnode = current_state.next
        return nnode, self.thread_id

    def get_state(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            nnode, self.thread_id, rev, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""

    def get_content(self, ):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            nnode, thread_id, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(item for item in content) + "\n\n")
        else:
            return ""

    def update_hist_pd(self, ):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            nnode = state.next
            st = f"{tid}:{nnode}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:last_node:next_node:rev:thread_ts",
                           choices=hist, value=hist[0], interactive=True)

    def find_config(self, thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return (None)

    def copy_state(self, hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread.
             This copies an old state to a new current state.
        '''
        thread_ts = hist_str.split(":")[-1]
        config = self.find_config(thread_ts)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)  # should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        nnode = new_state.next
        return nnode, new_thread_ts

    def update_thread_pd(self, ):
        # print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id, interactive=True)

    def switch_thread(self, new_thread_id):
        # print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return

    def modify_state(self, key, asnode, new_state):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        '''
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        return

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm', text_size="sm")) as demo:

            def updt_disp():
                ''' general update display on state change '''
                json_str: str = ''
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata['step'] < 1:  # ignore early states
                        continue
                    if "thread_ts" in state.config:
                        s_thread_ts = state.config['configurable']['thread_ts']
                    else:
                        s_thread_ts = ''
                    s_tid = state.config['configurable']['thread_id']
                    s_nnode = state.next
                    st = f"{s_tid}:{s_nnode}:{s_thread_ts}"
                    hist.append(st)
                if not current_state.metadata:  # handle init call
                    return {}
                else:
                    if len(self.response) < 1:
                        for msg in current_state[0]['messages']:
                            if len(msg.content) > 0 and isinstance(msg, AIMessage):
                                json_str += msg.content
                                json_str += '\n'
                    else:
                        for msg in current_state[0]['messages']:
                            if len(msg.content) > 0 and isinstance(msg, AIMessage):
                                json_str += msg.content
                                json_str += '\n'
                    return {
                        topic_bx: current_state.values["task"],
                        threadid_bx: self.thread_id,
                        live: json_str,
                        thread_pd: gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id,
                                               interactive=True),
                        step_pd: gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts",
                                             choices=hist, interactive=True),
                    }

            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ['plan', 'draft', 'critique']:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if 'content' in state.values:
                        for i in range(len(state.values['content'])):
                            state.values['content'][i] = state.values['content'][i][:20] + '...'
                    if 'writes' in state.metadata:
                        state.metadata['writes'] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                # print(f"vary_btn{stat}")
                return gr.update(variant=stat)

            with gr.Tab("Model Prompt"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="Model Prompt", value=self.user_prompt, lines=10, max_lines=10)
                    gen_btn = gr.Button("Execute Prompt", scale=0, min_width=80, variant='primary')
                    cont_btn = gr.Button("Continue Execution", scale=0, min_width=80, visible=False)
                with gr.Row():
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=10, visible=False)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(checks, label="Interrupt After State", value=checks, scale=0,
                                                  min_width=400, visible=False)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads, interactive=True, label="select thread",
                                                min_width=120, scale=0)
                        step_pd = gr.Dropdown(choices=['N/A'], interactive=True, label="select step", min_width=160,
                                              scale=1)

                live = gr.Textbox(label="Final Answer", lines=25, max_lines=25, show_copy_button=True)

                # actions
                sdisps = [topic_bx, step_pd, threadid_bx, thread_pd, live]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state, [step_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                gen_btn.click(vary_btn, gr.Number("secondary", visible=False), gen_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(True, visible=False),
                                               topic_bx,
                                               stop_after], outputs=[live],
                    show_progress=True).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), gen_btn).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
                cont_btn.click(vary_btn, gr.Number("secondary", visible=False), cont_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(False, visible=False),
                                               topic_bx,
                                               stop_after],
                    outputs=[live]).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
            with gr.Tab("StateSnapShots"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                snapshots = gr.Textbox(label="State Snapshots Summaries")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
            with gr.Tab("LLM System Prompt"):
                system_rules = gr.Textbox(label="System Prompt", lines=25, max_lines=25, value=self.system_prompt)
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
