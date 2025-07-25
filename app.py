from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from fastapi.responses import FileResponse
from fpdf import FPDF
import tempfile
from pydantic import BaseModel

load_dotenv()


memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(
    max_results=4,
)

tools = [search_tool]

# Set the API key as environment variable
llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)

# llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(tools=tools)

def generate_summary_from_messages(messages):
    summary = {
        "project_name": None,
        "platform": None,
        "features": [],
        "tech_stack": [],
        "budget": None,
        "timeline": None,
        "notes": []
    }

    for msg in messages:
        content = msg.get("content", "").lower()

        if "project name" in content:
            summary["project_name"] = msg.get("content")
        elif any(word in content for word in ["web", "mobile", "android", "ios"]):
            summary["platform"] = msg.get("content")
        elif "feature" in content or "include" in content or "want" in content:
            summary["features"].append(msg.get("content"))
        elif "technology" in content or "stack" in content or "framework" in content:
            summary["tech_stack"].append(msg.get("content"))
        elif "budget" in content or "$" in content or "pkr" in content:
            summary["budget"] = msg.get("content")
        elif "timeline" in content or "weeks" in content or "days" in content:
            summary["timeline"] = msg.get("content")
        else:
            summary["notes"].append(msg.get("content"))

    # 🧠 Build a well-formatted summary string
    formatted = f"""✅ **Project Summary**

📌 **Project Name:** {summary['project_name'] or 'Not specified'}
🖥️ **Platform:** {summary['platform'] or 'Not specified'}

🧩 **Key Features:**
{''.join(f"- {f}\n" for f in summary['features']) if summary['features'] else 'No features listed.'}

🛠️ **Preferred Tech Stack:**
{''.join(f"- {t}\n" for t in summary['tech_stack']) if summary['tech_stack'] else 'No stack mentioned.'}

💰 **Budget:** {summary['budget'] or 'Not mentioned'}
⏱️ **Timeline:** {summary['timeline'] or 'Not mentioned'}

📝 **Additional Notes:**
{''.join(f"- {n}\n" for n in summary['notes']) if summary['notes'] else 'No extra notes.'}
"""

    return formatted


VAGUE_RESPONSES = ["i don't know", "maybe", "anything", "not sure", "you decide", "idk", "whatever"]

def is_vague(text: str) -> bool:
    text = text.lower()
    return any(vague in text for vague in VAGUE_RESPONSES)


async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result], 
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last message
    tool_calls = state["messages"][-1].tool_calls
    
    # Initialize list to store tool messages
    tool_messages = []
    
    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Handle the search tool
        if tool_name == "tavily_search_results_json":
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)
            
            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_messages.append(tool_message)
    
    # Add the tool messages to the state
    return {"messages": tool_messages}

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if(isinstance(chunk, AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }
        system_prompt = """
        You are a JD Assistant — a professional chatbot that helps clients describe their software project ideas in detail.

        Your goal is to understand the client's project by asking clear and focused questions. Start by greeting them and asking:

        - What is your software project idea?
        - What problem does it aim to solve?
        - Who will use this product or service?

        Based on their answers, follow up with structured questions such as:
        - What are the key features or functions you need?
        - Should it be a website, mobile app, or both?
        - Any technology, tools, or frameworks in mind?
        - Will there be an admin panel or user dashboard?
        - What's your target budget and timeline?

        Stay focused on gathering complete project requirements. If the idea is clear, provide suggestions for:
        - Tech stack (frontend, backend, database, etc.)
        - Architecture
        - Estimated development timeline
        - Any other relevant advice

        If the user asks about current trends or tools, you can use web search. Keep your language clear, concise, and business-friendly.
        """


        
        # Initialize with first message
        events = graph.astream_events(
            {"messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = json.dumps(chunk_content)
            yield f"data: {{\"type\": \"content\", \"content\": {safe_content}}}\n\n"

            
        elif event_type == "on_chat_model_end":
            user_input = message.lower() if isinstance(message, str) else ""
            # Check if there are tool calls for search
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]
            
            if is_vague(user_input):
                clarification = (
                    "It seems your response was a bit unclear. "
                    "Could you please provide more specific details?"
                )
                yield f"data: {{\"type\": \"clarification\", \"content\": \"{clarification}\"}}\n\n"
                
            # Ask for confirmation at the end if conversation seems complete
            if any(x in user_input for x in ["that's all", "done", "complete", "finished"]):
                summary = generate_summary_from_messages(memory.memory_store.get(checkpoint_id, []))
                yield f"data: {{\"type\": \"summary\", \"content\": \"{summary}\"}}\n\n"

                confirmation = (
                    "Does this summary look accurate to you? "
                    "Would you like to proceed with development or make any changes?"
                )
                yield f"data: {{\"type\": \"confirmation\", \"content\": \"{confirmation}\"}}\n\n"
            
            if search_calls:
                # Signal that a search is starting
                search_query = search_calls[0]["args"].get("query", "")
                # Escape quotes and special characters
                safe_query = json.dumps(search_query)
                yield f"data: {{\"type\": \"search_start\", \"query\": {safe_query}}}\n\n"

                
        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            # Search completed - send results or error
            output = event["data"]["output"]
            
            # Check if output is a list 
            if isinstance(output, list):
                # Extract URLs from list of search results
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                
                # Convert URLs to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"
    
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )
    


