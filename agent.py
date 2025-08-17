import os
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain import hub
# from dotenv import load_dotenv

# load_dotenv()

# Get API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key = os.environ.get("SERPER_API_KEY")

# Check if API keys are set
if not openai_api_key or not serper_api_key:
    print("Error: Please set the OPENAI_API_KEY and SERPER_API_KEY environment variables.")
else:
    # Initialize the language model
    llm = OpenAI(temperature=0)

    # Initialize the tools
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    tools = [GoogleSerperRun(api_wrapper=search)]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Start the conversation
    print("AI Agent is ready. Type 'quit' to exit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response}")
