import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler

## set up the streamlit app
st.set_page_config(page_title="Text to Math problem Solver and Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

llm=ChatGroq(api_key=groq_api_key, model="gemma2-9b-It")

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Tool for searching the internet to find the various information on the topics mentioned"
)

## Inisialize the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression"
)

prompt="""
"You are an agent tasked for solving user's mathematical questions. Logically arrive at the solution and provide a detailed explaination and display it point wise for the question below.
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(input_variables=['question'],template=prompt)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## Initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a Math chatbot who can answer all your math questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])




## Interaction
question=st.text_area("Enter your question:","I had 12 bananas, 15 oranges, and 9 apples. I used 1/3 of the bananas to bake a cake and gave 5 oranges to a friend. Then I sold half of the remaining apples. Later, I bought 6 bananas, 3 apples, and twice as many oranges as I gave away. How many fruits do I have now in total?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})

            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter a question")