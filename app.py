import re
import typing
import pandas as pd
import umap
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from langchain.chains import LLMChain, SequentialChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import openai
from langchain.callbacks.base import BaseCallbackHandler
import os
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = ""
# Streamlit configurations
st.markdown("<h1 style='text-align: center;'>Get me a Nobel prize</h1>", unsafe_allow_html=True)


# Session state initialization
if 'calc_hypotheses' not in st.session_state:
    st.session_state['calc_hypotheses'] = []

if 'calc_embeddings' not in st.session_state:
    st.session_state['calc_embeddings'] = []

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

if 'current_stage' not in st.session_state:
    st.session_state['current_stage'] = "Ready"
if 'total_iterate' not in st.session_state:
    st.session_state['total_iterate'] = 5
if 'chat_history_download' not in st.session_state:
    st.session_state['chat_history_download'] = ""

class NewChainHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: typing.Dict[str, typing.Any], prompts: typing.List[str], **kwargs: typing.Any) -> typing.Any:
        """Run when chain starts running."""
        # Assuming serialized contains the chain name, modify as needed
        
        #chain_name_placeholder.text_input("Current Chain Name:", chain_name)
        stages = ["Initial solutions", "One sentence summary"]
        step_string = next((s for s in prompts if "Step" in s), None)
        if step_string:
            # Extract the step number using regex
            match = re.search(r"Step (\d+)", step_string)
            if match:
                step_number = match.group(1)  # This gives us the number after "Step"
                st.session_state['current_stage'] = stages[int(step_number)-1]
            else:
                st.session_state['current_stage'] = step_string
        else:
            st.session_state['current_stage'] ="LLM has started, but no step information found."
        
        chain_name_placeholder.write(f"Current stage: {st.session_state['current_stage']:}")
     


# Sidebar configurations
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Check if an API key is provided
if user_api_key:
    try:
        # Set the API key for OpenAI
        openai.api_key = user_api_key
        #os.environ["OPENAI_API_KEY"] = user_api_key
        st.sidebar.success("OpenAI API key has been set successfully!")
        embeddings_model = OpenAIEmbeddings()
    except Exception as e:
        st.sidebar.error(f"Error setting API Key: {e}")
else:
    st.sidebar.warning("Please enter an API key to proceed.")
st.sidebar.title("Tools and information")
model_name = st.sidebar.radio("Choose a model:", ("gpt-4", "gpt-3.5-turbo"))
# Number input for total_iterate
st.session_state['total_iterate'] = st.sidebar.number_input('Set Total Iterations:', min_value=1, value=5, step=1)

chain_name_placeholder = st.sidebar.empty()
chain_name_placeholder.write(f"Current stage: {st.session_state['current_stage']}")
st.sidebar.title("Examples")

# Examples
example_texts = ["I trained a logistic regression model on the Titanic dataset. The most important features were: Pclass, Sex, Age, SibSp,Parch. Suppose that you are a data scientist and expert on the Titanic catastrophe. Come up with 3 testable hypotheses which explains the difference in survival of the passengers. These hypotheses must be completely non-trivial, and something which can be only discovered when combining humanity’s all scientific knowledge with the results from the logistic regression model presented above. The more complex the hypotheses are, the better. Below you can see an example of the output I expect: HYPOTHESIS NAME: STEP-BY-STEP reasoning using your data scientist and Titanic catastrophe expertise and the top features from the model, including their relationship with each other"]
for example in example_texts:
    st.sidebar.text(example)

st.sidebar.title("Generated and stored hypotheses")
for hypoth in st.session_state['calc_hypotheses']:
    st.sidebar.text(hypoth)

# User input processing
user_input = st.sidebar.text_area("Input new hypotheses, delimited by '.':")
if user_input:
    new_hypotheses = user_input.split('.')
    
    for hypoth in new_hypotheses:
        if hypoth.strip() not in st.session_state['calc_hypotheses']:
            st.session_state['calc_hypotheses'].append(hypoth.strip())
            embedded_query = embeddings_model.embed_query(hypoth.strip())
            st.session_state['calc_embeddings'].append(embedded_query)
    df = pd.DataFrame(st.session_state['calc_embeddings'])
    if len(st.session_state['calc_embeddings']) > 3:
        fit = umap.UMAP()
        reduced_data = pd.DataFrame(fit.fit_transform(df))
        fig = px.scatter(x=reduced_data[0], y=reduced_data[1], hover_name=st.session_state['calc_hypotheses'])
        st.plotly_chart(fig)

# Model selection
if model_name == "gpt-3.5-turbo":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

if len(msgs.messages) == 0:
    msgs.add_ai_message("Describe your scientific problem or output from a machine learning algorithm (or copy-paste from the Examples)! A minimum of 4 generated hypotheses is required for the automatic visualization.")

view_messages = st.expander("View the message contents in session state")

# Chat model
class DocumentInput(BaseModel):
    question: str = Field()


if st.button('Initialize Chat History For Download'):
    st.session_state['chat_history_download'] = st.session_state['chat_history_download']
    st.write("Chat history initialized!")

st.download_button(
    label="Download Chat History",
    data=st.session_state['chat_history_download'],
    file_name="chat_history.txt",
    mime="text/plain"
)


# Prompts and chains are defined here
template ="""
Step 1 :
 
I have a problem related to {input}. Could you brainstorm three distinct hypotheses? These hypotheses must be completely non-trivial, and something which can be only discovered when combining humanity’s all scientific knowledge with the results from the machine learning method presented above. The more complex the hypotheses are, the better. Give them very detailed, informative hypothesis names. It is very important to be as detailed as possbile, especially with the methods you suggest. Please consider a variety of factors such as {perfect_factors}. At the end rank the three hypotheses by scientific soundness and novelty
A:
"""

prompt = PromptTemplate(
    input_variables=["input","perfect_factors"],
    template = template                      
)


class NamedLLMChain(LLMChain):
    def __init__(self, *args, name=None, **kwargs):
        super(NamedLLMChain, self).__init__(*args, **kwargs)
        self.__dict__['name'] = name or 'Unnamed Chain'
    
    def __setattr__(self, key, value):
        if key == "name":
            self.__dict__['name'] = value
        else:
            super(NamedLLMChain, self).__setattr__(key, value)
if(user_api_key):
    chain1 = NamedLLMChain(
        llm=ChatOpenAI(temperature=0.3, model=model, streaming=True,callbacks=[NewChainHandler()]),
        prompt=prompt,
        output_key="solutions",name = "Chain 1"
    )





    template ="""
    Step 2:

    Based on the highest ranked hypothesis, summarize the core of the hypothesis in one long sentence, and the second part of the sentence should describe why the hypothesis is thought to be true related to the measured variables in the model connected to knowledge about the world. It is very important to be only one sentence long and don't say anythin about this is a core hypothesis etc, just give me the core hypothesis
    {solutions}

    One sentence summary:"""

    prompt = PromptTemplate(
        input_variables=["solutions"],
        template = template                      
    )

    chain2 = LLMChain(
        llm=ChatOpenAI(temperature=0.3, model=model, streaming=True,callbacks=[NewChainHandler()]),
        prompt=prompt,
        output_key="one_sentence"
    )






    overall_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["input", "perfect_factors"],
        output_variables=["one_sentence"],
        verbose=True,callbacks=[NewChainHandler()]
    )





for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
       
        
        for i in range(st.session_state['total_iterate']):
            
            response = overall_chain({"input":prompt, "perfect_factors":"Try to adhare to the following format: HYPOTHESIS NAME: STEP-BY-STEP reasoning using your scientific field specific knowledge and the top features from the model, including their relationship with each other:  IN-SILICO testing of the hypothesis: REAL WORLD testing of the hypothesis:. Apart of these, the generated hypotheses should be entirely different to the hypotheses contained here: (and always reflect how they are different, with a section called RELATION TO PREVIOUS HYPOTHESES:): " + str(st.session_state['calc_hypotheses'])+ ". Again, it is very important not to generate new hypotheses conceptually related to these or if they contain the same variable names from the input." },callbacks=[st_callback])
            
            st.write(response)
            st.session_state['chat_history_download'] = str(st.session_state['chat_history_download']) + '\n' + str(response)
            st.session_state['model_name'].append(model_name)

            st.session_state['calc_hypotheses'].append(response["one_sentence"])
            
            st.sidebar.text(response["one_sentence"])
            
            embedded_query = embeddings_model.embed_query(response["one_sentence"])
            st.session_state['calc_embeddings'].append(embedded_query)
            df = pd.DataFrame(st.session_state['calc_embeddings'])
            print(st.session_state['calc_hypotheses'])
            if(len(df)>4):
                
                fit = umap.UMAP()
                reduced_data = pd.DataFrame(fit.fit_transform(df)) 
                
                fig = px.scatter(x=reduced_data[0], y=reduced_data[1], hover_name=st.session_state['calc_hypotheses'])
                st.plotly_chart(fig)
            
            
            
            
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
    

repo_url = "https://github.com/Paureel/get_me_a_nobel_prize_streamlit"
st.sidebar.markdown(f"Find the code [here]({repo_url})")