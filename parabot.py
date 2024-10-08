import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()
tok = os.getenv("HF_TOKEN")

def getLLamaresponse(input_text, no_words, blog_style):
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", max_length=64, temperature=0.5, Token=tok)
    
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Vimal's ParaGen vr.01", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.markdown("""
    <style>
        .stApp {
            background-image: url('https://www.webio.com/assets/components/layout/images/thought-box-images/blogs/design-chatbots.png');
            background-size: cover;
        }
        h1 {
            color: #FFFFFF;
            font-size: 60px;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput>div>input {
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

header
st.header("Generate Articles from Vimal's ParaBOT 🤖")


input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")


if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))

st.markdown("<h1>Vimal's ParaBOT using Mistral Model</h1>", unsafe_allow_html=True)
