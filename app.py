import langchain_groq
import streamlit as st
import os
from groq import Groq
import nltk
from nltk import pos_tag, word_tokenize
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Download the NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to get grammar information about a word
def get_grammar_info(word):
    tokens = word_tokenize(word)
    pos_tags = pos_tag(tokens)
    return pos_tags

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="AI Grammar & Vocabulary Assistant", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            color: #ffdd59;
        }
        .sidebar .sidebar-content {
            background: #292E49;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Get Groq API key
    api_key = os.getenv("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")

    client = Groq(api_key="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")

    # Display the logo and title
    st.title("AI-Enhanced Vocabulary & Grammar Assistant!")
    st.write("🚀 Learn, Discover, and Master New Words Effortlessly! Start your journey with our smart chatbot. Let's go!")

    # Sidebar options
    st.sidebar.title('Settings')
    system_prompt = st.sidebar.text_area("Customize System Prompt:", "You are a helpful and knowledgeable assistant.")
    model = st.sidebar.selectbox('Choose a Model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Enter a word or sentence:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize the Groq Langchain chat object
    groq_chat = ChatGroq(groq_api_key="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", model_name=model)

    if user_question:
        grammar_info = get_grammar_info(user_question)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        grammar_message = f"🔍 Grammar Info for '{user_question}': {grammar_info}"

        st.session_state.chat_history.append(message)
        st.write(f"**Chatbot:** {response}")
        st.write(f"**{grammar_message}**")

        # Allow user feedback
        st.sidebar.write("Did you find this helpful?")
        st.sidebar.button("👍 Yes")
        st.sidebar.button("👎 No")

    # New feature: Word of the Day
    st.sidebar.write("## 🌟 Word of the Day")
    words = ["serendipity", "ephemeral", "luminous", "zenith"]
    st.sidebar.write(random.choice(words))

    # New feature: Favorite Responses
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    if st.button("Save Response"):
        st.session_state.favorites.append(response)
        st.sidebar.write("✅ Response saved!")

    st.sidebar.write("## ⭐ Your Saved Responses")
    for fav in st.session_state.favorites:
        st.sidebar.write(f"- {fav}")

if __name__ == "__main__":
    main()
