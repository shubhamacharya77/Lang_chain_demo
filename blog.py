from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnableParallel
from dotenv import load_dotenv
load_dotenv()
import schema
model=ChatOllama(
    model="phi3:mini"
)
def heheh(data):
    result= data.upper()
    return result
parser= StrOutputParser()
prompt= PromptTemplate(
    template="""
You are an expert AI Blog Writer and Content Strategist.
Your task is to take a input provided by the user.
user: {user_input}
""",
input_variables=["user_input"]
)
prompt1=PromptTemplate(
    template="""
You are an expert AI Blog Writer and Content Strategist.
Your task is to generate a high-quality, engaging, SEO-friendly blog title based on the topic provided by the user.
Instructions:

1. Understand the user topic carefully.
2. Create a catchy and professional blog title.
3. Keep tone professional and reader-friendly.
4. title must be complate in not more then 50 words
user: {user_input}
""",
input_variables=["user_input"]
)

prompt2 =PromptTemplate(
    template="""
You are an expert content writer skilled at creating engaging blog introductions.
Your task is to write a powerful and reader-friendly introduction based on the user's input topic.

Instructions:

1. Understand the user's topic clearly.
2. Write an attention-grabbing opening sentence.
3. Introduce the topic in a simple and engaging way.
4. Make the reader curious to continue reading.
5. Keep the tone professional, clear, and interesting.
6. Use easy-to-understand language.
7. Keep the introduction concise (80–150 words unless asked otherwise).
8. Do not write the full blog, only the introduction.

user: {user_input}
""",
input_variables=["user_input"])

prompt3 =PromptTemplate(
    template="""
You are an expert SEO strategist and keyword research specialist.
Your task is to generate relevant, high-ranking SEO keywords based on the user's input topic.
                        
Instructions:
1. Understand the user's topic clearly.
2. Generate keywords directly related to the topic.
3. Include a mix of:
   - Short-tail keywords
   - Long-tail keywords
   - Related search phrases
4. Focus on keywords users are likely to search.
5. Keep keywords relevant, natural, and useful.
6. Avoid duplicate keywords.
7. Return 10–15 keywords unless otherwise requested.
8. Output only keywords, no explanations.

user: {user_input}
""",
input_variables=["user_input"])


prompt4 =PromptTemplate(
    template="""
You are an expert AI Blog Writer and Content Strategist.
Your task is to write an full Blog post in the basis of user input which contian blog title,introduction and SEO keywords
                        
Instructions:
1. Understand the user's input clearly.
2. Generate full Blog post related to the topic.
3. Keep keywords relevant, natural, and useful.
4. Avoid duplicate keywords.
user:
title: {input_title}
introduction:{input_introduction}
SEO: {input_SEO}
""",
input_variables=["input_title","input_introduction","input_SEO"])


parallel_chain=RunnableParallel({
    "input_title": prompt1|model|parser|RunnableLambda(heheh),
    "input_introduction": prompt2|model|parser,
    "input_SEO":prompt3|model|parser  
})
chain=prompt|model|parser|parallel_chain|prompt4|model|parser
chain.get_graph().print_ascii()
response=chain.invoke("my new BMW car")
print(response)