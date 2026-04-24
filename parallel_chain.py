from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel
import schema 
from dotenv import load_dotenv
import os 
load_dotenv()


# client=HuggingFaceEndpoint(
#     repo_id="zai-org/GLM-5.1:together",
#     huggingfacehub_api_token=os.getenv("HUGGING_FACE")
# )
# model= ChatHuggingFace(llm=client)
model=ChatOllama(
    model="phi3:mini"
)
#step 1
parser1= StrOutputParser()

prompt1 = PromptTemplate(
    template="""
You are an expert YouTube content strategist and viral title writer.
Your task is to generate highly clickable, engaging, SEO-friendly YouTube video titles based on the user's topic.

Rules:
1. Keep titles under 60 characters when possible.
2. Make titles curiosity-driven and attention-grabbing.
3. Use emotional triggers, power words, or urgency when relevant.
4. Include relevant keywords naturally for SEO.
5. Avoid misleading clickbait.
6. Match the tone of the topic (educational, entertainment, tech, finance, gaming, etc.).
7. Generate 10 unique title options each time.
8. Prioritize titles that increase CTR (Click Through Rate).
user: {user_input}
""",
input_variables=["user_input"]
)

#step 2 
prompt2 = PromptTemplate(
    template="""
You are an expert YouTube content writer and SEO strategist.
Your task is to create engaging, professional, and SEO-optimized video descriptions based on the user's video topic.
Rules:
1. Write clear and compelling descriptions.
2. First 2 lines must grab attention and summarize the video.
3. Naturally include relevant keywords for SEO.
4. Keep tone matching the niche (tech, gaming, education, finance, vlog, etc.).
user: {user_input}
""",
input_variables=["user_input"]
)

#step 3
prompt3 =PromptTemplate( 
    template="""
You are an expert YouTube growth strategist and thumbnail copywriter.

Your task is to create short, bold, high-converting thumbnail text based on the user's video topic.
Rules:
1. Generate 10 unique thumbnail text options.
2. Keep each option between 2–5 words.
3. Use strong emotional or curiosity-driven words.
4. Make text easy to read instantly.
5. Use uppercase for impact when suitable.
6. Avoid long sentences.
7. Match the niche (tech, finance, gaming, fitness, education, vlog, etc.).
8. Focus on clicks, urgency, surprise, or benefit
user: {user_input}
""",
input_variables=["user_input"]
)

chain= RunnableParallel({
    "title": prompt1|model|parser1,
    "content":prompt2|model|parser1,
    "thumbnail":prompt3|model|parser1

})
listofinput=[
      {"user_input": "Python Tutorial"},
    {"user_input": "AI Tools"},
    {"user_input": "Java Roadmap"}
]
response=chain.batch(listofinput)
print(response)