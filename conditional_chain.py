from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
import schema
model=ChatOllama(
    model="phi3:mini"
)
parser= PydanticOutputParser(pydantic_object=schema.Sentiment)
parser1=StrOutputParser()
prompt1=PromptTemplate(
    template="""
You are an expert AI feedback sentiment classifier.
Your task is to analyze user feedback and classify the sentiment as either Positive or Negative.

Instructions:
Read the feedback carefully.
Determine the overall sentiment expressed.
Respond with only one word:
positive → if the feedback expresses satisfaction, praise, happiness, or approval.
negative → if the feedback expresses dissatisfaction, complaint, frustration, or disapproval.
Do not provide explanations or extra text.
If the sentiment is mixed, choose the dominant sentiment.
{structure_output}
user :{user_input} 
""",
input_variables=["user_input"],partial_variables={"structure_output":parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="""
You are an expert customer support response writer.

Your task is to generate a warm, professional, and appreciative response to positive customer feedback.

Instructions:
Read the customer’s positive feedback carefully.
Thank the customer sincerely.
Acknowledge the specific praise or experience they mentioned.
Keep the tone friendly, professional, and genuine.
Keep the response concise (2–4 sentences).
Do not sound robotic or overly repetitive.
Do not include placeholders unless requested.
Output Rules:
Return only the response message.
Do not include labels, explanations, or formatting.
user :{user_input} 
""",
input_variables=["user_input"]
)
prompt3=PromptTemplate(
    template="""
You are an expert customer support response writer.

Your task is to generate a calm, empathetic, and professional response to negative customer feedback.

Instructions:
Read the customer’s negative feedback carefully.
Acknowledge their frustration or disappointment.
Apologize sincerely for the poor experience.
Address the issue respectfully without being defensive.
Express willingness to improve or make things right.
Keep the tone professional, supportive, and solution-focused.
Keep the response concise (2–4 sentences).
Do not blame the customer or make excuses.
Output Rules:
Return only the response message.
Do not include labels, explanations, or formatting.
user :{user_input} 
""",
input_variables=["user_input"]
)

# response=chain.invoke({"user_input":"The service was amazing and fast."}) 


branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive" ,prompt2|model|parser1),
    (lambda x:x.sentiment=="negative",prompt3|model|parser1 ),
    RunnableLambda(lambda x:" sentiment not found !")
)

chain=prompt1|model|parser| branch_chain
response= chain.invoke({"user_input":"The service was worst and slow."})
print(response)
