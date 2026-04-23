from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
import schema 
from dotenv import load_dotenv
import os 
load_dotenv()

client=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE")
)
model= ChatHuggingFace(llm=client)
#step 1
parser1=PydanticOutputParser(pydantic_object=schema.StudentInput)
#step 2
parser2=PydanticOutputParser(pydantic_object=schema.StudentMarks)
#step 3 
parser3=PydanticOutputParser(pydantic_object=schema.passOrfail)


#step 1 

prompt1 =PromptTemplate(template="""
system: you are helpful AI assistant which help to gererate output in structure format.
{structure_output}
user:{user_input}
""",input_variables=["user_input"],partial_variables={"structure_output":parser1.get_format_instructions()})

#step 2

prompt2=PromptTemplate(
    template="""
You are a helpful AI assistant.

Your task is to calculate the student's overall percentage based on the subjects and marks provided by the user.

Instructions:
1. The user will provide subject names and marks scored in each subject.
2. Assume each subject is out of 100 unless stated otherwise.
3. Calculate:
   - Total marks obtained
   - Total maximum marks
   - Percentage
4. Return the result clearly and neatly.
5. If subject names are included, mention them in the response.
6. If data is incomplete, politely ask for missing marks.
7. Return ONLY valid JSON.
Output Format:
{structure_output}
user :{user_input}
""",
input_variables=["user_input"],
partial_variables=({"structure_output":parser2.get_format_instructions()})
)

#step 3 

prompt3=PromptTemplate(
    template="""
You are a helpful AI assistant.

Your task is to tell the student's pass or fail based on the parcentage provided by the user.

Instructions:
1. The user will provide parcentage.
2. you just have to answer in pass or fail
4. Return the result clearly and neatly.
5. if user had more then 65 parcentage then only he/she will pass.
6. If data is incomplete, politely ask for missing parcentage.
Output Format:
{structure_output}
user :{user_input}
""",
input_variables=["user_input"],
partial_variables=({"structure_output":parser3.get_format_instructions()})
)


chain=prompt1|model|parser1|prompt2|model|parser2|prompt3|model|parser3

response=chain.invoke({"user_input":"hii my name is shubham,i just got my result i got 68 in math, 90 in english, 75 in hindi, 95 in science"})
print(response)