from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using SendGrid API."""
    try:
        sg = SendGridAPIClient(os.getenv("sendgrid_api_key"))
        message = Mail(
            from_email="rakuhasan2017@gmail.com",
            to_emails=to,
            subject=subject,
            html_content=body
        )
        response = sg.send(message)
        return f" Email sent to {to}, status: {response.status_code}"
    except Exception as e:
        return f" Error sending email: {str(e)}"

send_email_tool = StructuredTool.from_function(send_email)


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    huggingfacehub_api_token=os.getenv("huggingfacehub_api_token"),
    temperature=0.5,
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI that can send emails using the following tool:\n\n{tools}\n\n"
     "Use the correct tool with proper arguments."),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])


agent = StructuredChatAgent.from_llm_and_tools(
    llm=model,
    tools=[send_email_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[send_email_tool], verbose=True)


response = agent_executor.invoke({
    "input": "Send an email to anotherrakuhasan2017@gmail.com with subject 'LangChain Project' and body 'I have completed the task.'"
})

print(response)
