from agno.agent import Agent
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.models.google import Gemini


from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 科学上网 proxy
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"


hn_researcher = Agent(
    name="HackerNews Researcher",
    tools=[HackerNewsTools()],
)
article_reader = Agent(
    name="Article Reader",
    tools=[Newspaper4kTools()],
)

team = Team(members=[hn_researcher, article_reader],
                model=Gemini(id="gemini-2.5-flash", api_key=GEMINI_API_KEY)
            )
# 没有设置model属性，那么就使用 openai作为模型的mode provider
# 默认设置模型为： self.model = OpenAIChat(id="gpt-4o")
team.print_response("Research AI trends and summarize the top articles")
