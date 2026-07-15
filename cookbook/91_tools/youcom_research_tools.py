import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.youcom.research import YouResearchTools  # your new file

load_dotenv()

agent = Agent(
    id="You_Research_Agent",
    model=OpenAILike(
        id="gpt-5.4-mini",
        api_key=os.getenv("Azure_API_KEY"),
        base_url=os.getenv("Azure_Endpoint"),
        request_params={"parallel_tool_calls": False},
    ),
    tools=[
        YouResearchTools(
            format="markdown",       # readable output, not raw JSON
            show_results=True,       # <- KEY: logs full response before agent sees it
            research_effort="standard",
            timeout=120,             # research is slower than search, don't cut it short
        ),
    ],
    tool_call_limit=3,               # research shouldn't need many calls
    debug_mode=True,
    debug_level=2,                   # shows tool call params + raw response
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "Which global cities improved air quality the most over the past 10 years, and what measurable actions contributed?",
        markdown=True,
    )
