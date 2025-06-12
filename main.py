import os
import chainlit as cl
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    input_guardrail,
    TResponseInputItem,
)
from agents.run import RunConfig
from pydantic import BaseModel

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# External LLM client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Allowed domain keywords
domain_keywords = {
    'WebDeveloper': ["website", "frontend", "backend", "html", "css", "javascript", "react", "next.js"],
    'DigitalMarketer': ["seo", "marketing", "social media", "ad", "campaign", "ppc", "content strategy"],
    'ContentWriter': ["blog", "article", "content", "write", "copywriting", "post"],
}

# Input guardrail to reject any out-of-domain queries
guardrail_agent = Agent(
    name="TopicGuardrail",
    instructions="Detect if the user query is outside Web Development, Marketing, or Content Writing topics.",
    output_type=BaseModel,
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client)
)

@input_guardrail
async def topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    text = input if isinstance(input, str) else ' '.join(item.content for item in input)
    lower = text.lower()
    allowed = any(
        kw in lower for kws in domain_keywords.values() for kw in kws
    )
    # Use agent reasoning for additional clarity
    result = await Runner.run(guardrail_agent, text, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not allowed,
    )

# Core expert agents
web_dev_agent = Agent(
    name="WebDeveloper",
    instructions="You are a skilled web developer. Provide code samples, architecture decisions, and best practices.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client)
)

digital_marketer_agent = Agent(
    name="DigitalMarketer",
    instructions="You are a creative digital marketer. Craft marketing strategies, SEO recommendations, ad copy, and social media plans.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client)
)

content_writer_agent = Agent(
    name="ContentWriter",
    instructions="You are a talented content writer. Produce clear, engaging, and SEO-friendly content.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client)
)

# Manager agent with guardrails
data_model = BaseModel
manager = Agent(
    name="Manager",
    instructions=(
        "Delegate user queries to WebDeveloper, DigitalMarketer, or ContentWriter based on domain keywords. "
        "Reject any request outside these domains."
    ),
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client),
    input_guardrails=[topic_guardrail],
    output_type=data_model,
)

# Session storage
user_sessions: dict[str, str] = {}

@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hello! Choose: Web Development, Marketing, or Content Writing.").send()

@cl.on_message
async def main(message: cl.Message):
    user_id = message.author.id
    text_lower = message.content.strip().lower()

    # Domain selection
    if user_id not in user_sessions:
        if 'web' in text_lower or 'development' in text_lower:
            user_sessions[user_id] = 'WebDeveloper'
            await cl.Message(content="Great! You chose Web Development.").send()
        elif 'marketing' in text_lower:
            user_sessions[user_id] = 'DigitalMarketer'
            await cl.Message(content="Great! You chose Marketing.").send()
        elif 'content' in text_lower or 'writing' in text_lower:
            user_sessions[user_id] = 'ContentWriter'
            await cl.Message(content="Great! You chose Content Writing.").send()
        else:
            await cl.Message(content="Please choose one: Web Development, Marketing, or Content Writing.").send()
        return

    # Handle request with guardrail
    await cl.Message(content="⏳ Processing your request...").send()
    try:
        routed = user_sessions[user_id]
        prompt = f"Assign to {routed}: {message.content}"
        result = Runner.run_sync(manager, prompt, run_config=RunConfig(model=manager.model, model_provider=external_client))
        output_text = str(result.final_output)
        # Filter out default guardrail prefix if present
        guard_prefix = "Guardrail:"
        if output_text.startswith(guard_prefix):
            await cl.Message(content="Sorry, I can only help with Web Development, Marketing, or Content Writing.").send()
        else:
            await cl.Message(content=output_text).send()
    except InputGuardrailTripwireTriggered:
        await cl.Message(content="Sorry, I can only help with Web Development, Marketing, or Content Writing.").send()
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
    finally:
        user_sessions.pop(user_id, None)

if __name__ == "__main__":
    cl.run()
