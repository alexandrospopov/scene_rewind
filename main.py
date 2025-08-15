import os
import imageio
import replicate
import gradio as gr
from dotenv import load_dotenv
from langfuse import Langfuse, observe, get_client
import numpy as np
import openai 
import json 
from pydantic import BaseModel, conint
from utils import encode_image
import asyncio
import logfire
from agents import (
    Agent,
    function_tool,
    Runner,
    WebSearchTool,
)
import tempfile
from functools import partial, update_wrapper
from visual_search_agent import seach_google_for_images
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

load_dotenv()
class ImageEvaluation(BaseModel):
    score: conint(ge=1, le=10)
    feedback: str

class HistoricalGrounding(BaseModel):
    general_description: str
    building_architecture: str
    roads: str
    transportation: str
    people: str
    people_clothing: str

image_description = """
This image shows a stunning coastal view of a Mediterranean city, likely Nice on the French Riviera. The scene features:

A wide, curving bay with striking turquoise and deep blue waters transitioning beautifully from the shoreline outward.

A long pebble beach with scattered people relaxing, sunbathing, and strolling near the surf.

Palm-lined promenades running parallel to the shore, adding a tropical feel.

A wide coastal road with smooth curves following the bay, flanked by pedestrian and cycling lanes.

A cityscape in the background, with mid-rise buildings painted in warm tones (yellows, reds, and creams), stretching up into the hills.

Clear blue skies that suggest warm, sunny weather.

It’s a postcard-perfect scene capturing both the relaxing beach atmosphere and the vibrant urban energy of the French Riviera."""


historic_description = """
In the 2nd century AD, the crescent-shaped bay that today we know as the Baie des Anges would have looked remarkably similar in outline—its wide sweep of pebbly shore backed by clear, turquoise waters—yet almost entirely undeveloped by permanent seaside structures. Instead of hotels and promenades, the littoral zone would have been bordered by low cliffs and patches of maquis scrub, with fishermen’s shingle huts and simple wooden docks marking the only human presence along the shore.

Along the coast, the principal artery was the Via Julia Augusta, a paved Roman road laid out in the late 1st century BC to link Italia to Hispania. It hugged the contours of the bay, its broad stone slabs worn smooth by the passage of carts and marching legions. The road would have run just above the high-tide line, with occasional way-stations (mansiones) where travelers could rest and change horses ([fr.wikipedia.org](https://fr.wikipedia.org/wiki/Cemenelum?utm_source=chatgpt.com)).

Instead of the elegant, palm-lined Promenade des Anglais, one would have seen groves of olive and fig trees and stands of umbrella pines and cypresses—common Mediterranean species carefully tended for their oil, fruit, and timber. Beyond these, terraced plots of vineyards stretched partway up the hills, their produce carried down to small coastal landing places in flat-bottomed caiques (fishing boats) moored in shallow coves ([en.wikipedia.org](https://en.wikipedia.org/wiki/Nice?utm_source=chatgpt.com)).

Perched on the heights above the bay lay two distinct settlements. On the immediate seafront was Nikaia (Νίκαια), the older Greek foundation of c. 350 BC, which by the 2nd century AD had become a modest fishing and trading port, its stone quay dotted with amphorae-laden ships from Massalia (modern Marseille) or Sardinia. A small temple of Artemis likely crowned its acropolis, overlooking the masts and nets of local mariners ([en.wikipedia.org](https://en.wikipedia.org/wiki/Nice?utm_source=chatgpt.com)).

Further inland, on the hill now known as Cimiez, stood Cemenelum—the Roman civitas capital of the Alpes Maritimae province. Founded in 14 BC by Augustus as a military and administrative center, by the 2nd century it boasted a forum, elaborate public baths, and an amphitheater seating some 5,000 spectators ([intltravelnews.com](https://www.intltravelnews.com/2019/cemenelum-roman-city-french-riviera.html?utm_source=chatgpt.com), [bestofniceblog.com](https://www.bestofniceblog.com/what-to-see-in-nice/history-and-science-museums/roman-baths-ruins-cimiez-archeology-mueum/?utm_source=chatgpt.com)). From the beach one would see the white limestone walls of the bath complex rising among olive trees, steam curling from the caldarium’s hypocausts, while the rounded arches of the arena peered above nearby rooftops ([historytools.org](https://www.historytools.org/stories/journey-to-the-ancient-roman-heart-of-the-french-riviera-the-nice-cimiez-archaeological-museum?utm_source=chatgpt.com)).

Daily life on the beach would have contrasted sharply with today’s sunbathers and tourists. Instead, local Ligurian-Roman families in woolen tunics and leather sandals might gather at the water’s edge to wash wool or salt fish, while merchants in linen garments bartered amphorae of olive oil and imported wine on the shore. Slender fishermen’s boats plied the calm waters at dawn, their crews casting trammel nets into the blue-green depths.

Above all, the scene would have been one of a working Riviera: a blend of agricultural terraces, coastal trade, and provincial administration set within the timeless curve of sand and sea—far removed from the 21st-century bustle, yet already a crossroads of Mediterranean cultures under the Pax Romana.

"""


logfire.configure(
    service_name='my_agent_service',
    send_to_logfire=False,
)
logfire.instrument_openai_agents() 
langfuse = get_client()


twoBC_prompt = """Imagine how this image would look like in the 2nd BC."""

client = openai.OpenAI()  # uses OPENAI_API_KEY

JUDGE_PROMPT = """
You are a historic critic. 
You are provided with the description of scenes, a location and a year. 
Your job is to judge how plausible the items describes belong that place at that era. 

You must penalize items that are out-of-time.
Do not appraise the framing, the camera position or the camera technology. 

You must rate this "truthfullness" on a scale of 1 to 10 
and pricesely point out items that are out of time. 
"""

@observe(name="image_captionning", capture_input=False, as_type="generation")  
def image_caption(image, working_directory):

    if isinstance(image, np.ndarray):
        temp_image_path = os.path.join(working_directory, "temp_input_image.png")
        imageio.imwrite(temp_image_path, image)
        image_path = temp_image_path
    else:
        image_path = image

    response = client.responses.create(
        model="o4-mini-2025-04-16",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image, focusing on the human-maid items in the picture: buildings, roads, cloths,..."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                },
            ],
        }],
    )
    return response.output_text




@observe(name="llm_judge", capture_input=False, as_type="generation")  # creates a span; captures inputs/outputs automatically
def judge_answer(image_description, location, year):

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
            "role": "system",
            "content": [
                {"type": "input_text", "text": JUDGE_PROMPT},
            ],
            },
            {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f" image description : {image_description} . location : {location} . year : {year}"}
            ],
        }],
        text_format=ImageEvaluation
    )
    return json.loads(response.output_text)


@observe(name="image-generation", as_type="generation")
def generate_image(picture_design, input_image, working_directory):
    """
    Calls the Replicate API to generate an image based on the input image.
    Args:
        prompt (str): The text prompt.
    Returns:
        str: Path to the generated image.
    """
    # Gradio provides the image as a numpy array, but the replicate library expects a file path
    # So we save the numpy array as a temporary image file

    prompt = f"""
    You are an expert photoshop user.
    You are given a photo and you must transform it as to what it would have looked like at a certain time period.

    You must apply the changes described in: {picture_design}
    """

    if isinstance(input_image, np.ndarray):
        temp_image_path = "temp_input_image.png"
        imageio.imwrite(temp_image_path, input_image)
        input_image = temp_image_path


    with open(input_image, "rb") as image_file:
        output = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input={
                "prompt": prompt,
                "input_image": image_file,
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "safety_tolerance": 2,
                "prompt_upsampling": False
            }
        )
    num_images = len(os.listdir(working_directory))
    output_image_path = os.path.join(working_directory, f"output_{num_images}.jpg")
    print(f"Writing image in {output_image_path}")
    with open(output_image_path, "wb") as f:
        for chunk in output:
            f.write(chunk)
            
    return output_image_path

def create_rewind(image, text, date):
    """
    Processes the inputs from Gradio and generates an image and text.
    """
    prompt = f"{text} The scene is captured in the year {date}."
    generated_image_path = generate_image(prompt)
    
    output_text = f"This is the scene as it might have appeared in the year {date}."
    
    return generated_image_path, output_text

@observe(name="historical_grounding", as_type="generation")
def get_historical_grounding(image_description, location, year):
    instructions=f"""You are a historian. You are given the description of an image, a location and a time period.
    You must reflect on what the scenary would look like at the period.
    The nature of the scenary must remain unchanged : a seaside scenary remains a seaside scenary, a town center must remain a town center. 
    Be as historical accurate as possible about the items present in the image. Use the tools provided to search for images and look up information on internet.
    
    For the visual description of the item, you can use the tools you are provided as well.
    """
    response = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": f"image description: {image_description}, location: {location}, year: {year}",
            },
        ],
        text_format=HistoricalGrounding,
        tools=[{"type": "web_search_preview"}],
    )
    return response.output_text

def define_visual_cues_agent():
    visual_cues_agent = Agent(
    name="Visual Search Agent",
    instructions="""You search for visual cues to illustrate specific items within a specific time period.
    You provide a precise description of those items.
    You can use the tools provided to search for images and look up information on internet et specific images.""",
    model="gpt-5-mini-2025-08-07",
    tools=[seach_google_for_images, WebSearchTool()],
)
    return visual_cues_agent


def define_picture_designer_agent(image_path, working_directory):
    picture_designer_agent = Agent(
        name="Picture designer agent",
        instructions="""
        You are "picture designer" : you produce a text that will be used by an image generation tool
        You receive as input the description of an image and a historical grounding of what that scene would look like at certain period of time.  
        Your goal is to modify the items visible on the image in such a way, that it would plausible that this image has been taken at period of time. 

        For instance, if the image is a picture of the Eiffel tower in Paris and the period is 3th BC, obviously the eiffel tower should be replaced by something else.

        To help you in your task, you have access to the historical visual cue helper : this tool will provide you with precise descriptions of specific items, so that you can pricesely describe what to generate. 

        This text is to be interpreted by the image generation tool Replicate.
        Some items in the description might be flagged as violent (butcher's clever), sexual (prostitutes), unsanitary (wastes)
        Eliminate those possibly non-compliant items with the Replicate policy.
        """,
        model="o4-mini-2025-04-16",
        tools=[WebSearchTool()],
        handoffs=[define_visual_cues_agent()],
    )
    return picture_designer_agent


async def main(image_path, location, year) -> None:

    working_directory = tempfile.mkdtemp("_scene_rewind")
    print(f"Working in {working_directory}")

    with trace("LLM as a judge"):
        picture_description = image_caption(image_path, working_directory)
        historical_grounding = get_historical_grounding(picture_description, location, year)
        picture_designer_agent = define_picture_designer_agent(image_path, working_directory)
        

        input_items: list[TResponseInputItem] = [
            {
                "content": f"Description:{picture_description}\n Historical grounding: {historical_grounding}", 
                "role": "user"
            }]
        latest_outline: str | None = None
        

        while True:
            picture_design = await Runner.run(
                picture_designer_agent,
                input_items,
            )

            input_items = picture_design.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(picture_design.new_items)
            print("Story outline generated")

            try:
                output_path = generate_image(latest_outline, image_path, working_directory)
                created_image_caption = image_caption(output_path)
                judgment = judge_answer(created_image_caption, location, year)
            except Exception:
                judgment = {
                    "score": 0,
                    "feedback": "The image could not be produced as the content of the prompt as been flagged as sensitive by Replicate"
                } 

            print(f"Evaluator score: {judgment['score']}")

            if judgment["score"] > 6:
                print("Story outline is good enough, exiting.")
                break

            print("Re-running with feedback")

            input_items.append({"content": f"Feedback: {judgment['feedback']}", "role": "user"})

    print(f"Final story outline: {latest_outline}")

# Create Gradio interface
def create_interface():
    iface = gr.Interface(
        fn=main,  # Use wrapper function
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Textbox(label="Location/Prompt", placeholder="e.g., Paris, Times Square, etc."),
            gr.Slider(minimum=-2000, maximum=2000, value=1900, step=1, label="Target Year")
        ],
        outputs=[
            gr.Image(type="filepath", label="Historical Scene"),
            gr.Textbox(label="Historical Description")
        ],
        title="🕰️ Scene Rewind - Historical Time Travel",
        description="Upload an image of a location, specify the place, and select a year to see how it might have looked in the past!",
        examples=[
            ["images/paris.png", "Paris", 1700],
            ["images/newyork.jpg", "New York City", 1850],
        ] if False else None  # Set to True if you have example images
    )
    return iface


if __name__ == "__main__":
    # Option 1: Launch Gradio interface
    interface = create_interface()
    interface.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7860
    )