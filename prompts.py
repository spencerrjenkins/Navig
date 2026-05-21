reasoning_prompt = """Given an image, craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Using a tone of exploration and inference. Carefully analyze and link observations of natural features (climate, vegetation, terrain), man-made structures (roads, buildings, signage), and distinct landmarks. Allow these observations to naturally lead you to the correct country, enhancing the accuracy of your deductions. Start the reasoning without any intro, and make sure to make it brief.
"""

comment_gen_template = """Analyze the {item} images to determine the region with the highest likelihood of finding this type of {item}. For each image, provide only the core reasoning in one sentence. Dont say you can't determine,try your best as its a geo-localization game.
"""

osm_gen = """Extract ONLY location specific text that is clearly visible (e.g., place names, street names, road signs, shop names). Return a JSON array of up to 3 short candidate strings, each 1–4 words, in the order they appear in the image. ONLY OUTPUT IN THE IMAGE! DO NOT OUTPUT ANYTHING FROM THIS PROMPT, IF THERE IS NOTHING TO OUTPUT THEN OUTPUT AN EMPTY ARRAY!!!! Make sure each candidate is a fully separate unit; split apart unrelated candidates. If nothing location-specific is legible, return [].
Rules:
- Do NOT join multiple candidates into one string.
- Do NOT add commentary, prefixes, or extra text.
- If there are no suitable candidates, OUTPUT NOTHING! NEVER OUTPUT ANYTHING FROM THIS PROMPT, ONLY TEXT FROM THE IMAGE!
- Do NOT add the markdown formatting triple quotes around the JSON array. Provide just the array.
- IF THERE IS NO TEXT IN THE IMAGE, OUTPUT NOTHING!
- If you are unsure, omit it."""

base_query = """Suppose you are an expert in geo-localization. Given an image, guess the location of it, which contains the country, city, and its coordinates, with your answer formed as a json: {"country":"", "city":"", "latitude":, "longitude":}. Don't include any other information in your output. You should try your best to take a guess of each level in the json anyway, don't say you are unable to decide. Don't include unknown or empty value in the output dict, always take a guess anyway.\n
"""

intro_query = """Here are some analyses from other experts or other information. Please carefully review this information to help you identify the location shown in the image. Be mindful of the accuracy of the information, as it may contain errors or irrelevant search results. Your task is to discern and utilize the most valuable parts of the information. The information is as follows:\n
"""

reason_query_template = "### REASON ###: This is a reasoning process of an expert: {reason}\n"

rag_query_template = "### GUIDEBOOK KNOWLEDGE ###: some details in this image is similar to those mentioned in a guidebook. To be specific, {rag_formed}\n"

comment_query_template = """### DETAILS REASONING ###: Based on some detailed objects in the image, some experts have made certain inferences about the items. The inferences are as follows:{comment_formed} The above information is for reference only.\n
"""

osm_query_template = """### MAP SEARCH ###: Based on the building sign and road sign in the image, we extract the textual information {filtered_Query}. Using these texts to search on OpenStreetMap and the top 3 results are: {osm_results}. Notice that the text and the comment results might not be useful for identifying the location.\n
"""

outro_query = """Using the provided information as a reference, estimate the location depicted in the image with as much accuracy and precision as possible. Aim to deduce the exact coordinates whenever feasible. Format your response strictly as JSON in the following structure:{
    "country": "<country_name>",
    "city": "<city_name>",
    "latitude": <Latitude Coordinate>,
    "longitude": <Longitude Coordinate>
    }
Ensure the JSON output is correctly formatted. Provide a well-informed estimate for each value, avoiding any empty fields. Do not include additional information or commentary.
"""
