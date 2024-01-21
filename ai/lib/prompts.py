comguide_prompt = """You're a writing assistant, and you can only reply with JSON. Create a comprehensive guide in JSON format to help a ghost writer accurately imitate a specific author's writing style. The guide will be progressively built, with each iteration incorporating additional insights from new writing samples.

# Instructions:
1. **Review Existing Guide Content**: Start by reviewing the current version of the style guide.
2. **Analyze New Writing Samples**: Examine new writing samples, identifying stylistic elements, patterns, and unique characteristics.
3. **Update Guide with Powerful Observations and Direct Quotes**: Each section of the guide should be updated with powerful observations directly supported by specific examples. Always provide a direct quote from the writing samples to illustrate each observation. For instance:
   - **Vivid Imagery**: "The city breathed life like a living beast, its pulse beating in the neon lights" – showcases the author's use of vivid imagery.
   - **Conversational Tone**: "Hey, isn't it funny how coffee tastes better on Fridays?" – illustrates the author's informal, conversational style.
   - **Metaphorical Language**: "Time is a thief, stealthily stealing our moments" – demonstrates the author's penchant for metaphorical expressions.
   - **Sharp Dialogue**: "No, I don't dance. I'm a tree in a world of dancers" – reflects the author's use of sharp, character-revealing dialogue.
   - **Humorous Descriptions**: "He was as subtle as a brick in a sock" – exemplifies the author's humorous and vivid descriptive style.
4. **Emphasize Emotional Context**: Highlight elements in the writing that evoke the author's emotional context and personality, using direct quotes to demonstrate these elements.
5. **Maintain Structured Format**: Ensure the guide is clearly structured with headings, bullet points, and organized sections for easy reference and readability.
6. **Provide Detailed Analysis**: Offer deeper insights into the author's stylistic choices, with each point backed by a direct quote example, explaining the impact on the overall writing style and reader engagement.
7. **Output Complete Guide**: Produce the full, updated guide, integrating new insights while retaining all previous content for continuity.

## Progression:
With each iteration, the guide will become more detailed and authoritative, serving as a comprehensive reference for emulating the author's style effectively.

## Current Guide Content
{comguide}

Respond with a new guide, enhanced with the latest insights from the writing samples provided. Always include the entire guide, preserving previous content to maintain continuity.

The guide should maintain the same JSON structure throughout its development.

## New Writing Samples
"""

merge_comguides_prompt = """Your task is to merge two comprehensive guides into a single, cohesive JSON document. Ensure all unique content and nuances, especially writing samples, from both guides are preserved without duplication. Maintain the JSON structure.


# Steps:
1. **Combine Sections**: For each section, combine content from both guides. 
2. **Remove Duplicates**: Eliminate any duplicate information found in both guides.
3. **Preserve Nuances and Examples**: Maintain all unique nuances and writing samples from each guide.
4. **Ensure Logical Flow**: Rearrange content to ensure a logical, coherent flow in the merged guide.
5. **Maintain Consistency**: Check for consistent formatting and style throughout.
6. **Final Review**: Review the merged guide for completeness and clarity.

# Output:
Create a single, comprehensive guide that includes all relevant information from both sources, presented clearly and coherently.
Only include content from the two sources, do not add any commentary, conclusion or other information.
"""