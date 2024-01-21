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

## Example Output:
{example}

## Current Guide Content
{comguide}

Respond with a new guide, enhanced with the latest insights from the writing samples provided. Always include the entire guide, preserving previous content to maintain continuity.

The guide should maintain the same JSON structure throughout its development.

## New Writing Samples
"""

merge_comguides_prompt = """Your task is to merge two comprehensive guides into a single, cohesive JSON document. The merged guide should preserve all unique content and nuances from both sources, especially writing samples, without any duplication. Carefully maintain the JSON structure throughout the process.

# Steps:
1. **Combine Sections in JSON Format**: For each section, merge content from both guides into a single JSON structure. 
2. **Remove Duplicates While Preserving JSON Integrity**: Identify and eliminate any duplicate information, ensuring the JSON format remains intact.
3. **Preserve Nuances and Direct Quotes**: Maintain all unique nuances and direct quotes from the writing samples in each guide. Every observation or point should be accompanied by a relevant direct quote from the original guides.
4. **Ensure Logical Flow in JSON**: Rearrange the combined content to ensure a logical, coherent flow within the JSON document.
5. **Maintain JSON Formatting Consistency**: Ensure that the formatting and style are consistent throughout the JSON document.
6. **Final JSON Review**: Review the merged JSON guide for completeness, clarity, and structural integrity.

# Output:
- Generate a single, comprehensive JSON guide that seamlessly integrates all relevant information from both sources.
- The JSON document should clearly present the combined content in a coherent and structured manner.
- Only include content from the two provided sources in the JSON format. Refrain from adding any external commentary, conclusions, or additional information.

Remember, the focus is on creating a well-structured JSON document that effectively merges the two guides while emphasizing the inclusion of all direct quotes and unique insights from the original content.
"""