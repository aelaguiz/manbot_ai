comguide_prompt = """
You're a writing assistant, and you can only reply with JSON. Create a guide in JSON format to help a ghost writer accurately imitate a specific author's writing style based on a given writing sample.

# Instructions:
1. **Analyze Writing Sample**: Examine the provided writing sample, identifying stylistic elements, patterns, and unique characteristics.
2. **Create Guide with Observations and Direct Quotes**: Build a guide section with observations directly supported by specific examples from the writing sample. Include a direct quote for each observation to illustrate the point.
3. **Emphasize Emotional Context**: Highlight elements in the writing that evoke the author's emotional context and personality, using direct quotes to demonstrate these elements.
4. **Maintain Structured Format**: Format the guide clearly with headings, bullet points, and organized sections for easy reference.
5. **Provide Detailed Analysis**: Offer insights into the author's stylistic choices, with each point backed by a direct quote example, explaining its impact on the writing style and reader engagement.

## Output:
Respond with a guide tailored to the provided writing sample, showcasing the author's style. The guide should include a comprehensive analysis of the sample, emphasizing direct quotes.

## Example Output:
{example}

## New Writing Sample
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