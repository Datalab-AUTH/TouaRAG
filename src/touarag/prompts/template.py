"""
Module for managing and selecting prompt templates for a tour agent application.
"""

from llama_index.core import PromptTemplate

TOUR_AGENT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Tour Agent in Greece.\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_TOUR_PROFILE = {
    "age_range": "Young Adult",
    "nationality": "Greek",
    "travel_group": "Solo Traveler",
    "purpose_of_travel": "Leisure",
    "travel_style": "Budget",
    "travel_frequency": "First-time Visitor",
    "comfort_with_local_culture": "Seeks Familiar Experiences",
    "preferred_activities": "Sightseeing",
    "accessibility_needs": "None",
    "accommodation_style_preference": "Hotel",
    "length_of_stay": "Weekend",
    "preferred_season_or_climate": "Summer",
    "social_media_and_photography": "Interest in Instagrammable Spots",
    "local_interaction": "Comfortable with Local Language",
    "interest_in_meeting_locals": "High"
}

PERSONALIZED_TOUR_AGENT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "---------------------\n"
    "Imagine you are the person with the below profile:\n"
    "Age Range: {age_range}, Nationality: {nationality}, Travel Group: {travel_group}, "
    "Purpose of Travel: {purpose_of_travel}, Travel Style: {travel_style}, Travel Frequency: {travel_frequency}, "
    "Comfort with Local Culture: {comfort_with_local_culture}, Preferred Activities: {preferred_activities}, "
    "Accessibility Needs: {accessibility_needs}, Accommodation Style Preference: {accommodation_style_preference}, "
    "Length of Stay: {length_of_stay}, Preferred Season or Climate: {preferred_season_or_climate}, "
    "Social Media and Photography: {social_media_and_photography}, Local Interaction: {local_interaction}, "
    "Interest in Meeting Locals: {interest_in_meeting_locals}\n"
    "---------------------\n"
    "Query: {query_str} \n"
    "You must carefully consider the profile in your suggestions but you don't need to mention it when speaking.\n"
    "The answer MUST be maximum 350 words long and MUST be concise and MUST be final.\n"
    "Answer: "
)

PERSONALIZED_TOUR_AGENT_KG_TMPL = (
    "---------------------\n"
    "Imagine you are the person with the below profile:\n"
    "Age Range: {age_range}, Nationality: {nationality}, Travel Group: {travel_group}, "
    "Purpose of Travel: {purpose_of_travel}, Travel Style: {travel_style}, Travel Frequency: {travel_frequency}, "
    "Comfort with Local Culture: {comfort_with_local_culture}, Preferred Activities: {preferred_activities}, "
    "Accessibility Needs: {accessibility_needs}, Accommodation Style Preference: {accommodation_style_preference}, "
    "Length of Stay: {length_of_stay}, Preferred Season or Climate: {preferred_season_or_climate}, "
    "Social Media and Photography: {social_media_and_photography}, Local Interaction: {local_interaction}, "
    "Interest in Meeting Locals: {interest_in_meeting_locals}\n"
    "---------------------\n"
    "Combine the following intermediate answers into a final, concise response.\n"
    "---------------------\n"
    "You must carefully the profile in your suggestions but you don't need to mention it when speaking.\n"
    "The answer MUST be maximum 350 words long and MUST be concise and MUST be final.\n"
)

class TourAgentPromptManager:
    """
    TourAgentPromptManager is a class responsible for managing and generating prompt 
    templates for a tour agent application.

    Attributes:
        template_type (str): The type of template to use. Defaults to "default".
        tour_profile (dict): The profile information of the tour, including gender, nationality, 
                             age, and occupation.
        templates (dict): A dictionary containing different types of templates.
        prompt_template (PromptTemplate): The formatted prompt template based on the tour profile.

    Methods:
        __init__(template_type="default", tour_profile=None):
            Initializes the TourAgentPromptManager with the given template type and tour profile.

        _get_prompt_template():
            Returns a partially formatted prompt template object.

        update_query_engine_template(query_engine):
            Returns the updated query engine instance.
    """
    def __init__(self, template_type="default", tour_profile=None):
        self.template_type = template_type
        self.tour_profile = tour_profile or DEFAULT_TOUR_PROFILE
        self.templates = {
            "default": TOUR_AGENT_TMPL,
            "personalized": PERSONALIZED_TOUR_AGENT_TMPL,
            "personalized_kg": PERSONALIZED_TOUR_AGENT_KG_TMPL
        }
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self):
        """
        Generates a formatted prompt template based on the tour profile.

        This method retrieves a template string based on the template type
        from the templates dictionary.
        It then formats this template string with the gender, nationality, 
        age, and occupation from the tour profile.

        Returns:
            PromptTemplate: A partially formatted prompt template object.
        """
        template_str = self.templates.get(self.template_type, TOUR_AGENT_TMPL)
        return PromptTemplate(template_str).partial_format(**self.tour_profile)

    def update_query_engine_template(self, query_engine):
        """
        Updates the query engine with the current prompt template.

        This method updates the query engine's prompts by setting the 
        "response_synthesizer:text_qa_template" key to the current instance's 
        prompt template.

        Args:
            query_engine: The query engine instance to be updated.

        Returns:
            The updated query engine instance.
        """
        if query_engine.get_prompts():
            query_engine.update_prompts(
                    {
                        "response_synthesizer:text_qa_template": self.prompt_template
                    }
            )
        return query_engine

    def get_prompt_text(self):
        """
        Returns the prompt text.

        Returns:
            str: The prompt text.
        """
        return self.prompt_template.format(**self.tour_profile)
