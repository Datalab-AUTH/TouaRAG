import streamlit as st
import requests

# Initialize session state if not already set
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

# Default profile settings
DEFAULT_PROFILE = {
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

st.title("User Personalization")
st.write("Provide your details or use the default profile.")

with st.form("personalization_form"):
    profile = {
        "age_range": st.selectbox("Age Range", ["Young Adult", "Middle-aged", "Senior"]),
        "nationality": st.text_input("Nationality", "Greek"),
        "travel_group": st.selectbox("Travel Group", ["Solo Traveler", "Family", "Friends"]),
        "purpose_of_travel": st.selectbox("Purpose of Travel", ["Leisure", "Business", "Adventure"]),
        "travel_style": st.selectbox("Travel Style", ["Budget", "Luxury"]),
        "travel_frequency": st.selectbox("Travel Frequency", ["First-time Visitor", "Frequent Traveler"]),
        "comfort_with_local_culture": st.selectbox("Comfort with Local Culture", ["Seeks Familiar Experiences", "Eager to Explore"]),
        "preferred_activities": st.text_area("Preferred Activities", "Sightseeing"),
        "accessibility_needs": st.text_input("Accessibility Needs", "None"),
        "accommodation_style_preference": st.selectbox("Accommodation Style", ["Hotel", "Hostel", "Airbnb"]),
        "length_of_stay": st.selectbox("Length of Stay", ["Weekend", "1 Week", "2 Weeks"]),
        "preferred_season_or_climate": st.selectbox("Preferred Season or Climate", ["Summer", "Winter", "Autumn", "Spring"]),
        "social_media_and_photography": st.text_input("Social Media and Photography Interests", "Interest in Instagrammable Spots"),
        "local_interaction": st.selectbox("Comfortable with Local Language", ["Yes", "No"]),
        "interest_in_meeting_locals": st.selectbox("Interest in Meeting Locals", ["High", "Moderate", "Low"])
    }
    submitted = st.form_submit_button("Save Profile")
    if submitted:
        st.session_state["user_profile"] = profile
        
        # Send POST request to API
        try:
            response = requests.post(
                'http://127.0.0.1:8000/api/properties',
                headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                json=profile
            )
            if response.status_code == 200:
                st.success("Profile saved successfully and sent to API!")
            else:
                st.error(f"API request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error sending request to API: {str(e)}")

# If no profile has been set, initialize with the default profile
if st.session_state["user_profile"] is None:
    st.session_state["user_profile"] = DEFAULT_PROFILE

st.write("**Your Profile:**")
st.json(st.session_state["user_profile"])

